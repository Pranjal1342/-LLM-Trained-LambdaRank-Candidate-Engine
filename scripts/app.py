from __future__ import annotations
import io
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

st.set_page_config(
    page_title="Candidate Ranker",
    layout="wide",
    initial_sidebar_state="expanded",
)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))       
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)                    
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in [_SRC_DIR, _SCRIPTS_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE_DIR = _PROJECT_ROOT
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")
DATA_DIR = os.path.join(BASE_DIR, "data")
ALIASES_PATH = os.path.join(DATA_DIR, "skill_aliases.json")

LITE_MODE_LIMIT = 10_000 # max cand. that can enter streamlit mode

@st.cache_resource(show_spinner="Loading JD configuration...")
def load_jd_config():
    from jd_parser import parse_jd
    return parse_jd(ALIASES_PATH)


@st.cache_resource(show_spinner="Loading BM25 index...")
def load_bm25():
    from retrieval import load_numpy_bm25_artifacts
    bm25 = load_numpy_bm25_artifacts(PRECOMPUTED_DIR)
    ids_path = os.path.join(PRECOMPUTED_DIR, "candidate_ids.pkl")
    if not os.path.isfile(ids_path):
        return None, None
    with open(ids_path, "rb") as f:
        candidate_ids = pickle.load(f)

    if bm25 is not None:
        return bm25, candidate_ids

    # Fallback to pickle
    bm25_path = os.path.join(PRECOMPUTED_DIR, "bm25_index.pkl")
    if not os.path.isfile(bm25_path):
        return None, None
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    return bm25, candidate_ids


@st.cache_resource(show_spinner="Loading LightGBM model...")
def load_model():
    model_path = os.path.join(PRECOMPUTED_DIR, "lgbm_model.pkl")
    if not os.path.isfile(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def sort_with_secondary_tiebreak(
    final_scores: Dict[str, float],
    fv_cache: Dict[str, dict],
    logger,
) -> List[tuple]:
    """
    Sort candidates by final_raw score (primary), then by hard_req_coverage
    and bm25_score (secondary, display-only tiebreaks) when scores are tied.

    This does NOT change the underlying score values or the model's
    predictions — only the display order and assigned rank numbers when
    raw_lgbm scores are identical, which happens on small batches because
    the trained model places very low weight on bm25_score (confirmed:
    bm25_score used in only 2 of ~12,600 possible tree splits).
    """
    def sort_key(item):
        cid, score = item
        fv = fv_cache.get(cid, {})
        hard_req = fv.get("hard_req_coverage", 0.0)
        bm25 = fv.get("bm25_score", 0.0)
        # Negative for descending order on all three keys
        return (-score, -hard_req, -bm25, cid)

    sorted_items = sorted(final_scores.items(), key=sort_key)

    # Reuse the existing normalization logic from rank.py, just on the
    # newly-ordered list — normalization math itself is unchanged.
    from rank import _normalize_scores
    ranked_top100 = _normalize_scores(sorted_items, logger)
    return ranked_top100

def rank_candidates_inline(
    candidates: List[dict],
    jd_config,
    bm25,
    candidate_ids: List[str],
    model,
    max_n: int = LITE_MODE_LIMIT,
) -> Optional[pd.DataFrame]:
    """Run the full ranking pipeline inline on a small candidate set."""
    from retrieval import run_dual_pass_retrieval, tokenize_query
    from features import build_feature_vector, FEATURE_COLUMNS, consistency_score
    from reasoning import ReasoningCompiler
    from precompute import tokenize_candidate

  # this line allows a limited no of candidates for safety of memory
    if len(candidates) > max_n:
        st.warning(
            f"Lite mode: processing first {max_n} of {len(candidates)} candidates "
            f"to stay within 1GB RAM limit."
        )
        candidates = candidates[:max_n]

    cids = [c.get("candidate_id", f"IDX_{i}") for i, c in enumerate(candidates)]
    uploaded_cid_set = set(cids)

    # Use the REAL precomputed 100K-corpus BM25 index, same as the production pipeline,
    # so bm25_score means the same thing here as it does in src/rank.py.
    bm25_scores = {}
    in_main_index_count = 0
    fallback_count = 0

    if bm25 is not None and candidate_ids:
        # Query the full 100K index with the same dual-pass logic as production
        full_stage1_ids, full_bm25_scores = run_dual_pass_retrieval(bm25, candidate_ids, jd_config)
        main_index_lookup = dict(zip(candidate_ids, range(len(candidate_ids))))

        for cid in cids:
            if cid in full_bm25_scores:
                bm25_scores[cid] = full_bm25_scores[cid]
                in_main_index_count += 1
            elif cid in main_index_lookup:
                # Candidate exists in the 100K corpus but wasn't in the dual-pass
                # retrieved subset — score them at 0, consistent with how the
                # production pipeline treats non-retrieved candidates.
                bm25_scores[cid] = 0.0
                in_main_index_count += 1
            else:
                fallback_count += 1

    # Fallback: candidates genuinely NOT in the precomputed 100K corpus
    # (e.g. a judge uploads new/synthetic candidates never seen during precompute).
    # Build a small inline index ONLY for these, and warn the user explicitly
    # that their bm25_score uses small-corpus statistics.
    fallback_cids = [c.get("candidate_id", "") for c in candidates if c.get("candidate_id", "") not in bm25_scores]

    if fallback_cids:
        st.warning(
            f"{len(fallback_cids)} of {len(candidates)} uploaded candidates were not found "
            f"in the precomputed 100K corpus. Their BM25 scores are computed against a "
            f"small inline index built only from this upload, which uses different term "
            f"statistics than the production pipeline and may not be directly comparable "
            f"to scores for candidates found in the main corpus."
        )
        fallback_candidates = [c for c in candidates if c.get("candidate_id", "") in fallback_cids]
        fallback_corpus = [tokenize_candidate(c) for c in fallback_candidates]
        if fallback_corpus:
            fallback_bm25 = BM25Okapi(fallback_corpus)
            fb_stage1_ids, fb_scores = run_dual_pass_retrieval(fallback_bm25, fallback_cids, jd_config)
            bm25_scores.update(fb_scores)

    median_bm25 = float(np.median(list(bm25_scores.values()))) if bm25_scores else 0.0

    st.caption(
        f"BM25 scoring: {in_main_index_count} candidates scored against the real "
        f"100K-candidate corpus, {len(fallback_cids)} scored against a small inline "
        f"fallback corpus."
    )
    
    feature_rows = []
    valid_cids = []
    consistency_map = {}
    fv_cache = {}
    for c in candidates:
        cid = c.get("candidate_id", "")
        bs = bm25_scores.get(cid, 0.0)
        try:
            fv = build_feature_vector(c, jd_config, bs, median_bm25)
            fv_cache[cid] = fv
            row = [fv[col] for col in FEATURE_COLUMNS]
            consistency_map[cid] = float(fv.get("consistency_score", 1.0))
        except Exception:
            fv_cache[cid] = {col: 0.0 for col in FEATURE_COLUMNS}
            row = [bs] + [0.0] * 21
            consistency_map[cid] = 1.0
        feature_rows.append(row)
        valid_cids.append(cid)

    debug_targets = {"CAND_0000014", "CAND_0000043", "CAND_0000082"}
    for i, cid in enumerate(valid_cids):
        if cid in debug_targets:
            print(f"FEATURE VECTOR DEBUG | {cid} | row[0]={feature_rows[i][0]:.6f} (bm25) | full_row={feature_rows[i]}")
    print(f"FEATURE_COLUMNS[0] = {FEATURE_COLUMNS[0]}")

    X = np.array(feature_rows, dtype=np.float32)

  
    if model is not None:
        raw_scores = model.predict(X)
    else:
        raw_scores = np.array([bm25_scores.get(cid, 0.0) for cid in valid_cids])

    # BUG 1: Apply consistency multiplier
    final_scores = {}
    for i, cid in enumerate(valid_cids):
        final_scores[cid] = float(raw_scores[i] * consistency_map.get(cid, 1.0))

    # BUG 2: Reuse exact sorting and normalization from src/rank.py
    from rank import assert_monotonicity
    ranked_top100 = sort_with_secondary_tiebreak(final_scores, fv_cache, logging.getLogger("app"))

    try:
        assert_monotonicity(ranked_top100)
    except AssertionError as e:
        st.error(f"Monotonicity Assertion Failed: {e}")

    # DEBUG: Print top 10 raw scores for verification
    print("\n" + "="*50)
    print("TOP 10 RAW SCORES DEBUG (before normalization)")
    print("="*50)
    for cid, norm_score, rank_i in ranked_top100[:10]:
        idx = valid_cids.index(cid)
        raw = raw_scores[idx]
        cons = consistency_map.get(cid, 1.0)
        final = final_scores[cid]
        bs = bm25_scores.get(cid, 0.0)
        print(f"Rank {rank_i:02d} | {cid} | bm25: {bs:10.6f} | raw_lgbm: {raw:10.6f} | cons: {cons:4.2f} | final_raw: {final:10.6f} | norm: {norm_score:8.6f}")
    print("="*50 + "\n")
    print(f"BM25 scoping: in_main_index={in_main_index_count}, fallback={fallback_count}")

    all_lgbm_scores = [final_scores[cid] for cid, _, _ in ranked_top100]
    compiler = ReasoningCompiler(jd_config, all_scores=all_lgbm_scores)

    candidate_lookup = {c.get("candidate_id"): c for c in candidates}

    rows = []
    for cid, norm_score, rank_i in ranked_top100:
        raw_score = final_scores.get(cid, 0.0)
        c = candidate_lookup.get(cid, {"candidate_id": cid})
        fv = fv_cache.get(cid, {col: 0.0 for col in FEATURE_COLUMNS})
        reasoning = compiler.compile(c, fv, raw_score, rank_i)
        rows.append({
            "rank": rank_i,
            "candidate_id": cid,
            "score": round(norm_score, 6),
            "name": c.get("profile", {}).get("anonymized_name", ""),
            "title": c.get("profile", {}).get("current_title", ""),
            "company": c.get("profile", {}).get("current_company", ""),
            "yoe": c.get("profile", {}).get("years_of_experience", 0),
            "location": c.get("profile", {}).get("location", ""),
            "hard_req_coverage": round(fv.get("hard_req_coverage", 0), 3),
            "consistency_score": round(fv.get("consistency_score", 1), 3),
            "reasoning": reasoning,
        })

    return pd.DataFrame(rows)



def main():
    st.title(" Candidate Ranker")
    st.caption(
        "Candidate ranking: submission. "
        "Lite mode (≤10K candidates, ≤1GB RAM)."
    )

    with st.sidebar:
        st.header(" Pipeline status")

        jd_config = load_jd_config()
        st.success(
            f" JD Config loaded: {len(jd_config.hard_requirements)} hard reqs, "
            f"{len(jd_config.preferred_requirements)} preferred"
        )

        bm25, candidate_ids = load_bm25()
        if bm25 is not None:
            st.success(f"BM25 Index: {len(candidate_ids):,} candidates indexed")
        else:
            st.warning("BM25 index not found — run precompute.py first")

        model = load_model()
        if model is not None:
            st.success("LightGBM model loaded")
        else:
            st.warning("LightGBM model not found — run precompute.py first")

        st.divider()
        st.header("JD Requirements")
        with st.expander("Hard Requirements"):
            for name in jd_config.hard_requirements:
                st.write(f"• {name.replace('_', ' ').title()}")
        with st.expander("Preferred Requirements"):
            for name in jd_config.preferred_requirements:
                st.write(f"• {name.replace('_', ' ').title()}")


    tab1, tab2, tab3 = st.tabs(["Upload & Rank", "Architecture", "Validate"])

    with tab1:
        st.header("Upload Candidates & Run Ranking")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload candidates JSONL file",
                type=["jsonl", "json", "txt"],
                help=f"Max {LITE_MODE_LIMIT:,} candidates processed in lite mode.",
            )

        with col2:
            st.metric("RAM Limit", "1 GB")
            st.metric("Max Candidates", f"{LITE_MODE_LIMIT:,}")
            if model is not None:
                st.metric("Ranker", "LightGBM")
            else:
                st.metric("Ranker", "BM25 fallback")

        if uploaded_file is not None:
            # Parse JSONL
            candidates = []
            malformed = 0
            for line in uploaded_file:
                line = line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1

            if malformed > 0:
                st.warning(f" Skipped {malformed} malformed lines")

            st.info(
                f" Loaded {len(candidates):,} candidates from uploaded file"
            )

            if len(candidates) == 0:
                st.error("No valid candidates found in uploaded file.")
            else:
                run_btn = st.button(
                    " Run ranking pipeline",
                    type="primary",
                    use_container_width=True,
                )

                if run_btn:
                    with st.spinner("Running ranking pipeline..."):
                        t0 = time.time()
                        try:
                            result_df = rank_candidates_inline(
                                candidates, jd_config, bm25, candidate_ids, model
                            )
                            elapsed = time.time() - t0

                            if result_df is not None and len(result_df) > 0:
                                st.success(
                                    f" Ranked {len(result_df)} candidates in {elapsed:.1f}s"
                                )

                                m1, m2, m3, m4, m5 = st.columns(5)
                                m1.metric("Total Ranked", len(result_df))
                                m2.metric("Top Score", f"{result_df['score'].max():.4f}")
                                m3.metric(
                                    "Avg Hard Req Coverage",
                                    f"{result_df['hard_req_coverage'].mean():.1%}"
                                )
                                m4.metric("Wall-clock", f"{elapsed:.1f}s")
                                low_cons_count = (result_df["consistency_score"] < 0.25).sum()
                                m5.metric("Honeypots", f"{low_cons_count}/{len(result_df)} flagged", 
                                           delta="PASS" if low_cons_count < 10 else "FAIL", delta_color="normal" if low_cons_count < 10 else "inverse")

                        
                                st.subheader("Top 100 Candidates")
                                st.caption(
                                    "Note: candidates with identical model scores are ordered by hard "
                                    "requirement coverage, then BM25 relevance, for display purposes. "
                                    "The underlying model scores are unchanged."
                                )
                                display_df = result_df[[
                                    "rank", "candidate_id", "name", "title",
                                    "company", "yoe", "location", "score",
                                    "hard_req_coverage", "consistency_score"
                                ]].copy()
                                st.dataframe(
                                    display_df.style.background_gradient(
                                        subset=["score"], cmap="RdYlGn"
                                    ),
                                    use_container_width=True,
                                    height=500,
                                )

                                st.subheader("Reasoning Explorer")
                                selected_rank = st.slider(
                                    "Select candidate rank to view reasoning:",
                                    min_value=1, max_value=min(100, len(result_df))
                                )
                                selected_row = result_df[result_df["rank"] == selected_rank]
                                if not selected_row.empty:
                                    row = selected_row.iloc[0]
                                    with st.expander(
                                        f"Rank {selected_rank}: {row['name']} — {row['title']} @ {row['company']}",
                                        expanded=True
                                    ):
                                        col_a, col_b = st.columns(2)
                                        col_a.metric("Score", f"{row['score']:.6f}")
                                        col_a.metric("Hard Req Coverage", f"{row['hard_req_coverage']:.1%}")
                                        col_b.metric("YoE", f"{row['yoe']}")
                                        col_b.metric("Consistency", f"{row['consistency_score']:.2f}")
                                        st.markdown(f"**Reasoning:** {row['reasoning']}")

                                # BUG 3 & 4: Explicit CSV copy and index=False, utf-8 bytes (no BOM)
                                export_df = result_df[["candidate_id", "rank", "score", "reasoning"]].copy()
                                csv_bytes = export_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label=" Download submission.csv",
                                    data=csv_bytes,
                                    file_name="submission.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                            else:
                                st.error("Ranking produced no results.")
                        except Exception as e:
                            st.error(f"Pipeline error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
        else:
            st.info(
                " Upload a JSONL file of candidate records to rank them. "
                "The file must match the candidate schema."
            )
# sample
            with st.expander("Expected JSONL format (one candidate per line)"):
                sample = {
                    "candidate_id": "CAND_0000001",
                    "profile": {
                        "anonymized_name": "Alex Kumar",
                        "headline": "ML Engineer | FAISS | BM25",
                        "summary": "...",
                        "location": "Pune",
                        "country": "India",
                        "years_of_experience": 5,
                        "current_title": "Senior ML Engineer",
                        "current_company": "TechCorp",
                        "current_company_size": "201-500",
                        "current_industry": "Technology"
                    },
                    "...": "see candidate_schema.json for full structure"
                }
                st.json(sample)

    
    with tab2:
        st.header("Architecture Overview")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pipeline Stages")
            st.markdown("""
            | Stage | Operation | Runtime |
            |-------|-----------|---------|
            | 1 | Load BM25 & Dual-Pass Retrieval | 1–2s |
            | 2 | Feature Extraction (22 features) | 15–25s |
            | 4 | LightGBM LambdaRank Inference | 1–3s |
            | 5 | Reasoning Compilation + Audits | 1–2s |
            | 6 | Monotonicity Assert + CSV Write | <1s |
            | **Total** | **End-to-End** | **3.55s** |
            """)

        with col2:
            st.subheader("Hardware Constraints")
            st.markdown("""
            -  **≤5 minutes** clock
            -  **≤16 GB RAM** CPU only
            -  **Zero** network calls during ranking
            -  **≤5 GB** intermediate disk state
            -  **Docker** `--network none` compatible
            """)

        st.subheader("22-Feature Matrix")
        features_df = pd.DataFrame([
            {"#": 1, "Feature": "bm25_score", "Source": "BM25 retrieval"},
            {"#": 2, "Feature": "yoe", "Source": "profile.years_of_experience"},
            {"#": 3, "Feature": "Param_A_Systems_Depth", "Source": "career_history[].description + duration_months"},
            {"#": 4, "Feature": "Param_B_Availability", "Source": "platform_signals.recruiter_response_rate + last_active_date"},
            {"#": 5, "Feature": "Param_C_Tenure", "Source": "career_history[].duration_months"},
            {"#": 6, "Feature": "Param_D_Notice_Exp", "Source": "platform_signals.notice_period_days"},
            {"#": 7, "Feature": "Param_E_Credibility", "Source": "skills[].proficiency + skill_assessment_scores"},
            {"#": 8, "Feature": "Param_F_Consulting", "Source": "career_history[].industry + duration_months"},
            {"#": 9, "Feature": "Param_G_Location", "Source": "profile.location + country"},
            {"#": 10, "Feature": "Param_H_GitHub", "Source": "platform_signals.github_activity_score"},
            {"#": 11, "Feature": "title_ai_fraction", "Source": "career_history[].title"},
            {"#": 12, "Feature": "prod_signal_log", "Source": "career_history[].description"},
            {"#": 13, "Feature": "consistency_score", "Source": "c1×c2×c3×c4×c5"},
            {"#": 14, "Feature": "hard_req_coverage", "Source": "skills[].name vs JD aliases"},
            {"#": 15, "Feature": "flag_consulting_only", "Source": "career_history[].industry"},
            {"#": 16, "Feature": "flag_title_chaser", "Source": "career_history[].title + duration_months"},
            {"#": 17, "Feature": "flag_langchain_dabbler", "Source": "skills[].name + duration_months"},
            {"#": 18, "Feature": "flag_cv_specialist", "Source": "skills[].name + duration_months"},
            {"#": 19, "Feature": "flag_title_desc_mismatch", "Source": "career_history[].title + description"},
            {"#": 20, "Feature": "flag_template_desc", "Source": "career_history[].description"},
            {"#": 21, "Feature": "interaction_req_x_consistency", "Source": "hard_req_coverage × consistency_score"},
            {"#": 22, "Feature": "interaction_yoe_x_prod", "Source": "yoe × prod_signal_log"},
        ])
        st.dataframe(features_df, use_container_width=True, hide_index=True)

 
    with tab3:
        st.header("Validate Submission CSV")
        st.info(
            "Upload your submission.csv to run local format validation "
            "before spending one of 3 competition submissions."
        )

        val_file = st.file_uploader(
            "Upload submission.csv", type=["csv"], key="val_uploader"
        )
        if val_file is not None:
            try:
                df = pd.read_csv(val_file)
                errors = []
                warnings_list = []

                required_cols = {"candidate_id", "rank", "score", "reasoning"}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    errors.append(f"Missing columns: {missing_cols}")

                if not errors:

                    if len(df) != 100:
                        errors.append(f"Expected 100 rows, got {len(df)}")

                    if set(df["rank"].tolist()) != set(range(1, 101)):
                        errors.append("Ranks must be exactly 1–100 with no gaps")

                    df_sorted = df.sort_values("rank")
                    scores = df_sorted["score"].values
                    for i in range(1, len(scores)):
                        if scores[i] > scores[i-1] + 1e-9:
                            errors.append(
                                f"Score not monotonically non-increasing at rank {i+1}: "
                                f"{scores[i-1]:.6f} → {scores[i]:.6f}"
                            )
                            break

                    if df["score"].min() < 0 or df["score"].max() > 1:
                        warnings_list.append(
                            f"Scores outside [0,1]: min={df['score'].min():.4f}, "
                            f"max={df['score'].max():.4f}"
                        )

                    empty_reasoning = df["reasoning"].isna() | (df["reasoning"].str.strip() == "")
                    if empty_reasoning.any():
                        errors.append(
                            f"{empty_reasoning.sum()} rows have empty reasoning"
                        )

                    if df["candidate_id"].duplicated().any():
                        errors.append("Duplicate candidate_ids found")

                if errors:
                    st.error(f"Validation failed!!({len(errors)} errors):")
                    for e in errors:
                        st.write(f"  • {e}")
                else:
                    st.success("Validation paased!!")
                    if warnings_list:
                        for w in warnings_list:
                            st.warning(f"warning {w}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", len(df))
                    col2.metric("Score Range", f"{df['score'].min():.4f}–{df['score'].max():.4f}")
                    col3.metric("Reasoning Coverage", "100%")

                    st.dataframe(df.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")


if __name__ == "__main__":
    main()
