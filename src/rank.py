from __future__ import annotations
import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple                                                                            
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))                   
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                             
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
for _p in [_SRC_DIR, _SCRIPTS_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

def setup_logging(base_dir: str) -> logging.Logger:
    """Set up file + console logging."""
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"rank_{timestamp}.log")

    logger = logging.getLogger("rank")
    logger.setLevel(logging.DEBUG)

                                
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    ))

                                 
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger

def load_artifacts(precomputed_dir: str, logger: logging.Logger):
    """Load BM25 scorer, candidate IDs, and LightGBM model.

    Tries fast NumPy / native-format artifacts first; falls back to pickle
    if the fast artifacts haven't been built yet (backward-compatible).
    """
                                                                           
    from retrieval import load_numpy_bm25_artifacts
    bm25 = load_numpy_bm25_artifacts(precomputed_dir)
    if bm25 is not None:
        logger.info("Stage 0: NumpyBM25 loaded (fast path)")
    else:
        bm25_path = os.path.join(precomputed_dir, "bm25_index.pkl")
        if not os.path.isfile(bm25_path):
            logger.error("Missing artifact: %s — run precompute.py first", bm25_path)
            sys.exit(1)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        logger.info("Stage 0: BM25Okapi loaded (legacy pickle path)")

                                                                           
    ids_path = os.path.join(precomputed_dir, "candidate_ids.pkl")
    if not os.path.isfile(ids_path):
        logger.error("Missing artifact: %s — run precompute.py first", ids_path)
        sys.exit(1)
    with open(ids_path, "rb") as f:
        candidate_ids = pickle.load(f)

                                                                            
    lgbm_txt = os.path.join(precomputed_dir, "lgbm_model.txt")
    lgbm_pkl = os.path.join(precomputed_dir, "lgbm_model.pkl")
    model = None
    if os.path.isfile(lgbm_txt):
        try:
            import lightgbm as lgb
            t0 = time.time()
            model = lgb.Booster(model_file=lgbm_txt)
            logger.info("Stage 0: LightGBM loaded from native text (%.2f s)", time.time() - t0)
        except Exception as exc:
            logger.warning("lgbm native load failed (%s), falling back to pickle", exc)
    if model is None:
        if not os.path.isfile(lgbm_pkl):
            logger.error("Missing artifact: %s — run precompute.py first", lgbm_pkl)
            sys.exit(1)
        with open(lgbm_pkl, "rb") as f:
            model = pickle.load(f)
        logger.info("Stage 0: LightGBM loaded from pickle (legacy path)")

                                                                           
    static_path = os.path.join(precomputed_dir, "static_features.pkl")
    static_features = None
    if os.path.isfile(static_path):
        try:
            t0 = time.time()
            with open(static_path, "rb") as f:
                static_features = pickle.load(f)
            logger.info("Stage 0: Loaded static features (%d candidates) in %.2fs", len(static_features), time.time() - t0)
        except Exception as exc:
            logger.warning("static_features.pkl load failed (%s), falling back to live calculation", exc)
    else:
        logger.warning("static_features.pkl not found — falling back to live calculation")

    logger.info(
        "Artifacts loaded: BM25 scorer (%s, %d candidates), LightGBM model",
        type(bm25).__name__,
        len(candidate_ids),
    )
    return bm25, candidate_ids, model, static_features
                                                                        

def load_stage1_candidates(
    candidates_path: str,
    stage1_ids: List[str],
    logger: logging.Logger,
) -> Tuple[List[dict], int]:
    """
    Stream-read candidates.jsonl and return only Stage 1 candidates.
    Defensive against malformed records, missing fields, null values.

    Returns:
        (candidate_list, malformed_count)
    """
    stage1_set = set(stage1_ids)
    found: Dict[str, dict] = {}
    malformed_count = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                malformed_count += 1
                logger.warning("Malformed JSON at line %d: %s", line_num, e)
                continue

            cid = c.get("candidate_id")
            if cid and cid in stage1_set:
                found[cid] = c
                if len(found) == len(stage1_set):
                    break                             

    if malformed_count > 0:
        logger.warning("Skipped %d malformed JSONL lines during loading", malformed_count)

    missing = stage1_set - set(found.keys())
    if missing:
        logger.warning(
            "%d stage1 candidates not found in JSONL: %s...",
            len(missing), list(missing)[:5]
        )

                                                             
    ordered = [found[cid] for cid in stage1_ids if cid in found]
    logger.info(
        "Loaded %d stage1 candidates (%d missing, %d malformed)",
        len(ordered), len(missing), malformed_count
    )
    return ordered, malformed_count

def load_stage1_candidates_fast(
    candidates_path: str,
    stage1_ids: List[str],
    offsets: Dict[str, int],
    logger: logging.Logger,
) -> Tuple[List[dict], int]:
    """
    Load Stage 1 candidate records using a precomputed byte-offset index.

    Instead of streaming all 487 MB of candidates.jsonl, performs one
    f.seek() + f.readline() per candidate.  For ~8500 candidates this
    reads ~43 MB total instead of 487 MB, reducing Stage 2 from ~4 s
    to ~0.1–0.3 s.

    Returns:
        (candidate_list, malformed_count)
    """
    ordered: List[dict] = []
    malformed_count = 0
    missing: List[str] = []

    with open(candidates_path, "rb") as f:
        for cid in stage1_ids:
            offset = offsets.get(cid)
            if offset is None:
                missing.append(cid)
                continue
            f.seek(offset)
            raw = f.readline()
            try:
                c = json.loads(raw.decode("utf-8", errors="ignore").strip())
                ordered.append(c)
            except json.JSONDecodeError as exc:
                logger.warning("Malformed record at offset %d for %s: %s", offset, cid, exc)
                malformed_count += 1

    if missing:
        logger.warning(
            "%d stage1 candidates not in offset index: %s ...",
            len(missing), missing[:5],
        )
    logger.info(
        "Loaded %d stage1 candidates via offset index (%d missing, %d malformed)",
        len(ordered), len(missing), malformed_count,
    )
    return ordered, malformed_count


def extract_features_for_ranking(
    candidates: List[dict],
    jd_config,
    bm25_scores: Dict[str, float],
    stage1_bm25_median: float,
    logger: logging.Logger,
    static_features: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """
    Extract the 22-feature matrix for all Stage 1 candidates.

    Returns:
        (X: np.ndarray[N, 22], ordered_ids: List[str], consistency_map: Dict[str, float])
    """
    from features import build_feature_vector, FEATURE_COLUMNS

    feature_rows = []
    ordered_ids = []
    consistency_map = {}
    failed_count = 0

    for candidate in candidates:
        cid = candidate.get("candidate_id", "UNKNOWN")
        bm25_score = bm25_scores.get(cid, 0.0)

        try:
            fv = build_feature_vector(
                candidate, jd_config,
                bm25_score=bm25_score,
                stage1_bm25_median=stage1_bm25_median,
                precomputed_static=static_features.get(cid) if static_features else None
            )
            row = [fv[col] for col in FEATURE_COLUMNS]
            consistency_map[cid] = float(fv.get("consistency_score", 1.0))
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", cid, e)
            row = [0.0] * len(FEATURE_COLUMNS)
            consistency_map[cid] = 1.0
            failed_count += 1

        feature_rows.append(row)
        ordered_ids.append(cid)

    if failed_count > 0:
        logger.warning("Feature extraction failed for %d candidates (zeroed out)", failed_count)

    X = np.array(feature_rows, dtype=np.float32)
    logger.info("Feature matrix: shape=%s", X.shape)
    return X, ordered_ids, consistency_map

def run_lightgbm_inference(
    model,
    X: np.ndarray,
    ordered_ids: List[str],
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Run LightGBM predict on the feature matrix.

    Returns:
        {candidate_id: lgbm_score}
    """
    t0 = time.time()
    raw_scores = model.predict(X)
    elapsed = time.time() - t0
    logger.info(
        "LightGBM inference: %d candidates in %.2fs", len(ordered_ids), elapsed
    )
    return {cid: float(score) for cid, score in zip(ordered_ids, raw_scores)}

def _normalize_scores(top_100_raw: List[Tuple[str, float]], logger: logging.Logger) -> List[Tuple[str, float, int]]:
    """
    Apply min-max normalization to raw scores and assign ranks 1..N.
    Returns: List of (candidate_id, score, rank) sorted by rank.
    """
    if not top_100_raw:
        return []

    top_scores = [s for _, s in top_100_raw]
    score_min = top_scores[-1]                      
    score_max = top_scores[0]                      
    score_range = score_max - score_min

    result = []
    prev_normalized = None

    for rank, (cid, raw_score) in enumerate(top_100_raw, 1):
        if score_range > 0:
            normalized = 0.01 + 0.99 * (raw_score - score_min) / score_range
        else:
            normalized = 1.0 - (rank - 1) / 99.0

                                
        if prev_normalized is not None and normalized > prev_normalized + 1e-9:
                                                              
            logger.error("MONOTONICITY VIOLATION at rank %d", rank)
        prev_normalized = normalized
        result.append((cid, normalized, rank))

    logger.info("Top 100 selected: score range [%.6f, %.6f]",
                result[-1][1], result[0][1])
    return result

def sort_and_enforce_monotonicity(
    lgbm_scores: Dict[str, float],
    logger: logging.Logger,
) -> List[Tuple[str, float, int]]:
    """
    Sort candidates by score descending. Break ties by ascending candidate_id.
    Assign ranks 1..N.

    Returns:
        List of (candidate_id, score, rank) sorted by rank.
    """
                                                                                         
    sorted_candidates = sorted(
        lgbm_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

                                                          
    top_100_raw = sorted_candidates[:100]
    return _normalize_scores(top_100_raw, logger)


def assert_monotonicity(ranked: List[Tuple[str, float, int]]) -> None:
    """
    Explicit runtime assertion: scores must be monotonically non-increasing by rank.
    This runs BEFORE writing the CSV — not just by sorting and hoping.

    Raises AssertionError if violated.
    """
    for i in range(1, len(ranked)):
        prev_score = ranked[i-1][1]
        curr_score = ranked[i][1]
        assert curr_score <= prev_score + 1e-9, (
            f"Monotonicity violation: rank {i} score {prev_score:.8f} "
            f"< rank {i+1} score {curr_score:.8f}"
        )                                                                       

def run_honeypot_audit(
    top_100_candidates: List[dict],
    feature_vectors: Dict[str, dict],
    logger: logging.Logger,
) -> None:
    """
    Section 8.1: Pre-Submission Honeypot Audit.
    assert count(consistency_score < 0.25 in top_100) < 10.

    If this assertion fails, rank.py exits non-zero.
    """
    low_consistency_count = sum(
        1 for c in top_100_candidates
        if feature_vectors.get(c.get("candidate_id", ""), {}).get("consistency_score", 1.0) < 0.25
    )

    logger.info(
        "Honeypot audit: %d of 100 candidates have consistency_score < 0.25",
        low_consistency_count
    )

    if low_consistency_count >= 10:
        logger.error(
            "HONEYPOT AUDIT FAILED: %d candidates with consistency_score < 0.25 "
            "(threshold: < 10). Pipeline is broken — honeypots bypassed filters.",
            low_consistency_count
        )
        sys.exit(2)

    logger.info("Honeypot audit PASSED.")


def run_diversity_audit(
    top_100_candidates: List[dict],
    feature_vectors: Dict[str, dict],
    logger: logging.Logger,
) -> None:
    """
    Section 8.2: Top 100 Diversity & Homogeneity Audit.
    Uses validate_pipeline.check_top100_diversity.

    If the check fails, rank.py exits non-zero with a clear error.
    This is a BLOCKING check — not just a warning.
    """
    from validate_pipeline import check_top100_diversity, print_diversity_report

    report = check_top100_diversity(
        top_100_candidates,
        feature_vectors,
        max_signature_share=0.25,
        max_single_company_share=0.30,                                        
    )

    print_diversity_report(report)
    logger.info(
        "Diversity audit: %d distinct archetypes, max_company=%.1f%%, max_sig=%.1f%%",
        report["n_distinct_signatures"],
        report["most_common_company_share"] * 100,
        report["most_common_signature_share"] * 100,
    )

    if not report["pass"]:
        if report["flagged_companies"]:
            logger.error(
                "DIVERSITY AUDIT FAILED: company concentration too high: %s",
                report["flagged_companies"]
            )
        if report["flagged_signatures"]:
            logger.error(
                "DIVERSITY AUDIT FAILED: archetype signature concentration too high: %s",
                report["flagged_signatures"]
            )
        sys.exit(3)

    logger.info("Diversity audit PASSED.")                                                                       

def write_reasoning_trace(
    top_30_traces: List[dict],
    base_dir: str,
    logger: logging.Logger,
) -> None:
    """Write reasoning_trace.jsonl for top 30 candidates."""
    trace_path = os.path.join(base_dir, "reasoning_trace.jsonl")
    with open(trace_path, "w", encoding="utf-8") as f:
        for trace in top_30_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    logger.info("Reasoning trace written: %s (%d entries)", trace_path, len(top_30_traces))
def pipeline_fn(
    candidates: List[dict],
    jd_config,
    disable_consistency: bool = False,
    disable_param_a: bool = False,
    disable_features: bool = False,
) -> List[str]:
    """
    Pipeline function compatible with validate_pipeline.run_ablation.
    Accepts a list of candidate dicts + jd_config, returns ranked candidate_ids.

    This runs the full in-memory pipeline (for small candidate sets).
    """
    from features import build_feature_vector, FEATURE_COLUMNS, consistency_score
    from precompute import tokenize_candidate

                                
    corpus = [tokenize_candidate(c) for c in candidates]
    bm25 = BM25Okapi(corpus)
    cids = [c.get("candidate_id", f"IDX_{i}") for i, c in enumerate(candidates)]

               
    from retrieval import tokenize_query
    query_tokens = tokenize_query(jd_config.get_all_query_terms() + jd_config.production_keywords)
    raw_scores = bm25.get_scores(query_tokens)
    bm25_scores = {cids[i]: float(raw_scores[i]) for i in range(len(cids))}
    median_bm25 = float(np.median(list(bm25_scores.values())))
      
    feature_rows = []
    for c in candidates:
        cid = c.get("candidate_id", "")
        bs = bm25_scores.get(cid, 0.0)

        if disable_features:
            row = [bs] + [0.0] * 21
        else:
            try:
                fv = build_feature_vector(c, jd_config, bs, median_bm25)
                if disable_consistency:
                    fv["consistency_score"] = 1.0
                if disable_param_a:
                    fv["Param_A_Systems_Depth"] = 0.0
                row = [fv[col] for col in FEATURE_COLUMNS]
            except Exception:
                row = [bs] + [0.0] * 21

        feature_rows.append(row)                                                                     
    try:
        import pickle
        base = _PROJECT_ROOT
        with open(os.path.join(base, "precomputed", "lgbm_model.pkl"), "rb") as f:
            model = pickle.load(f)
        X = np.array(feature_rows, dtype=np.float32)
        scores = model.predict(X)
    except Exception:
                                            
        scores = np.array([bm25_scores.get(cid, 0.0) for cid in cids])

    ranked = sorted(
        zip(cids, scores.tolist()),
        key=lambda x: (-x[1], x[0])
    )
    return [cid for cid, _ in ranked]
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Candidate Ranking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out",
        default="./CTRL_COFFEE_REPEAT.csv",
        help="Path for output CTRL_COFFEE_REPEAT.csv",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory (defaults to directory containing rank.py)",
    )
    args = parser.parse_args()

                                                                             
                                                                          
    script_dir = _PROJECT_ROOT
    base_dir = os.path.abspath(args.base_dir) if args.base_dir else script_dir
    candidates_path = os.path.abspath(args.candidates)
    out_path = os.path.abspath(args.out)
    precomputed_dir = os.path.join(base_dir, "precomputed")

    logger = setup_logging(base_dir)
    wall_start = time.time()

    logger.info("=" * 60)
    logger.info("RANKING PIPELINE")
    logger.info("Candidates: %s", candidates_path)
    logger.info("Output: %s", out_path)
    logger.info("Base dir: %s", base_dir)
    logger.info("=" * 60)                                                                       
    t0 = time.time()
    bm25, candidate_ids, model, static_features = load_artifacts(precomputed_dir, logger)
    logger.info("Stage 0 (load artifacts): %.2fs", time.time() - t0)                         
                                                                             
    t1 = time.time()
    from jd_parser import parse_jd
    from retrieval import run_dual_pass_retrieval

    jd_config = parse_jd(os.path.join(base_dir, "data", "skill_aliases.json"))
    stage1_ids, bm25_scores = run_dual_pass_retrieval(bm25, candidate_ids, jd_config)

    stage1_bm25_scores_list = list(bm25_scores.values())
    stage1_bm25_median = float(np.median(stage1_bm25_scores_list))

    logger.info(
        "Stage 1 (retrieval): %d candidates retrieved, median BM25=%.4f in %.2fs",
        len(stage1_ids), stage1_bm25_median, time.time() - t1
    )                                                                   
    t2 = time.time()                                                                            
    offsets_path = os.path.join(precomputed_dir, "candidate_offsets.pkl")
    if os.path.isfile(offsets_path):
        with open(offsets_path, "rb") as f:
            candidate_offsets = pickle.load(f)
        stage1_candidates, malformed_count = load_stage1_candidates_fast(
            candidates_path, stage1_ids, candidate_offsets, logger
        )
    else:
                                                             
        logger.info("Stage 2: offset index not found — streaming full JSONL (slow)")
        stage1_candidates, malformed_count = load_stage1_candidates(
            candidates_path, stage1_ids, logger
        )

    logger.info(
        "Stage 2 (load records): %d candidates loaded (%d malformed) in %.2f s",
        len(stage1_candidates), malformed_count, time.time() - t2
    )
                                                                     
    t2b = time.time()
    X, ordered_ids, consistency_map = extract_features_for_ranking(
        stage1_candidates, jd_config, bm25_scores, stage1_bm25_median, logger,
        static_features=static_features
    )
    logger.info("Stage 2b (features): %.2fs", time.time() - t2b)
                                                                    
    t4 = time.time()
    lgbm_scores = run_lightgbm_inference(model, X, ordered_ids, logger)
    
                                                                       
    for cid in lgbm_scores:
        lgbm_scores[cid] *= consistency_map.get(cid, 1.0)
        
    logger.info("Stage 4 (LightGBM + multiplier): %.2fs", time.time() - t4)                                                          
    t5 = time.time()
    ranked_top100 = sort_and_enforce_monotonicity(lgbm_scores, logger)

                                              
    assert_monotonicity(ranked_top100)
    logger.info("Monotonicity assertion PASSED.")

    assert len(ranked_top100) == 100, (
        f"Expected exactly 100 candidates, got {len(ranked_top100)}"
    )
    logger.info("Count assertion PASSED: exactly 100 candidates.")

    top100_ids = [cid for cid, _, _ in ranked_top100]                             
                                                                             
    from features import build_feature_vector, FEATURE_COLUMNS                        
    candidate_lookup: Dict[str, dict] = {
        c.get("candidate_id"): c for c in stage1_candidates
    }

    feature_vectors: Dict[str, dict] = {}
    for cid in top100_ids:
        c = candidate_lookup.get(cid)
        if c is None:
            feature_vectors[cid] = {col: 0.0 for col in FEATURE_COLUMNS}
            continue
        bs = bm25_scores.get(cid, 0.0)
        try:
            feature_vectors[cid] = build_feature_vector(
                c, jd_config, bs, stage1_bm25_median,
                precomputed_static=static_features.get(cid) if static_features else None
            )
        except Exception:
            feature_vectors[cid] = {col: 0.0 for col in FEATURE_COLUMNS}

    top100_candidates = [candidate_lookup[cid] for cid in top100_ids if cid in candidate_lookup]
                                                
    run_honeypot_audit(top100_candidates, feature_vectors, logger)                    
                                                                             
    run_diversity_audit(top100_candidates, feature_vectors, logger)                                                      
    t5r = time.time()
    from reasoning import ReasoningCompiler

    all_lgbm_scores = [lgbm_scores[cid] for cid in top100_ids if cid in lgbm_scores]
    compiler = ReasoningCompiler(jd_config, all_scores=all_lgbm_scores)

    reasoning_texts: Dict[str, str] = {}
    reasoning_traces: List[dict] = []

    for cid, norm_score, rank in ranked_top100:
        c = candidate_lookup.get(cid, {"candidate_id": cid})
        fv = feature_vectors.get(cid, {col: 0.0 for col in FEATURE_COLUMNS})
        raw_lgbm = lgbm_scores.get(cid, 0.0)

        if rank <= 30:
            trace = compiler.compile_trace(c, fv, raw_lgbm, rank)
            reasoning_traces.append(trace)
            reasoning_texts[cid] = trace["reasoning"]
        else:
            reasoning_texts[cid] = compiler.compile(c, fv, raw_lgbm, rank)

    logger.info("Stage 5 (reasoning): %.2fs", time.time() - t5r)
    write_reasoning_trace(reasoning_traces, base_dir, logger)                                              
    rows = []
    for cid, norm_score, rank in ranked_top100:
        rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(norm_score, 6),
            "reasoning": reasoning_texts.get(cid, ""),
        })

    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])                    
    assert len(df) == 100, f"DataFrame has {len(df)} rows, expected 100"
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"], \
        f"Unexpected columns: {list(df.columns)}"                                              
    scores_arr = df["score"].values
    for i in range(1, len(scores_arr)):
        assert scores_arr[i] <= scores_arr[i-1] + 1e-9, (
            f"DataFrame monotonicity violation at row {i}: "
            f"{scores_arr[i-1]:.8f} -> {scores_arr[i]:.8f}"
        )
    logger.info("Final DataFrame monotonicity assertion PASSED.")                                                                      
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Submission CSV written: %s", out_path)
    wall_elapsed = time.time() - wall_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Wall-clock time: %.2fs (limit: 300s)", wall_elapsed)
    logger.info("Output: %s", out_path)
    logger.info("Candidates ranked: 100")
    logger.info("=" * 60)

    if wall_elapsed > 300:
        logger.error(
            "TIMING VIOLATION: Pipeline took %.1fs > 300s limit", wall_elapsed
        )
        sys.exit(4)                                            
    print("\n--- submission.csv (first 5 rows) ---")
    print(df.head(5).to_string(index=False))
    print(f"\nTotal rows: {len(df)}")
    print(f"Score range: [{df['score'].min():.6f}, {df['score'].max():.6f}]")
    print(f"Wall-clock: {wall_elapsed:.1f}s")


if __name__ == "__main__":
    main()
