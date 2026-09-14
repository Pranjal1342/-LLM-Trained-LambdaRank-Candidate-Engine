from __future__ import annotations
import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np                                                                           
_THIS_FILE    = os.path.abspath(__file__)
_EXP_DIR      = os.path.dirname(_THIS_FILE)
_EXPERIMENTS  = os.path.dirname(_EXP_DIR)
_PROJECT_ROOT = os.path.dirname(_EXPERIMENTS)
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

for _p in [_SRC_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)    
                                                                             
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pairwise] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pairwise_llm")                                                                 
_PROVIDER_SLEEP: Dict[str, float] = {
    "groq":      2.1,
    "anthropic": 0.1,
    "ollama":    0.5,
    "cerebras":  0.5,
}
_PROVIDER_PRICE: Dict[str, Tuple[float, float]] = {
    "groq":      (0.0,  0.0),
    "anthropic": (3.0, 15.0),
    "ollama":    (0.0,  0.0),
    "cerebras":  (0.0,  0.0),
}
_DEFAULT_MODELS: Dict[str, str] = {
    "groq":      "llama-3.1-8b-instant",
    "anthropic": "claude-sonnet-4-6",
    "ollama":    "gemma3:4b",
    "cerebras":  "llama3.1-8b",
}

                                                                           

def build_jd_summary(jd_config) -> str:
    lines = ["JOB: Senior AI/ML Engineer — Retrieval & Ranking Systems"]
    lines.append("HARD REQUIREMENTS (must have):")
    for req in jd_config.hard_requirements:
        lines.append(f"  - {req}")
    lines.append("PREFERRED (good to have):")
    preferred = jd_config.preferred_requirements
    keys = list(preferred.keys()) if isinstance(preferred, dict) else list(preferred)[:6]
    for req in keys[:6]:
        lines.append(f"  - {req}")
    lines.append("EXPLICIT DISQUALIFIERS:")
    lines.append("  - Entire career at IT-services/consulting firms (TCS, Infosys, Wipro, etc.)")
    lines.append("  - AI experience is only LangChain/OpenAI API with no pre-LLM IR or ML foundation")
    lines.append("  - CV/speech-only ML background with no NLP/IR experience")
    lines.append("  - Title-chaser: avg tenure < 15 months across 3+ jobs")
    lines.append("LOCATION PREFERENCE: Noida or Pune strongly preferred; other India acceptable; "
                 "outside India only if willing to relocate (no visa sponsorship)")
    lines.append("EXPERIENCE: 5-9 years preferred")
    lines.append("NOTICE PERIOD: Sub-30 days preferred; 30+ days raises the bar")
    return "\n".join(lines)
                                                                          

def build_candidate_summary(candidate: dict) -> str:
    
    profile  = candidate.get("profile", {}) or {}
    signals  = candidate.get("platform_signals", {}) or {}

    lines = []
    lines.append(f"ID: {candidate.get('candidate_id', 'unknown')}")
    lines.append(
        f"Title: {profile.get('current_title', 'unknown')} "
        f"at {profile.get('current_company', 'unknown')}"
    )
    lines.append(f"YOE: {profile.get('years_of_experience', 0)}")
    lines.append(
        f"Location: {profile.get('location', 'unknown')}, "
        f"{profile.get('country', 'unknown')}"
    )

                                       
    skills = sorted(
        candidate.get("skills", []) or [],
        key=lambda s: s.get("duration_months", 0),
        reverse=True,
    )[:5]
    assessments = signals.get("skill_assessment_scores", {}) or {}
    skill_lines = []
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        dur  = s.get("duration_months", 0)
        score = assessments.get(name)
        if score is not None:
            skill_lines.append(f"{name} ({prof}, {dur}mo, assessed: {score}/100)")
        else:
            skill_lines.append(f"{name} ({prof}, {dur}mo, unverified)")
    lines.append(f"Skills: {'; '.join(skill_lines)}")

                          
    for i, role in enumerate((candidate.get("career_history", []) or [])[:3]):
        desc = (role.get("description") or "")[:60].replace("\n", " ")
        lines.append(
            f"Role {i+1}: {role.get('title')} @ {role.get('company')} "
            f"({role.get('industry')}, {role.get('company_size')}, "
            f"{role.get('duration_months')}mo) — {desc}..."
        )

    lines.append(f"Notice: {signals.get('notice_period_days', 'unknown')} days")
    lines.append(f"Last active: {signals.get('last_active_date', 'unknown')}")
    lines.append(f"GitHub score: {signals.get('github_activity_score', -1)}")
    lines.append(f"Response rate: {signals.get('recruiter_response_rate', 'unknown')}")
    lines.append(f"Willing to relocate: {signals.get('willing_to_relocate', 'unknown')}")

    return "\n".join(lines)


def _call_groq(client, model: str, prompt: str) -> Tuple[str, int, int]:
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip().upper()
    return text, response.usage.prompt_tokens, response.usage.completion_tokens


def _call_anthropic(client, model: str, prompt: str) -> Tuple[str, int, int]:
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip().upper()
    return text, response.usage.input_tokens, response.usage.output_tokens


def _call_cerebras(client, model: str, prompt: str) -> Tuple[str, int, int]:
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip().upper()
    return text, response.usage.prompt_tokens, response.usage.completion_tokens


def _call_ollama(model: str, prompt: str) -> Tuple[str, int, int]:
    
    import requests as _req
    try:
        response = _req.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 10,
                    "num_ctx": 2048,
                    "num_gpu": 99,
                    "stop": ["\n", ".", " \n"],
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json()["response"].strip().upper()
        if "CANDIDATE_A" in raw:
            return "CANDIDATE_A", 0, 0
        elif "CANDIDATE_B" in raw:
            return "CANDIDATE_B", 0, 0
        else:
            return "TIE", 0, 0
    except _req.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama at localhost:11434. "
            "It starts automatically on Windows after install. "
            "Verify with: ollama list"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")


def get_pairwise_judgment(
    client,
    provider: str,
    model: str,
    jd_summary: str,
    summary_a: str,
    summary_b: str,
    pair_idx: int,
) -> Tuple[str, int, int]:
    prompt = f"""You are an expert technical recruiter. Read the job requirements and both candidate profiles carefully, then judge which candidate is the stronger fit.

{jd_summary}

--- CANDIDATE A ---
{summary_a}

--- CANDIDATE B ---
{summary_b}

Which candidate is a better fit for this specific role?

Respond with EXACTLY one of these three strings and nothing else:
CANDIDATE_A
CANDIDATE_B
TIE

No explanation. No punctuation. Just the label."""

    def _dispatch():
        if provider == "groq":
            return _call_groq(client, model, prompt)
        elif provider == "anthropic":
            return _call_anthropic(client, model, prompt)
        elif provider == "cerebras":
            return _call_cerebras(client, model, prompt)
        elif provider == "ollama":
            return _call_ollama(model, prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    try:
        text, inp, out = _dispatch()
    except Exception as e:
        logger.warning("API error on pair %d: %s", pair_idx, e)
        time.sleep(5)
        try:
            text, inp, out = _dispatch()
        except Exception as e2:
            logger.warning("Retry failed on pair %d: %s — defaulting to TIE", pair_idx, e2)
            return "TIE", 0, 0

    verdict = text if text in ("CANDIDATE_A", "CANDIDATE_B", "TIE") else "TIE"
    if text not in ("CANDIDATE_A", "CANDIDATE_B", "TIE"):
        logger.warning("Pair %d: unexpected output %r — defaulting to TIE", pair_idx, text)
    return verdict, inp, out


                                                                             
                          
                                                                             

def compute_elo_scores(
    annotations: List[dict],
    candidate_ids: List[str],
) -> Dict[str, float]:
    wins:   Dict[str, float] = {cid: 0.0 for cid in candidate_ids}
    losses: Dict[str, float] = {cid: 0.0 for cid in candidate_ids}

    for ann in annotations:
        a, b, verdict = ann["candidate_a"], ann["candidate_b"], ann["verdict"]
        if verdict == "CANDIDATE_A":
            wins[a] += 1.0;   losses[b] += 1.0
        elif verdict == "CANDIDATE_B":
            wins[b] += 1.0;   losses[a] += 1.0
        else:
            wins[a] += 0.5;   losses[a] += 0.5
            wins[b] += 0.5;   losses[b] += 0.5

    elo: Dict[str, float] = {}
    for cid in candidate_ids:
        total = wins[cid] + losses[cid]
        if total == 0:
            elo[cid] = 1500.0
        else:
            win_rate = (wins[cid] + 0.5) / (total + 1)                      
            elo[cid] = 400 * math.log10(win_rate / (1 - win_rate)) + 1500
    return elo


                                                                             
                           
                                                                             

def elo_to_labels(elo_scores: Dict[str, float]) -> Dict[str, int]:
    values = sorted(elo_scores.values())
    n  = len(values)
    q75 = values[int(0.75 * n)]
    q50 = values[int(0.50 * n)]
    q25 = values[int(0.25 * n)]
    labels: Dict[str, int] = {}
    for cid, elo in elo_scores.items():
        if elo >= q75:   labels[cid] = 3
        elif elo >= q50: labels[cid] = 2
        elif elo >= q25: labels[cid] = 1
        else:            labels[cid] = 0
    return labels


                                                                             
                             
                                                                             

def _get_top_skill(candidate: dict) -> str:
    skills = sorted(
        candidate.get("skills", []) or [],
        key=lambda s: s.get("duration_months", 0),
        reverse=True,
    )
    return skills[0].get("name", "N/A") if skills else "N/A"


def _spearman(
    candidate_ids: List[str],
    ranks_a: Dict[str, int],
    ranks_b: Dict[str, int],
) -> float:
    
    from scipy.stats import spearmanr
    common = [cid for cid in candidate_ids if cid in ranks_a and cid in ranks_b]
    if len(common) < 2:
        return 0.0
    ra = [ranks_a[cid] for cid in common]
    rb = [ranks_b[cid] for cid in common]
    rho, _ = spearmanr(ra, rb)
    return float(rho)


def print_model_comparison(
    stage1_candidates: Dict[str, dict],
    stage1_ids: List[str],
    bm25_scores: Dict[str, float],
    stage1_bm25_median: float,
    jd_config,
    old_model,
    new_model,
    feature_columns: List[str],
) -> None:
    
    from features import build_feature_vector

    logger.info("Building full feature matrix for comparison report...")

    feature_rows       = []
    ordered_ids        = []
    consistency_map: Dict[str, float] = {}

    for cid in stage1_ids:
        candidate = stage1_candidates.get(cid)
        if candidate is None:
            continue
        bs = bm25_scores.get(cid, 0.0)
        try:
            fv = build_feature_vector(
                candidate, jd_config,
                bm25_score=bs,
                stage1_bm25_median=stage1_bm25_median,
            )
            row = [fv[col] for col in feature_columns]
            consistency_map[cid] = float(fv.get("consistency_score", 1.0))
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", cid, e)
            row = [0.0] * len(feature_columns)
            consistency_map[cid] = 1.0
        feature_rows.append(row)
        ordered_ids.append(cid)

    X_full = np.array(feature_rows, dtype=np.float32)
    logger.info("Comparison feature matrix: shape=%s", X_full.shape)

                                            
    old_raw    = old_model.predict(X_full)
    old_scores = {cid: float(s) for cid, s in zip(ordered_ids, old_raw)}
    old_ranked = sorted(old_scores.items(), key=lambda x: (-x[1], x[0]))
    old_rank_map = {cid: rank for rank, (cid, _) in enumerate(old_ranked, 1)}

                                                                       
                                                                             
                                                                             
    new_raw    = new_model.predict(X_full)
    new_scores = {
        cid: float(s) * consistency_map.get(cid, 1.0)
        for cid, s in zip(ordered_ids, new_raw)
    }
    new_ranked   = sorted(new_scores.items(), key=lambda x: (-x[1], x[0]))
    new_rank_map = {cid: rank for rank, (cid, _) in enumerate(new_ranked, 1)}

    old_top10 = [cid for cid, _ in old_ranked[:10]]
    new_top10 = [cid for cid, _ in new_ranked[:10]]
    overlap   = len(set(old_top10) & set(new_top10))

                                                               
    top100_old = [cid for cid, _ in old_ranked[:100]]
    rho = _spearman(top100_old, old_rank_map, new_rank_map)

                      
    moved_up:   List[Tuple[str, int, int]] = []
    moved_down: List[Tuple[str, int, int]] = []
    for cid in ordered_ids:
        old_r = old_rank_map.get(cid, 9999)
        new_r = new_rank_map.get(cid, 9999)
        delta = old_r - new_r
        if delta >= 20:
            moved_up.append((cid, old_r, new_r))
        elif delta <= -20:
            moved_down.append((cid, old_r, new_r))

    moved_up.sort(key=lambda x: x[1] - x[2], reverse=True)
    moved_down.sort(key=lambda x: x[2] - x[1], reverse=True)

                                             
    new_top100        = [cid for cid, _ in new_ranked[:100]]
    low_cons_count    = sum(1 for cid in new_top100 if consistency_map.get(cid, 1.0) < 0.25)
    honeypot_pass     = low_cons_count < 10

                            
    print("\n" + "=" * 60)
    print("=== MODEL COMPARISON REPORT ===")
    print("=" * 60)

    print("\nCurrent model (heuristic labels) top-10:")
    for rank, cid in enumerate(old_top10, 1):
        c  = stage1_candidates.get(cid, {})
        p  = c.get("profile", {}) or {}
        print(f"  {rank:2d}. {cid} — {p.get('current_title','N/A')}, "
              f"{p.get('years_of_experience',0)}y, {_get_top_skill(c)}")

    print("\nNew model (LLM pairwise labels + consistency multiplier) top-10:")
    for rank, cid in enumerate(new_top10, 1):
        c  = stage1_candidates.get(cid, {})
        p  = c.get("profile", {}) or {}
        cons = consistency_map.get(cid, 1.0)
        print(f"  {rank:2d}. {cid} — {p.get('current_title','N/A')}, "
              f"{p.get('years_of_experience',0)}y, {_get_top_skill(c)}, "
              f"cons={cons:.2f}")

    print(f"\nOverlap: {overlap} of 10 top-10 candidates appear in both rankings")
    print(f"Spearman correlation (top-100): {rho:.3f}  "
          f"[range: -1.0 to +1.0, higher = more agreement]")

    print("\nCandidates that MOVED UP 20+ positions in new model:")
    for cid, old_r, new_r in moved_up[:10]:
        c = stage1_candidates.get(cid, {})
        p = c.get("profile", {}) or {}
        print(f"  - {cid}: old={old_r}, new={new_r} | "
              f"{p.get('current_title','N/A')}, {_get_top_skill(c)}")

    print("\nCandidates that MOVED DOWN 20+ positions in new model:")
    for cid, old_r, new_r in moved_down[:10]:
        c = stage1_candidates.get(cid, {})
        p = c.get("profile", {}) or {}
        print(f"  - {cid}: old={old_r}, new={new_r} | "
              f"{p.get('current_title','N/A')}, {_get_top_skill(c)}")

    print(f"\nConsistency check — low-consistency (< 0.25) in new top-100:")
    print(f"  Count: {low_cons_count}  (must be < 10 to pass honeypot audit)")
    print(f"  NOTE: consistency multiplier applied — this number should now be 0.")

    print("\n" + "=" * 60)
    print("=== VERDICT ===")
    print(f"Honeypot audit:              {'PASS ✓' if honeypot_pass else 'FAIL ✗'}")
    print(f"Top-10 overlap with current: {overlap}/10")
    print(f"Spearman correlation:        {rho:.3f}")

    if honeypot_pass and rho > 0.4:
        rec = "PROMISING — consider swapping model"
    elif honeypot_pass and rho <= 0.4:
        rec = "MIXED — honeypot passes but ranking diverges significantly; review movers"
    else:
        rec = "RISKY — honeypot audit fails; do not swap without further investigation"
    print(f"Recommendation:              {rec}")
    print("=" * 60 + "\n")

    if not honeypot_pass:
        logger.error(
            "HONEYPOT AUDIT FAILED: %d low-consistency candidates in new top-100. "
            "The consistency multiplier should have fixed this — check that "
            "consistency_score is being computed correctly for these candidates.",
            low_cons_count,
        )
    else:
        logger.info("Honeypot audit PASSED: %d low-consistency in new top 100.", low_cons_count)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline pairwise LLM annotation experiment. "
            "If lgbm_model_llm.pkl already exists, runs Step 11 comparison only. "
            "NEVER imported by rank.py or any production module."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--base-dir",   required=True)
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["groq", "anthropic", "ollama", "cerebras"],
    )
    parser.add_argument("--model",       default=None)
    parser.add_argument("--api-key",     default=None)
    parser.add_argument("--ollama-url",  default="http://localhost:11434")
    args = parser.parse_args()

    provider   = args.provider
    model      = args.model or _DEFAULT_MODELS[provider]
    call_sleep = _PROVIDER_SLEEP[provider]
    price_in, price_out = _PROVIDER_PRICE[provider]

                                  
    if provider in ("groq", "anthropic", "cerebras") and not args.api_key:
        logger.error("--api-key is required when --provider is %s", provider)
        sys.exit(1)
    if provider == "ollama" and args.api_key:
        logger.info("--api-key ignored for ollama provider")

    base_dir        = os.path.abspath(args.base_dir)
    candidates_path = os.path.abspath(args.candidates)
    precomputed_dir = os.path.join(base_dir, "precomputed")
    data_dir        = os.path.join(base_dir, "data")
    annotations_path = os.path.join(_EXP_DIR, "annotations.jsonl")
    new_model_path   = os.path.join(precomputed_dir, "lgbm_model_llm.pkl")
    old_model_path   = os.path.join(precomputed_dir, "lgbm_model.pkl")

    logger.info("=" * 60)
    logger.info("PAIRWISE LLM ANNOTATION EXPERIMENT")
    logger.info("Provider: %s | Model: %s", provider, model)
    logger.info("Rate limit sleep: %.1fs between calls", call_sleep)
    logger.info("Cost: %s", "FREE" if price_in == 0 else f"${price_in}/M input, ${price_out}/M output")
    logger.info("Base dir:    %s", base_dir)
    logger.info("Annotations: %s", annotations_path)
    logger.info("New model → %s", new_model_path)
    logger.info("Old model   %s  (will NOT be touched)", old_model_path)
    logger.info("=" * 60)

                                         
    if provider == "ollama":
        import requests as _req
        try:
            r = _req.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            available = [m["name"] for m in r.json().get("models", [])]
            found = (
                model in available
                or model.split(":")[0] in [m.split(":")[0] for m in available]
            )
            if not found:
                logger.error(
                    "Model '%s' not in Ollama. Available: %s. "
                    "Pull it: ollama pull %s", model, available, model
                )
                sys.exit(1)
            logger.info("Ollama reachable. Model '%s' available.", model)
        except _req.exceptions.ConnectionError:
            logger.error(
                "Ollama not running at localhost:11434. "
                "On Windows it auto-starts after install — "
                "check Task Manager for 'ollama' process."
            )
            sys.exit(1)

                                      
    from features import build_feature_vector, FEATURE_COLUMNS
    from features import (
        c1_timeline_impossibility, c2_signup_anomaly,
        c3_salary_inversion, c4_assessment_contradiction,
        c5_engagement_mismatch,
    )
    from jd_parser import parse_jd
    from retrieval import load_numpy_bm25_artifacts, run_dual_pass_retrieval

                                         
    logger.info("STEP 1: Loading Stage 1 candidate pool...")

    bm25 = load_numpy_bm25_artifacts(precomputed_dir)
    if bm25 is None:
        bm25_path = os.path.join(precomputed_dir, "bm25_index.pkl")
        if not os.path.isfile(bm25_path):
            logger.error("Missing bm25_index.pkl — run precompute.py first.")
            sys.exit(1)
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        logger.info("Loaded legacy BM25Okapi")
    else:
        logger.info("Loaded NumpyBM25 (fast path)")

    ids_path = os.path.join(precomputed_dir, "candidate_ids.pkl")
    with open(ids_path, "rb") as f:
        all_candidate_ids = pickle.load(f)

    aliases_path = os.path.join(data_dir, "skill_aliases.json")
    jd_config    = parse_jd(aliases_path)
    logger.info(
        "JD config: %d hard reqs, %d preferred reqs",
        len(jd_config.hard_requirements), len(jd_config.preferred_requirements),
    )

    stage1_ids, bm25_scores = run_dual_pass_retrieval(bm25, all_candidate_ids, jd_config)
    stage1_bm25_median = float(np.median(list(bm25_scores.values())))
    logger.info("Stage 1 pool: %d candidates, median BM25=%.4f", len(stage1_ids), stage1_bm25_median)

                          
    offsets_path = os.path.join(precomputed_dir, "candidate_offsets.pkl")
    stage1_candidate_list: List[dict] = []
    if os.path.isfile(offsets_path):
        with open(offsets_path, "rb") as f:
            candidate_offsets = pickle.load(f)
        logger.info("Loading Stage 1 records via byte-offset index...")
        with open(candidates_path, "rb") as f:
            for cid in stage1_ids:
                offset = candidate_offsets.get(cid)
                if offset is None:
                    continue
                f.seek(offset)
                raw = f.readline()
                try:
                    c = json.loads(raw.decode("utf-8", errors="ignore").strip())
                    stage1_candidate_list.append(c)
                except json.JSONDecodeError:
                    pass
    else:
        logger.info("No offset index — streaming JSONL (slow)...")
        stage1_id_set = set(stage1_ids)
        found: Dict[str, dict] = {}
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = c.get("candidate_id")
                if cid and cid in stage1_id_set:
                    found[cid] = c
                    if len(found) == len(stage1_id_set):
                        break
        stage1_candidate_list = [found[cid] for cid in stage1_ids if cid in found]

    stage1_candidates: Dict[str, dict] = {
        c.get("candidate_id"): c
        for c in stage1_candidate_list
        if c.get("candidate_id")
    }
    logger.info("Stage 1 records loaded: %d candidates", len(stage1_candidates))

                                                                          
    model_already_exists  = os.path.isfile(new_model_path)
    annots_already_exist  = os.path.isfile(annotations_path)

    if model_already_exists and annots_already_exist:
        logger.info(
            "lgbm_model_llm.pkl and annotations.jsonl both exist — "
            "skipping Steps 2-10, running Step 11 comparison only."
        )
        with open(old_model_path, "rb") as f:
            old_model = pickle.load(f)
        with open(new_model_path, "rb") as f:
            new_model = pickle.load(f)

        logger.info("STEP 11: Generating model comparison report...")
        print_model_comparison(
            stage1_candidates=stage1_candidates,
            stage1_ids=stage1_ids,
            bm25_scores=bm25_scores,
            stage1_bm25_median=stage1_bm25_median,
            jd_config=jd_config,
            old_model=old_model,
            new_model=new_model,
            feature_columns=FEATURE_COLUMNS,
        )
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("New model: %s", new_model_path)
        logger.info(
            "To swap into production:  copy %s %s",
            new_model_path, old_model_path,
        )
        logger.info("(Manual, deliberate action only — verify top-10 first)")
        logger.info("=" * 60)
        return

                                                                        

                                           
    logger.info("STEP 2: Stratified sampling of 500 candidates...")
    random.seed(42)

    from features import build_feature_vector
    import lightgbm as lgb

    with open(old_model_path, "rb") as f:
        old_model_for_ranking = pickle.load(f)

    logger.info("Computing feature vectors for all Stage 1 candidates...")
    all_feature_rows = []
    all_fv_ids       = []
    consistency_scores_all: Dict[str, float] = {}

    for cid in stage1_ids:
        candidate = stage1_candidates.get(cid)
        if candidate is None:
            continue
        bs = bm25_scores.get(cid, 0.0)
        try:
            fv  = build_feature_vector(candidate, jd_config, bm25_score=bs, stage1_bm25_median=stage1_bm25_median)
            row = [fv[col] for col in FEATURE_COLUMNS]
            consistency_scores_all[cid] = float(fv.get("consistency_score", 1.0))
        except Exception:
            row = [0.0] * len(FEATURE_COLUMNS)
            consistency_scores_all[cid] = 1.0
        all_feature_rows.append(row)
        all_fv_ids.append(cid)

    X_all = np.array(all_feature_rows, dtype=np.float32)
    logger.info("Feature matrix (Stage 1): shape=%s", X_all.shape)

    raw_scores  = old_model_for_ranking.predict(X_all)
    lgbm_ranked = sorted(zip(all_fv_ids, raw_scores), key=lambda x: -x[1])
    lgbm_rank_map = {cid: rank for rank, (cid, _) in enumerate(lgbm_ranked, 1)}

              
    TOTAL     = 500
    N_A, N_B, N_C = 75, 100, 325
    MIN_LOW_CONS   = 25

    top100_cids    = [cid for cid, _ in lgbm_ranked[:100]]
    ranks_101_300  = [cid for cid, _ in lgbm_ranked[100:300]]
    ranks_301_plus = [cid for cid, _ in lgbm_ranked[300:]]

    stratum_a = random.sample(top100_cids, min(N_A, len(top100_cids)))
    stratum_b = random.sample(ranks_101_300, min(N_B, len(ranks_101_300)))

    low_cons_pool = [cid for cid in ranks_301_plus if consistency_scores_all.get(cid, 1.0) < 0.5]
    guaranteed_low = random.sample(low_cons_pool, min(MIN_LOW_CONS, len(low_cons_pool)))
    remaining_c    = [cid for cid in ranks_301_plus if cid not in guaranteed_low]
    fill_c         = random.sample(remaining_c, max(0, N_C - len(guaranteed_low)))
    stratum_c      = guaranteed_low + fill_c

    sample_ids = list(dict.fromkeys(stratum_a + stratum_b + stratum_c))[:TOTAL]
    logger.info(
        "Stratum sizes: A=%d (top-50 + 25 from 51-150), B=%d (51-150), C=%d (151+)",
        len(stratum_a), len(stratum_b), len(stratum_c),
    )
    logger.info("Low-consistency guaranteed in Stratum C: %d (target: ≥%d)",
                len(guaranteed_low), MIN_LOW_CONS)
    logger.info("Total sample pool: %d candidates", len(sample_ids))

                                         
    logger.info("STEP 3: Generating pairwise matchups (5 opponents per candidate)...")
    N_OPPONENTS = 5
    seen_pairs:  set = set()
    pairs:       List[Tuple[str, str]] = []

    for cid_a in sample_ids:
        pool = [c for c in sample_ids if c != cid_a]
        random.shuffle(pool)
        count = 0
        for cid_b in pool:
            key = frozenset({cid_a, cid_b})
            if key not in seen_pairs and count < N_OPPONENTS:
                seen_pairs.add(key)
                pairs.append((cid_a, cid_b))
                count += 1

    logger.info("Unique pairs generated: %d", len(pairs))

                                  
    logger.info("STEP 6: Annotating pairs with %s (%s)...", provider, model)

                                                
    existing_annotations: List[dict] = []
    existing_pair_keys: set = set()
    if os.path.isfile(annotations_path):
        logger.info("Found existing annotations file — loading for resumability...")
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = json.loads(line)
                    existing_annotations.append(ann)
                    existing_pair_keys.add(frozenset({ann["candidate_a"], ann["candidate_b"]}))
                except json.JSONDecodeError:
                    pass
        logger.info("Loaded %d existing annotations (will skip these pairs)", len(existing_annotations))

    remaining_pairs = [(a, b) for a, b in pairs if frozenset({a, b}) not in existing_pair_keys]
    logger.info("Pairs remaining to annotate: %d of %d", len(remaining_pairs), len(pairs))

                           
    jd_summary = build_jd_summary(jd_config)

                 
    client = None
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=args.api_key)
        logger.info("Groq client initialized (model: %s)", model)
    elif provider == "anthropic":
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=args.api_key)
        logger.info("Anthropic client initialized (model: %s)", model)
    elif provider == "cerebras":
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=args.api_key)
        logger.info("Cerebras client initialized (model: %s)", model)
    else:
        logger.info("Ollama provider: calls go directly to localhost:11434 via requests")

                            
    logger.info("Running 5-call timing probe for Ollama...")
    probe_pairs  = remaining_pairs[:5] if len(remaining_pairs) >= 5 else pairs[:5]
    probe_times  = []
    probe_inp    = []
    probe_out    = []

    for i, (a, b) in enumerate(probe_pairs):
        sa = build_candidate_summary(stage1_candidates.get(a, {"candidate_id": a}))
        sb = build_candidate_summary(stage1_candidates.get(b, {"candidate_id": b}))
        t0 = time.time()
        _, inp, out = get_pairwise_judgment(client, provider, model, jd_summary, sa, sb, i)
        elapsed = time.time() - t0
        probe_times.append(elapsed)
        probe_inp.append(inp)
        probe_out.append(out)
        time.sleep(call_sleep)

    avg_secs   = sum(probe_times) / len(probe_times)
    avg_inp    = sum(probe_inp)   / len(probe_inp)
    avg_out    = sum(probe_out)   / len(probe_out)
    est_min    = (avg_secs + call_sleep) * len(remaining_pairs) / 60
    est_cost   = (avg_inp * len(remaining_pairs) / 1e6 * price_in +
                  avg_out * len(remaining_pairs) / 1e6 * price_out)

    print("\n" + "=" * 50)
    print("=== RUN ESTIMATE ===")
    print(f"Provider:  {provider} ({model})")
    print(f"Pairs to annotate: {len(remaining_pairs)}")
    if provider in ("groq", "anthropic", "cerebras"):
        print(f"Avg input tokens per call: {avg_inp:.0f}")
    else:
        print(f"Avg seconds per call: {avg_secs:.1f}s")
    print(f"Estimated cost: {'FREE' if est_cost == 0 else f'${est_cost:.2f}'}")
    print(f"Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} hrs)")
    if provider == "ollama":
        print(f"GPU acceleration: {'YES' if avg_secs < 2.0 else 'NO — running on CPU (slow)'}")
    print("=" * 50)

    confirm = input("Proceed with full run? (yes/no): ").strip().lower()
    if confirm != "yes":
        logger.info("User declined — exiting. Run again to resume.")
        sys.exit(0)

    logger.info("Starting full annotation run (%d pairs remaining)...", len(remaining_pairs))

    total_inp = 0
    total_out = 0
    annot_file = open(annotations_path, "a", encoding="utf-8")

    try:
        for idx, (cid_a, cid_b) in enumerate(remaining_pairs):
            sa = build_candidate_summary(stage1_candidates.get(cid_a, {"candidate_id": cid_a}))
            sb = build_candidate_summary(stage1_candidates.get(cid_b, {"candidate_id": cid_b}))

            verdict, inp, out = get_pairwise_judgment(
                client, provider, model, jd_summary, sa, sb, idx
            )
            total_inp += inp
            total_out += out

            record = {
                "pair_id":     idx,
                "candidate_a": cid_a,
                "candidate_b": cid_b,
                "verdict":     verdict,
                "input_tokens":  inp,
                "output_tokens": out,
            }
            annot_file.write(json.dumps(record) + "\n")
            annot_file.flush()
            existing_annotations.append(record)

            time.sleep(call_sleep)

            if (idx + 1) % 100 == 0:
                cost_so_far = (total_inp / 1e6 * price_in + total_out / 1e6 * price_out)
                logger.info(
                    "Progress: %d/%d pairs | cost: $%.2f | elapsed: ~%d min",
                    idx + 1, len(remaining_pairs), cost_so_far, int((idx+1)*(avg_secs+call_sleep)/60)
                )

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("INTERRUPTED by user (Ctrl+C) — progress saved cleanly.")
        logger.info("Pairs completed so far: %d", len(existing_annotations))
        logger.info("Annotations file: %s", annotations_path)
        logger.info("Re-run the same command to resume from pair %d.", len(existing_annotations))
        logger.info("=" * 60)
        annot_file.close()
        sys.exit(0)
    finally:
        annot_file.close()

    actual_cost = total_inp / 1e6 * price_in + total_out / 1e6 * price_out
    logger.info(
        "Annotation complete. Total tokens: %d input / %d output. "
        "Actual total cost: $%.2f",
        total_inp, total_out, actual_cost,
    )

                           
    logger.info("STEP 7: Computing Elo scores from pairwise verdicts...")
    elo_scores = compute_elo_scores(existing_annotations, sample_ids)
    elo_vals   = list(elo_scores.values())
    logger.info(
        "Elo distribution: min=%.1f, max=%.1f, mean=%.1f, std=%.1f",
        min(elo_vals), max(elo_vals),
        sum(elo_vals)/len(elo_vals),
        float(np.std(elo_vals)),
    )
    winners = sum(1 for v in elo_vals if v > 1500)
    logger.info("Elo above 1500 (winners): %d | at/below 1500 (losers): %d",
                winners, len(elo_vals) - winners)

                              
    logger.info("STEP 8: Converting Elo scores to 0-3 relevance labels...")
    labels = elo_to_labels(elo_scores)
    dist   = {0: 0, 1: 0, 2: 0, 3: 0}
    for v in labels.values():
        dist[v] += 1
    logger.info("Label distribution: 0=%d, 1=%d, 2=%d, 3=%d",
                dist[0], dist[1], dist[2], dist[3])
    if dist[3] < 30:
        logger.warning(
            "Only %d candidates with label 3 — training signal may be sparse.", dist[3]
        )

                                                   
    logger.info("STEP 9: Extracting feature matrix for %d annotated candidates...", len(sample_ids))
    train_rows = []
    train_ids  = []
    for cid in sample_ids:
        candidate = stage1_candidates.get(cid)
        if candidate is None:
            continue
        bs = bm25_scores.get(cid, 0.0)
        try:
            fv  = build_feature_vector(candidate, jd_config, bm25_score=bs, stage1_bm25_median=stage1_bm25_median)
            row = [fv[col] for col in FEATURE_COLUMNS]
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", cid, e)
            row = [0.0] * len(FEATURE_COLUMNS)
        train_rows.append(row)
        train_ids.append(cid)

    X_train_full = np.array(train_rows, dtype=np.float32)
    y_full       = np.array([labels.get(cid, 0) for cid in train_ids], dtype=np.int32)
    logger.info("Feature matrix (%d): shape=%s", len(sample_ids), X_train_full.shape)

                                       
    logger.info("STEP 10: Training LightGBM on LLM pairwise labels...")

    random.seed(42)
    n_val     = int(len(train_ids) * 0.2)
    perm      = list(range(len(train_ids)))
    random.shuffle(perm)
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    X_tr  = X_train_full[train_idx]
    y_tr  = y_full[train_idx]
    X_vl  = X_train_full[val_idx]
    y_vl  = y_full[val_idx]

    logger.info("Train/val split: %d train, %d val", len(train_idx), len(val_idx))
    dist_tr = {k: int((y_tr == k).sum()) for k in [0,1,2,3]}
    logger.info("Train label distribution: 0=%d, 1=%d, 2=%d, 3=%d", *[dist_tr[k] for k in [0,1,2,3]])

    train_ds = lgb.Dataset(X_tr, label=y_tr, group=[len(train_idx)], feature_name=FEATURE_COLUMNS)
    val_ds   = lgb.Dataset(X_vl, label=y_vl, group=[len(val_idx)],   feature_name=FEATURE_COLUMNS, reference=train_ds)

    params = {
        "objective":       "lambdarank",
        "metric":          "ndcg",
        "eval_at":         [5, 10, 50],
        "num_leaves":      63,
        "learning_rate":   0.05,
        "min_child_samples": 20,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "random_state":    42,
        "n_jobs":          -1,
        "verbose":         -1,
    }

    t0 = time.time()
    new_model = lgb.train(
        params,
        train_ds,
        num_boost_round=300,
        valid_sets=[val_ds],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    logger.info("LightGBM training complete in %.1fs", time.time() - t0)

    importances = sorted(
        zip(FEATURE_COLUMNS, new_model.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("Top 5 feature importances (gain):")
    for fname, imp in importances[:5]:
        logger.info("  %s: %.2f", fname, imp)

    with open(new_model_path, "wb") as f:
        pickle.dump(new_model, f)
    logger.info("New model saved to: %s", new_model_path)
    logger.info("lgbm_model.pkl untouched: %s", old_model_path)

                                   
    logger.info("STEP 11: Generating model comparison report...")
    with open(old_model_path, "rb") as f:
        old_model_final = pickle.load(f)

    print_model_comparison(
        stage1_candidates=stage1_candidates,
        stage1_ids=stage1_ids,
        bm25_scores=bm25_scores,
        stage1_bm25_median=stage1_bm25_median,
        jd_config=jd_config,
        old_model=old_model_final,
        new_model=new_model,
        feature_columns=FEATURE_COLUMNS,
    )

    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("Annotations: %s", annotations_path)
    logger.info("New model:   %s", new_model_path)
    logger.info(
        "To swap into production:  copy %s %s  (manual, deliberate action only)",
        new_model_path, old_model_path,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
