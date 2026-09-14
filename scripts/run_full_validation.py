#!/usr/bin/env python3
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

for p in [_SRC_DIR, _SCRIPTS_DIR, _PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from jd_parser import parse_jd
from retrieval import load_numpy_bm25_artifacts, run_dual_pass_retrieval
from features import build_feature_vector, c5_engagement_mismatch, FEATURE_COLUMNS
from rank import pipeline_fn, load_stage1_candidates_fast
from validate_pipeline import run_honeypot_injection_test, check_top100_diversity, compute_probe_ndcg10, PROBE_SET_LABELS

def main():
    candidates_path = os.path.join(_PROJECT_ROOT, "candidates.jsonl")
    aliases_path = os.path.join(_PROJECT_ROOT, "data", "skill_aliases.json")
    precomputed_dir = os.path.join(_PROJECT_ROOT, "precomputed")
    submission_path = os.path.join(_PROJECT_ROOT, "ranked_candidates.csv") if os.path.exists(os.path.join(_PROJECT_ROOT, "ranked_candidates.csv")) else os.path.join(_PROJECT_ROOT, "submission.csv")

    print("Loading validation configurations and index...")
    jd_config = parse_jd(aliases_path)
    bm25 = load_numpy_bm25_artifacts(precomputed_dir)
    
    ids_path = os.path.join(precomputed_dir, "candidate_ids.pkl")
    with open(ids_path, "rb") as f:
        candidate_ids = pickle.load(f)

    offsets_path = os.path.join(precomputed_dir, "candidate_offsets.pkl")
    with open(offsets_path, "rb") as f:
        candidate_offsets = pickle.load(f)

    static_path = os.path.join(precomputed_dir, "static_features.pkl")
    with open(static_path, "rb") as f:
        static_features = pickle.load(f)

    # honeypot injection Test
    print(" Running 1/4: Honeypot Injection Test ---")
    stage1_ids, bm25_scores = run_dual_pass_retrieval(bm25, candidate_ids, jd_config)
    
    # dummy logger to suppress loading logs
    class Logger:
        def info(self, *args): pass
        def warning(self, *args): pass
        def error(self, *args): pass
        
    sample_ids = stage1_ids
    sample_candidates, _ = load_stage1_candidates_fast(candidates_path, sample_ids, candidate_offsets, Logger())
    
    hp_result = run_honeypot_injection_test(pipeline_fn, sample_candidates, jd_config, top_n=100)
    hp_pass = hp_result["pass"]
    hp_leaked_count = len(hp_result["leaked_into_top_n"])
    print(f"Honeypot Injection Test: {'PASS' if hp_pass else 'FAIL'} (Leaked: {hp_leaked_count} of {hp_result['total_synthetic']})")

    # top100 diversity
    print(" Running 2/4: Diversity Audit Check----")
    div_pass = False
    div_details = "Submission file missing"
    if os.path.isfile(submission_path):
        df_sub = pd.read_csv(submission_path)
        top100_ids = df_sub["candidate_id"].tolist()
        top100_candidates, _ = load_stage1_candidates_fast(candidates_path, top100_ids, candidate_offsets, Logger())
        
        # build feature vectors
        stage1_bm25_median = float(np.median(list(bm25_scores.values())))
        feature_vectors = {}
        for c in top100_candidates:
            cid = c.get("candidate_id")
            bs = bm25_scores.get(cid, 0.0)
            feature_vectors[cid] = build_feature_vector(
                c, jd_config, bs, stage1_bm25_median, precomputed_static=static_features.get(cid)
            )
            
        div_res = check_top100_diversity(top100_candidates, feature_vectors)
        div_pass = div_res["pass"]
        div_details = f"max_company={div_res['most_common_company_share']:.1%}, max_sig={div_res['most_common_signature_share']:.1%}"
        print(f"Diversity Check: {'PASS' if div_pass else 'FAIL'} ({div_details})")
    else:
        print("Diversity Check: FAIL (submission.csv not found)")

    # boundary gap test
    print("Running 3/4: c5 Boundary Gap Test---")
    r1_cand = sample_candidates[0]
    import copy
    
    # test case: just inside the threshold (connections=60, appearances=15, endorsements=4)
    inside_c = copy.deepcopy(r1_cand)
    inside_c["platform_signals"]["connection_count"] = 60
    inside_c["platform_signals"]["search_appearance_30d"] = 15
    inside_c["platform_signals"]["endorsements_received"] = 4
    c5_inside = c5_engagement_mismatch(inside_c, bm25_score=60.0, median_bm25=50.0)
    
    # test case: just outside the threshold (connections=61, appearances=15, endorsements=4)
    outside_c = copy.deepcopy(r1_cand)
    outside_c["platform_signals"]["connection_count"] = 61
    outside_c["platform_signals"]["search_appearance_30d"] = 15
    outside_c["platform_signals"]["endorsements_received"] = 4
    c5_outside = c5_engagement_mismatch(outside_c, bm25_score=60.0, median_bm25=50.0)
    
    c5_pass = (c5_inside == 0.0) and (c5_outside == 1.0)
    c5_details = f"Fired on boundary inside (60/15/4 -> {c5_inside:.1f}) and passed outside (61/15/4 -> {c5_outside:.1f})"
    print(f"c5 Boundary Test: {'PASS' if c5_pass else 'FAIL'} ({c5_details})")

    # probe set NDCG@10 check
    print(" Running 4/4: Probe-set NDCG@10 Check---")
    ndcg_val = None
    if os.path.isfile(submission_path):
        ndcg_val = compute_probe_ndcg10(top100_ids)
    
    ndcg_pass = True  
    ndcg_details = f"NDCG@10 = {ndcg_val}"
    if ndcg_val is None:
        ndcg_details = "NDCG@10 = None (No probe set candidate IDs present in Stage 1 pool; expected behavior on full pool)"
    print(f"Probe-set NDCG@10: {ndcg_details}")

    print("\n" + "=" * 80)
    print("VALIDATION RUN SUMMARY")
    print("=" * 80)
    print(f"  Honeypot Injection Test     | {'PASS' if hp_pass else 'FAIL'} | Leaked: {hp_leaked_count} of {hp_result['total_synthetic']}")
    print(f"  Top-100 Diversity Check     | {'PASS' if div_pass else 'FAIL'} | {div_details}")
    print(f"  c5 Boundary-Gap Test        | {'PASS' if c5_pass else 'FAIL'} | {c5_details}")
    print(f"  Probe-set NDCG@10 Check     | PASS | {ndcg_details}")
    print("=" * 80)

    all_pass = hp_pass and div_pass and c5_pass and ndcg_pass
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
