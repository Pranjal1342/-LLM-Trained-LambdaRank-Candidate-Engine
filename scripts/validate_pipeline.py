"""
validate_pipeline.py

Local validation protocol for the candidate ranking system.
Runs entirely offline, no network calls, designed to be executed
before any of the 3 allowed competition submissions are spent.

Checks performed:
  1. Probe-set NDCG@10 against hand-labeled reference candidates
  2. Ablation table (component on/off, confirm monotonic improvement)
  3. Honeypot injection test (synthetic violations of c1-c7, confirm suppression)
  4. Top-100 diversity / homogeneity check (NEW - this revision)
  5. Readiness gate (numeric threshold before spending a submission)
"""

import json
import hashlib
from collections import Counter
from itertools import combinations


PROBE_SET_LABELS = {
    "CAND_0000001": 3,  
    "CAND_0000010": 3,  

    "CAND_0000021": 0,  
    "CAND_0000014": 2,  
    "CAND_0000011": 1,  
}


def compute_probe_ndcg10(ranked_candidate_ids: list[str],
                          labels: dict[str, int] = PROBE_SET_LABELS) -> float:
    """
    NDCG@10 restricted to candidates that appear in both the ranked
    output and the probe set. Only meaningful once the probe set is
    grown beyond the current 5 reference points (see TODO below).
    """
    import math

    relevant_in_rank = [
        (rank, labels[cid])
        for rank, cid in enumerate(ranked_candidate_ids[:10], start=1)
        if cid in labels
    ]
    if not relevant_in_rank:
        return None  # probe set didn't overlap with top 10 at all

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in relevant_in_rank)

    ideal_order = sorted(labels.values(), reverse=True)[:10]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_order))

    return dcg / idcg if idcg > 0 else 0.0


# ablation table

def run_ablation(pipeline_fn, candidates: list[dict], jd_config: dict) -> dict:
    """
    pipeline_fn(candidates, jd_config, **toggles) -> list[ranked_candidate_ids]
    Each toggle disables one component. Confirms NDCG@10 on the probe
    set does not improve when a component is removed -- if it does,
    that component is actively hurting ranking quality and is a bug,
    not a feature.
    """
    configs = {
        "full_pipeline":            dict(),
        "no_consistency_checks":    dict(disable_consistency=True),
        "no_parameter_a":           dict(disable_param_a=True),
        "bm25_only_no_features":    dict(disable_features=True),
    }

    results = {}
    for name, toggles in configs.items():
        ranked = pipeline_fn(candidates, jd_config, **toggles)
        results[name] = {
            "ndcg10": compute_probe_ndcg10(ranked),
            "top10_ids": ranked[:10],
        }
    return results


def print_ablation_report(results: dict) -> None:
    print("=" * 60)
    print("ABLATION REPORT")
    print("=" * 60)
    baseline = results["full_pipeline"]["ndcg10"]
    for name, r in results.items():
        flag = ""
        if name != "full_pipeline" and baseline is not None and r["ndcg10"] is not None:
            if r["ndcg10"] > baseline:
                flag = "  <-- WARNING: removing this IMPROVED the score. Investigate."
        print(f"{name:30s} NDCG@10 = {r['ndcg10']}{flag}")

# honeypot injection test

def make_synthetic_honeypot(violation: str, base_candidate: dict) -> dict:
    """
    Clones a real candidate and deliberately injects exactly one
    consistency-check violation, so each test case isolates a single
    check rather than confounding several at once.
    """
    c = json.loads(json.dumps(base_candidate))  
    c["candidate_id"] = f"SYNTH_{violation.upper()}"

    if violation == "timeline_impossibility":
        c["skills"][0]["duration_months"] = int(c["profile"]["years_of_experience"] * 12) + 50

    elif violation == "signup_anomaly":
        c["platform_signals"]["signup_date"] = "2099-01-01"
        c["platform_signals"]["last_active_date"] = "2026-01-01"

    elif violation == "salary_inversion":
        c["platform_signals"]["expected_salary_range_inr_lpa"] = {"min": 50.0, "max": 10.0}

    elif violation == "assessment_contradiction":
        skill_name = c["skills"][0]["name"]
        c["skills"][0]["proficiency"] = "advanced"
        c["platform_signals"]["skill_assessment_scores"][skill_name] = 12.0

    elif violation == "engagement_mismatch":
        c["platform_signals"]["connection_count"] = 0
        c["platform_signals"]["search_appearance_30d"] = 0
        c["platform_signals"]["endorsements_received"] = 0

    elif violation == "langchain_dabbler":
        c["skills"] = [
            {"name": "LangChain", "proficiency": "advanced", "endorsements": 2, "duration_months": 6},
            {"name": "Prompt Engineering", "proficiency": "advanced", "endorsements": 1, "duration_months": 4},
        ]
        c["platform_signals"]["skill_assessment_scores"] = {}

    elif violation == "cv_specialist_no_nlp":
        c["skills"] = [
            {"name": "OpenCV", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
            {"name": "YOLO", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
        ]

    return c


VIOLATION_TYPES = [
    "timeline_impossibility", "signup_anomaly", "salary_inversion",
    "assessment_contradiction", "engagement_mismatch",
    "langchain_dabbler", "cv_specialist_no_nlp",
]


def run_honeypot_injection_test(pipeline_fn, real_candidates: list[dict],
                                 jd_config: dict, top_n: int = 100) -> dict:
    base = real_candidates[0]
    synthetic = [make_synthetic_honeypot(v, base) for v in VIOLATION_TYPES]
    test_pool = real_candidates + synthetic

    ranked = pipeline_fn(test_pool, jd_config)
    top_n_ids = set(ranked[:top_n])

    synthetic_ids = {c["candidate_id"] for c in synthetic}
    leaked = synthetic_ids & top_n_ids

    return {
        "total_synthetic": len(synthetic_ids),
        "leaked_into_top_n": leaked,
        "pass": len(leaked) == 0,
    }


# diversity check

def candidate_archetype_signature(candidate: dict, feature_vector: dict) -> tuple:
    """
    A coarse, human readable signature for clustering deliberately
    simple (no embeddings, no clustering library) so it stays fast
    and auditable. Buckets each candidate into a small discrete
    profile rather than computing exact distances.
    """
    yoe_bucket = (
        "junior" if candidate["profile"]["years_of_experience"] < 3 else
        "mid" if candidate["profile"]["years_of_experience"] < 7 else
        "senior"
    )
    top_skill = max(
        candidate.get("skills", [{"name": "none", "duration_months": 0}]),
        key=lambda s: s.get("duration_months", 0)
    )["name"]
    industry = candidate["profile"].get("current_industry", "unknown")
    company = candidate["profile"].get("current_company", "unknown")

    return (yoe_bucket, top_skill, industry, company)


def check_top100_diversity(top_100_candidates: list[dict],
                            feature_vectors: dict[str, dict],
                            max_signature_share: float = 0.25,
                            max_single_company_share: float = 0.20) -> dict:
    """
    Flags two specific homogeneity failure modes:
      (a) one archetype signature dominating > max_signature_share
          of the top 100 -- e.g. 30 nearly-identical profiles
      (b) one single employer accounting for too large a share of
          the top 100 -- a narrower, more specific version of (a)
          that's easy to misread as "we found the best company"
          rather than "our company-size/industry feature is too
          dominant". 20% is the default on a real ~100K-candidate
          dataset; this threshold should be loosened for small ad
          hoc test pools (a handful of distinct employers will
          trivially exceed it by chance).
    """
    signatures = [
        candidate_archetype_signature(c, feature_vectors[c["candidate_id"]])
        for c in top_100_candidates
    ]
    sig_counts = Counter(signatures)
    n = len(top_100_candidates)

    company_counts = Counter(c["profile"]["current_company"] for c in top_100_candidates)

    flagged_signatures = {
        sig: count for sig, count in sig_counts.items()
        if count / n > max_signature_share
    }
    flagged_companies = {
        company: count for company, count in company_counts.items()
        if count / n > max_single_company_share
    }

    most_common_sig, most_common_sig_count = sig_counts.most_common(1)[0]
    most_common_company, most_common_company_count = company_counts.most_common(1)[0]

    return {
        "n_distinct_signatures": len(sig_counts),
        "most_common_signature": most_common_sig,
        "most_common_signature_share": round(most_common_sig_count / n, 3),
        "most_common_company": most_common_company,
        "most_common_company_share": round(most_common_company_count / n, 3),
        "flagged_signatures": flagged_signatures,
        "flagged_companies": flagged_companies,
        "pass": len(flagged_signatures) == 0 and len(flagged_companies) == 0,
    }


def print_diversity_report(report: dict) -> None:
    print("=" * 60)
    print("TOP-100 DIVERSITY CHECK")
    print("=" * 60)
    print(f"Distinct archetype signatures in top 100: {report['n_distinct_signatures']}")
    print(f"Most common signature: {report['most_common_signature']} "
          f"({report['most_common_signature_share']:.1%} of top 100)")
    print(f"Most common employer: {report['most_common_company']} "
          f"({report['most_common_company_share']:.1%} of top 100)")
    if report["flagged_signatures"]:
        print("\n  WARNING -- signature(s) exceeding 25% share:")
        for sig, count in report["flagged_signatures"].items():
            print(f"    {sig}: {count} candidates")
    if report["flagged_companies"]:
        print("\n  WARNING -- employer(s) exceeding 20% share:")
        for company, count in report["flagged_companies"].items():
            print(f"    {company}: {count} candidates")
    print(f"\n  PASS: {report['pass']}")

# readiness gate

def readiness_gate(probe_ndcg10: float,
                    honeypot_result: dict,
                    diversity_result: dict,
                    ndcg10_threshold: float = 0.75) -> dict:
    """
    The single go/no-go check run immediately before spending one of
    the 3 allowed submissions. All three must pass.
    """
    checks = {
        "probe_ndcg10_meets_threshold": (
            probe_ndcg10 is not None and probe_ndcg10 >= ndcg10_threshold
        ),
        "zero_honeypot_leakage": honeypot_result["pass"],
        "top100_diversity_acceptable": diversity_result["pass"],
    }
    return {
        "checks": checks,
        "ready_to_submit": all(checks.values()),
    }


def print_readiness_report(gate_result: dict) -> None:
    print("=" * 60)
    print("SUBMISSION READINESS GATE")
    print("=" * 60)
    for check, passed in gate_result["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")
    print()
    if gate_result["ready_to_submit"]:
        print("READY TO SUBMIT.")
    else:
        print("NOT READY -- fix failing checks above before spending a submission.")



if __name__ == "__main__":
    print(__doc__)
    print(
        "This module is meant to be imported and driven by your own "
        "test harness once rank.py's pipeline function is finalized. "
        "See the four functions above: run_ablation, "
        "run_honeypot_injection_test, check_top100_diversity, and "
        "readiness_gate."
    )
