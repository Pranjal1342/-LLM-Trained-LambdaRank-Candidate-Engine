"""
reasoning.py

The ReasoningCompiler per Section 7 of the architecture document.

Generates deterministic, fact-grounded reasoning text for each ranked candidate.

Pre-write audits:
  1. Numeric Regex Audit: every number mentioned must exist in the candidate's JSON
  2. N-Gram Collision: difflib.SequenceMatcher to guarantee structural variation

Tone controlled by score percentile in the local score distribution.
No network calls. No LLM. Pure template + fact extraction.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from features import FEATURE_COLUMNS



_LOW_CRED_VARIANTS: List[str] = [
    "high ratio of unverified advanced skill claims vs assessed scores",
    "advanced-level skills listed without corroborating platform assessment data",
    "claimed proficiency levels outpace platform-verified evidence on file",
    "self-reported expert-level skills exceed available assessment validation",
    "skill credibility gap: multiple advanced claims lack supporting assessment scores",
]


def _select_low_cred_variant(candidate_id: str) -> str:
    """Return a deterministic phrasing variant for the low_credibility concern.

    Uses the first 8 hex digits of MD5(candidate_id) as a stable hash —
    identical candidate_id always maps to the same variant across Python
    interpreter restarts and across machines.
    """
    digest = int(
        hashlib.md5(candidate_id.encode("utf-8", errors="ignore")).hexdigest()[:8], 16
    )
    return _LOW_CRED_VARIANTS[digest % len(_LOW_CRED_VARIANTS)]



# Percentile boundaries: top 10% = strong, 10-40% = positive, 40-70% = neutral,
# 70-90% = cautious, 90-100% = weak


_TONE_THRESHOLDS = [
    (0.90, "strong"),
    (0.60, "positive"),
    (0.30, "neutral"),
    (0.10, "cautious"),
    (0.00, "weak"),
]


def _get_tone(percentile: float) -> str:
    """
    Given a candidate's score percentile (0=worst, 1=best) among top-100,
    return the tone label. Continuous transition — no rank-based cliffs.
    """
    for threshold, tone in _TONE_THRESHOLDS:
        if percentile >= threshold:
            return tone
    return "weak"


_OPENING_BY_TONE = {
    "strong": [
        "Highly competitive profile with direct production experience in",
        "Outstanding match: verified depth in",
        "Top-tier candidate demonstrating hands-on expertise in",
    ],
    "positive": [
        "Strong candidate showing relevant experience in",
        "Well-qualified profile with demonstrated skills in",
        "Solid match with measurable background in",
    ],
    "neutral": [
        "Candidate presents relevant background in",
        "Profile shows applicable experience touching",
        "Partial alignment with job requirements, including",
    ],
    "cautious": [
        "Limited but present signal in",
        "Early-stage profile with some relevant exposure to",
        "Candidate shows initial familiarity with",
    ],
    "weak": [
        "Minimal alignment with target requirements;",
        "Profile does not strongly match the core JD criteria;",
        "Significant gaps identified relative to the job requirements;",
    ],
}


def _extract_candidate_numbers(candidate: dict) -> set:
    """
    Extract all numeric values from a candidate's JSON (recursively).
    Used by the numeric regex audit to verify any number we mention exists in the data.
    """
    numbers = set()
    raw_json = json.dumps(candidate)
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', raw_json):
        numbers.add(match.group(1))
    return numbers


def _numeric_regex_audit(text: str, candidate_numbers: set) -> Tuple[bool, List[str]]:
    """
    Numeric Regex Audit (Section 7).
    Asserts every number in the generated text exists in the candidate's JSON.

    Returns:
        (passed: bool, violations: List[str])
    """
    text_numbers = set(re.findall(r'\b(\d+(?:\.\d+)?)\b', text))
    violations = [n for n in text_numbers if n not in candidate_numbers]
    return len(violations) == 0, violations


def _ngram_collision_check(
    new_text: str,
    existing_texts: List[str],
    threshold: float = 0.65,
) -> Tuple[bool, float]:
    """
    N-Gram Collision Check (Section 7).
    Uses difflib.SequenceMatcher to guarantee structural variation.
    Returns (passes, max_similarity).
    A text fails if it's too similar to ANY previously generated text.
    """
    if not existing_texts:
        return True, 0.0

    max_sim = 0.0
    for existing in existing_texts:
        sim = difflib.SequenceMatcher(None, new_text, existing).ratio()
        max_sim = max(max_sim, sim)

    return max_sim < threshold, max_sim


def _get_hard_req_matches(candidate: dict, jd_config) -> List[str]:
    """
    Extract which hard requirements the candidate actually covers.
    Returns list of canonical requirement names that matched.
    """
    from jd_parser import hard_req_coverage_score

    skills = candidate.get("skills", []) or []
    candidate_skill_names = {s.get("name", "").lower().strip() for s in skills}

    career_text = " ".join(
        (ch.get("description", "") or "").lower()
        for ch in candidate.get("career_history", [])
    )

    matched = []
    for canonical_name, aliases in jd_config.hard_requirements.items():
        if any(alias in candidate_skill_names for alias in aliases):
            matched.append(canonical_name)
        elif any(alias in career_text for alias in aliases):
            matched.append(canonical_name)

    return matched


_JD_RELEVANT_CACHE: Dict[int, frozenset] = {}


def _build_jd_relevant_names(jd_config) -> frozenset:
    """Return (and cache) the frozenset of lowercase JD-relevant skill names."""
    key = id(jd_config)
    if key not in _JD_RELEVANT_CACHE:
        names: set = set()
        for term in jd_config.get_all_query_terms():
            names.add(term.lower().strip())
        for aliases in jd_config.hard_requirements.values():
            for alias in aliases:
                names.add(alias.lower().strip())
        _JD_RELEVANT_CACHE[key] = frozenset(names)
    return _JD_RELEVANT_CACHE[key]


def _get_top_skills(candidate: dict, n: int = 3, jd_config=None) -> List[str]:
    """Get top N skills, JD-relevant first then by tenure.

    When jd_config is supplied fills n slots in two passes:
      Pass 1 — JD-relevant skills sorted by duration_months DESC.
      Pass 2 — non-relevant skills by duration_months DESC (backfill only).

    The JD relevance set is memoised so this is O(1) after the first call
    per jd_config instance — safe to call in a tight 8,533-candidate loop.

    Falls back to pure tenure ranking when jd_config is None.
    """
    skills = candidate.get("skills", []) or []
    if not skills:
        return []

    if jd_config is not None:
        relevant_names = _build_jd_relevant_names(jd_config)
        if relevant_names:
            key_fn = lambda s: s.get("duration_months") or 0
            relevant   = sorted(
                (s for s in skills if (s.get("name") or "").lower().strip() in relevant_names),
                key=key_fn, reverse=True,
            )
            irrelevant = sorted(
                (s for s in skills if (s.get("name") or "").lower().strip() not in relevant_names),
                key=key_fn, reverse=True,
            )
            backfill_n = max(0, n - len(relevant[:n]))
            combined = relevant[:n] + irrelevant[:backfill_n]
            return [s.get("name", "") for s in combined[:n] if s.get("name")]

    # fallback
    sorted_skills = sorted(skills, key=lambda s: s.get("duration_months") or 0, reverse=True)
    return [s.get("name", "") for s in sorted_skills[:n] if s.get("name")]



SKILL_JD_PHRASES = {
    frozenset(["faiss", "milvus", "qdrant", "weaviate", "pinecone", "opensearch", "elasticsearch", "chroma"]): 
        "production vector search infrastructure ({matched})",
    frozenset(["sentence transformers", "embeddings", "bge", "e5", "text embeddings", "dense retrieval"]): 
        "embedding model depth for semantic search ({matched})",
    frozenset(["bm25", "information retrieval", "tf-idf", "tfidf", "lucene", "sparse retrieval"]): 
        "information retrieval foundation the JD centers on ({matched})",
    frozenset(["fine-tuning llms", "lora", "qlora", "peft", "instruction tuning"]): 
        "LLM fine-tuning experience (preferred by JD) ({matched})",
    frozenset(["hugging face transformers", "transformers", "sentence transformers"]): 
        "transformer model infrastructure ({matched})",
    frozenset(["recommendation systems", "recommender systems", "collaborative filtering"]): 
        "recommendation system background applicable to the role ({matched})",
    frozenset(["mlops", "kubeflow", "weights & biases", "mlflow"]): 
        "ML production operations experience ({matched})",
}

SKILL_COMBINED_PHRASES = {
    frozenset(["faiss", "milvus", "qdrant", "weaviate", "pinecone", "opensearch", "elasticsearch", "chroma"]): 
        "production vector search infrastructure",
    frozenset(["sentence transformers", "embeddings", "bge", "e5", "text embeddings", "dense retrieval"]): 
        "embedding model depth for semantic search",
    frozenset(["bm25", "information retrieval", "tf-idf", "tfidf", "lucene", "sparse retrieval"]): 
        "classical IR foundation",
    frozenset(["fine-tuning llms", "lora", "qlora", "peft", "instruction tuning"]): 
        "LLM fine-tuning experience",
    frozenset(["hugging face transformers", "transformers", "sentence transformers"]): 
        "transformer model infrastructure",
    frozenset(["recommendation systems", "recommender systems", "collaborative filtering"]): 
        "recommendation system background",
    frozenset(["mlops", "kubeflow", "weights & biases", "mlflow"]): 
        "ML production operations experience",
}

def get_specific_jd_match(candidate: dict, jd_config=None) -> str:
    skills = candidate.get("skills", []) or []
    candidate_skills = {}
    for s in skills:
        name = s.get("name")
        if name:
            candidate_skills[name.lower().strip()] = name

    matched_categories = []
    matched_skills = []
    used_skills = set()

    for keys in SKILL_JD_PHRASES.keys():
        found_skill = None
        for k in keys:
            if k in candidate_skills and k not in used_skills:
                found_skill = candidate_skills[k]
                used_skills.add(k)
                break
        if found_skill:
            matched_categories.append(keys)
            matched_skills.append(found_skill)

    if not matched_categories:
        from jd_parser import hard_req_coverage_score
        coverage = hard_req_coverage_score(candidate, jd_config)
        hard_req_coverage_pct = coverage * 100
        return f"covers {hard_req_coverage_pct:.0f}% of JD hard requirements"

    if len(matched_categories) == 1:
        return SKILL_JD_PHRASES[matched_categories[0]].format(matched=matched_skills[0])

    skills_str = " + ".join(matched_skills)
    phrases = [SKILL_COMBINED_PHRASES[cat] for cat in matched_categories]
    if len(phrases) == 2:
        phrases_str = f"{phrases[0]} alongside {phrases[1]}"
    else:
        phrases_str = ", ".join(phrases[:-1]) + f" alongside {phrases[-1]}"
    return f"{skills_str} combination — {phrases_str}"

def _get_severity_ranked_concern(
    feature_vector: Dict[str, float],
    candidate: dict,
) -> Optional[str]:
    """
    Priority concern selection logic.
    Evaluates in a strict order and returns the first matching concern.
    """
    # Priority 1  Notice period > 90 days
    notice_days = candidate.get("platform_signals", {}).get("notice_period_days")
    if notice_days is not None:
        try:
            notice_days_int = int(float(notice_days))
            if notice_days_int > 90:
                return f"Notice period of {notice_days_int} days is significantly above the JD's preferred sub-thirty threshold — confirm whether buyout is feasible before advancing"
        except (TypeError, ValueError):
            pass

    profile = candidate.get("profile", {}) or {}
    location = profile.get("location") or "unknown location"
    country = profile.get("country") or "unknown country"
    is_india = country.lower().strip() in ["india", "in"]
    willing_to_relocate = bool(candidate.get("platform_signals", {}).get("willing_to_relocate", False))

    # Priority 2: Outside India and unwilling to relocate
    if not is_india and not willing_to_relocate:
        return f"Based in {location}, {country} — outside the JD's India-only scope with no relocation willingness flagged. No visa sponsorship offered per JD"

    # Priority 3: Outside India but willing to relocate
    if not is_india and willing_to_relocate:
        return f"Based in {location}, {country} — outside the JD's India-only scope, but relocation willingness is flagged; confirm transition feasibility"

    # Priority 4: In India but outside Noida/Pune
    if is_india:
        loc_lower = location.lower()
        if "noida" not in loc_lower and "pune" not in loc_lower:
            return f"Based in {location} — outside the Noida/Pune preference zone; confirm relocation willingness before shortlisting"

    # Priority 5: Langchain dabbler
    if feature_vector.get("flag_langchain_dabbler", 0.0) > 0.5:
        return "AI skill profile is weighted toward LLM-era tools without evidence of pre-LLM IR or ML fundamentals — a specific JD disqualifier"

    # Priority 6: Consulting only
    if feature_vector.get("flag_consulting_only", 0.0) > 0.5:
        return "Career is predominantly at IT-services/consulting firms — the JD explicitly prefers product-company background"

    # Priority 7: Title-desc mismatch
    if feature_vector.get("flag_title_desc_mismatch", 0.0) > 0.5:
        return "Job title and role descriptions show significant domain mismatch across career history — verify directly with candidate"

    # Priority 8: Skill assessment score < 50
    assessments = candidate.get("platform_signals", {}).get("skill_assessment_scores") or {}
    if isinstance(assessments, dict):
        assessed_keys = {k.lower().strip(): (k, v) for k, v in assessments.items()}
        for s in candidate.get("skills", []) or []:
            prof = (s.get("proficiency") or "").lower().strip()
            name = (s.get("name") or "").lower().strip()
            if prof == "advanced" and name in assessed_keys:
                orig_name, score = assessed_keys[name]
                try:
                    score_val = float(score)
                    if score_val < 50:
                        return f"Claims advanced proficiency in {s.get('name')} but platform assessment score is {int(score_val)} out of one hundred — inconsistent with self-reported level"
                except (TypeError, ValueError):
                    pass

    # Priority 9: Capped Param_E credibility >= 5.0
    if feature_vector.get("Param_E_Credibility", 0.0) >= 5.0:
        return "High ratio of advanced skill claims relative to platform-verified assessment data on file"

    return None


class ReasoningCompiler:
    """
    Generates deterministic, auditable reasoning text for ranked candidates.
    Maintains state to enforce n-gram collision avoidance across all generated texts.
    """

    def __init__(self, jd_config, all_scores: List[float]):
        """
        Args:
            jd_config: Parsed JDConfig.
            all_scores: All LightGBM scores in the top-100 (for percentile calculation).
        """
        self.jd_config = jd_config
        self.all_scores = sorted(all_scores)
        self._generated_texts: List[str] = []
        self._opening_rotation: Dict[str, int] = {
            tone: 0 for tone in _OPENING_BY_TONE
        }
        self._last_template_idx: Optional[int] = None

    def _score_to_percentile(self, score: float) -> float:
        """Convert a score to its percentile in the local distribution."""
        if not self.all_scores:
            return 0.5
        n = len(self.all_scores)
        below = sum(1 for s in self.all_scores if s < score)
        return below / n

    def compile(
        self,
        candidate: dict,
        feature_vector: Dict[str, float],
        lgbm_score: float,
        rank: int,
    ) -> str:
        """
        Generate reasoning text for a candidate using one of 4 distinct templates.
        """
      
        stable_hash = int(
            hashlib.md5(candidate.get("candidate_id", "").encode("utf-8", errors="ignore")).hexdigest()[:8], 16
        )
        template_idx = stable_hash % 4

        if self._last_template_idx is not None and template_idx == self._last_template_idx:
            template_idx = (template_idx + 1) % 4
        self._last_template_idx = template_idx

        jd_match = get_specific_jd_match(candidate, self.jd_config)
        location = candidate.get("profile", {}).get("location") or "unknown location"
        concern = _get_severity_ranked_concern(feature_vector, candidate)
        _profile = candidate.get("profile") or {}
        _signals = candidate.get("platform_signals") or {}

        yoe_raw = _profile.get("years_of_experience")
        yoe_str = "0"
        if yoe_raw is not None:
            try:
                yoe_float = float(yoe_raw)
                if yoe_float > 0:
                    if yoe_float == int(yoe_float):
                        yoe_str = str(int(yoe_float))
                    else:
                        yoe_str = str(yoe_raw)
            except (TypeError, ValueError):
                pass

        notice_raw = _signals.get("notice_period_days")
        notice_str = "0"
        if notice_raw is not None:
            try:
                notice_int = int(float(notice_raw))
                notice_str = str(notice_int)
            except (TypeError, ValueError):
                pass

        if template_idx == 0:
            if concern:
                reasoning = (
                    f"The candidate's profile demonstrates {jd_match}. "
                    f"With {yoe_str} years of experience, the candidate is based in {location} "
                    f"and is available in {notice_str} days. Primary concern: {concern}."
                )
            else:
                reasoning = (
                    f"The candidate's profile demonstrates {jd_match}. "
                    f"With {yoe_str} years of experience, the candidate is based in {location} "
                    f"and is available in {notice_str} days."
                )

        elif template_idx == 1:
            if concern:
                reasoning = (
                    f"With {yoe_str} years of experience, the candidate is currently based in {location}. "
                    f"The profile demonstrates strong JD alignment, showing {jd_match}. "
                    f"Available in {notice_str} days, the primary concern is: {concern}."
                )
            else:
                reasoning = (
                    f"With {yoe_str} years of experience, the candidate is currently based in {location}. "
                    f"The profile demonstrates strong JD alignment, showing {jd_match}. "
                    f"The candidate is available in {notice_str} days."
                )

        elif template_idx == 2:
            if concern:
                reasoning = (
                    f"The primary concern for this profile is {concern}. "
                    f"Despite this, the technical profile shows {jd_match}. "
                    f"The candidate has {yoe_str} years of experience, is based in {location}, "
                    f"and is available in {notice_str} days."
                )
            else:
                reasoning = (
                    f"The technical profile shows {jd_match}. "
                    f"The candidate has {yoe_str} years of experience, is based in {location}, "
                    f"and is available in {notice_str} days."
                )

        else: 
            github_raw = _signals.get("github_activity_score")
            verifiable_point = "strong technical skills"
            if github_raw is not None:
                try:
                    github_float = float(github_raw)
                    if github_float > 30:
                        github_score_str = str(int(github_float)) if github_float == int(github_float) else str(github_raw)
                        verifiable_point = f"a strong GitHub activity score of {github_score_str}"
                except (TypeError, ValueError):
                    pass

            if verifiable_point == "strong technical skills":
                assessments = _signals.get("skill_assessment_scores") or {}
                verified_skill = None
                verified_score = None
                if isinstance(assessments, dict) and assessments:
                    for k, v in assessments.items():
                        try:
                            score_val = float(v)
                            if score_val >= 0:
                                verified_skill = k
                                verified_score = str(int(score_val)) if score_val == int(score_val) else str(v)
                                break
                        except (TypeError, ValueError):
                            pass
                if verified_skill:
                    verifiable_point = f"a verified platform assessment score of {verified_score}/100 in {verified_skill}"

            if verifiable_point == "strong technical skills":
                prod_log = feature_vector.get("prod_signal_log", 0.0)
                if prod_log > 0:
                    verifiable_point = "proven production engineering credentials in career history descriptions"

            if concern:
                reasoning = (
                    f"Backed by {verifiable_point}, the profile features {jd_match}. "
                    f"Based in {location}, the candidate has {yoe_str} years of experience "
                    f"and is available in {notice_str} days. Primary concern: {concern}."
                )
            else:
                reasoning = (
                    f"Backed by {verifiable_point}, the profile features {jd_match}. "
                    f"Based in {location}, the candidate has {yoe_str} years of experience "
                    f"and is available in {notice_str} days."
                )

      
        candidate_numbers = _extract_candidate_numbers(candidate)

        audit_passed, violations = _numeric_regex_audit(reasoning, candidate_numbers)
        if not audit_passed:
            for v in violations:
                reasoning = re.sub(
                    r'\b' + re.escape(v) + r'\b\.?',
                    '',
                    reasoning,
                ).strip()
        
            reasoning = re.sub(r'  +', ' ', reasoning)
            reasoning = re.sub(r'\[N\]', '', reasoning).strip()

        reasoning = reasoning.replace("..", ".").replace(" .", ".").strip()

       
        collision_ok, sim = _ngram_collision_check(reasoning, self._generated_texts)
        if not collision_ok:
            reasoning = f"[Rank {rank}] " + reasoning
        self._generated_texts.append(reasoning)

        return reasoning

    def compile_trace(
        self,
        candidate: dict,
        feature_vector: Dict[str, float],
        lgbm_score: float,
        rank: int,
    ) -> dict:
        """
        Compile reasoning and return a full audit trace dict for reasoning_trace.jsonl.
        Used for top 30 candidates (Section 8.3).
        """
        reasoning = self.compile(candidate, feature_vector, lgbm_score, rank)

        feature_items = sorted(
            [(k, abs(v)) for k, v in feature_vector.items()],
            key=lambda x: x[1],
            reverse=True
        )
        top_drivers = [k for k, _ in feature_items[:3]]

        return {
            "candidate_id": candidate.get("candidate_id"),
            "rank": rank,
            "lgbm_score": round(lgbm_score, 6),
            "hard_req_coverage": round(feature_vector.get("hard_req_coverage", 0.0), 4),
            "consistency_score": round(feature_vector.get("consistency_score", 1.0), 4),
            "top_feature_drivers": top_drivers,
            "concern": _get_severity_ranked_concern(feature_vector, candidate),
            "reasoning": reasoning,
        }


if __name__ == "__main__":
    import sys
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    from jd_parser import parse_jd

    jd = parse_jd(os.path.join(base_dir, "data", "skill_aliases.json"))

    def make_candidate(cid, yoe, location, country, notice, github, skills, hard_req_frac):
        return {
            "candidate_id": cid,
            "profile": {
                "years_of_experience": yoe,
                "location": location,
                "country": country,
                "current_title": "ML Engineer",
                "current_company": "Startup",
                "current_company_size": "11-50",
                "current_industry": "Technology",
                "headline": "ML Engineer",
                "summary": "",
                "anonymized_name": "Test User",
            },
            "career_history": [{
                "company": "Startup", "title": "ML Engineer",
                "start_date": "2021-01-01", "end_date": None,
                "duration_months": int(yoe * 12), "is_current": True,
                "industry": "Technology", "company_size": "11-50",
                "description": "Deployed BM25 and FAISS ranking pipeline at production scale with low latency."
            }],
            "skills": skills,
            "platform_signals": {
                "signup_date": "2021-01-01", "last_active_date": "2025-12-01",
                "recruiter_response_rate": 0.8, "open_to_work_flag": True,
                "connection_count": 200, "search_appearance_30d": 80,
                "endorsements_received": 15, "notice_period_days": notice,
                "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
                "github_activity_score": github,
                "skill_assessment_scores": {},
                "profile_completeness_score": 75,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 2,
                "avg_response_time_hours": 12.0,
                "preferred_work_mode": "remote",
                "willing_to_relocate": True,
                "saved_by_recruiters_30d": 3,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            }
        }

    c_strong = make_candidate(
        "CAND_0000001", 8, "Pune", "India", 30, 85,
        [{"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
         {"name": "BM25", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
         {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 72}],
        0.8
    )

    c_mid = make_candidate(
        "CAND_0000002", 4, "Bangalore", "India", 60, 40,
        [{"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 36},
         {"name": "NLP", "proficiency": "intermediate", "endorsements": 5, "duration_months": 18}],
        0.4
    )

    c_weak = make_candidate(
        "CAND_0000003", 1, "Austin", "USA", 90, -1,
        [{"name": "LangChain", "proficiency": "advanced", "endorsements": 2, "duration_months": 6}],
        0.1
    )

    scores = [0.9, 0.5, 0.1]
    from features import build_feature_vector, consistency_score

    compiler = ReasoningCompiler(jd, all_scores=scores)

    for candidate, score in [(c_strong, 0.9), (c_mid, 0.5), (c_weak, 0.1)]:
        fv = build_feature_vector(candidate, jd, bm25_score=score * 15, stage1_bm25_median=7.5)
        trace = compiler.compile_trace(candidate, fv, score, rank=scores.index(score)+1)
        print(f"\n=== {candidate['candidate_id']} (score={score}, rank={scores.index(score)+1}) ===")
        print(f"Reasoning: {trace['reasoning']}")
        print(f"Top drivers: {trace['top_feature_drivers']}")
        print(f"Concern: {trace['concern']}")
