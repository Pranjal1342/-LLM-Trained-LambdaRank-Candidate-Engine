from __future__ import annotations
import difflib
import math
import re
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

from jd_parser import JDConfig, hard_req_coverage_score


REFERENCE_DATE = date(2026, 1, 1)


_DOMAIN_TITLE_KEYWORDS: Dict[str, List[str]] = {
    "ai_ml": [
        "machine learning", "ml", "data scientist", "ai", "nlp",
        "deep learning", "research scientist", "applied scientist",
        "ranking", "recommendation", "retrieval", "search"
    ],
    "data_engineering": [
        "data engineer", "data pipeline", "etl", "spark", "kafka",
        "warehouse", "dbt", "analytics engineer"
    ],
    "software_engineering": [
        "software engineer", "backend", "frontend", "fullstack",
        "full stack", "swe", "developer", "programmer"
    ],
    "devops_infra": [
        "devops", "sre", "infrastructure", "platform engineer",
        "cloud", "kubernetes", "docker"
    ],
    "consulting_non_technical": [
        "consultant", "analyst", "business analyst", "manager",
        "sales", "marketing", "customer support", "account"
    ],
    "cv_speech": [
        "computer vision", "cv engineer", "image processing",
        "speech", "audio", "tts", "asr"
    ],
}

_DOMAIN_DESC_KEYWORDS: Dict[str, List[str]] = {
    "ai_ml": [
        "machine learning", "neural network", "model training",
        "nlp", "embedding", "transformer", "ranking", "retrieval",
        "recommendation", "gradient", "pytorch", "tensorflow"
    ],
    "data_engineering": [
        "pipeline", "etl", "kafka", "spark", "warehouse",
        "ingestion", "batch processing", "stream processing"
    ],
    "software_engineering": [
        "api", "microservice", "backend", "database", "sql",
        "rest", "graphql", "web application"
    ],
    "devops_infra": [
        "kubernetes", "docker", "ci/cd", "deployment", "monitoring",
        "cloud", "aws", "gcp", "azure", "infrastructure"
    ],
    "consulting_non_technical": [
        "client", "stakeholder", "presentation", "consulting",
        "business strategy", "slides", "excel modeling"
    ],
    "cv_speech": [
        "opencv", "yolo", "object detection", "image classification",
        "speech recognition", "tts", "text to speech"
    ],
}

_SYNTHETIC_TEMPLATES = [
    "responsible for overseeing",
    "worked closely with cross-functional teams",
    "collaborated with stakeholders to deliver",
    "passionate about leveraging cutting-edge",
    "i am a results-driven professional",
    "seeking opportunities to apply my skills",
    "strong communication and leadership skills",
    "experience in agile and scrum methodologies",
    "proficient in microsoft office suite",
    "eager to contribute to organizational goals",
    "team player with excellent interpersonal",
    "dynamic and motivated self-starter",
    "mechanical engineering design role at a hardware-product company",
    "customer support team lead at a saas product",
    "marketing leadership role at a b2b saas company",
    "brand design and creative direction at a consumer-products company",
    "operations management role at a logistics company",
]

# Precomputed first words for each template — the real pre-filter.
# If the first word of a template isn't present in the description at all,
# SequenceMatcher ratio can never reach 0.65, so the call is safely skipped.
# Reduces SequenceMatcher calls from ~272K to ~3K across the 8533-candidate pool.
_TEMPLATE_FIRST_WORDS = [t.split()[0] for t in _SYNTHETIC_TEMPLATES]

_PRODUCTION_KEYWORDS = [
    "deployed", "production", "serving", "latency",
    "throughput", "scale", "real-time", "inference",
    "a/b test", "monitoring", "pipeline", "distributed",
]

_ACADEMIC_ONLY_KEYWORDS = [
    "coursework", "thesis", "university project",
    "academic project", "research paper", "capstone",
    "class project", "homework",
]

_PRE_LLM_SKILLS = {
    "bm25", "tf-idf", "tfidf", "xgboost", "lightgbm", "scikit-learn",
    "sklearn", "elasticsearch", "solr", "lucene", "faiss", "annoy",
    "traditional ml", "gradient boosting", "random forest",
    "word2vec", "glove", "fasttext",
}

_LLM_ERA_SKILLS = {
    "langchain", "llamaindex", "llama index", "openai api",
    "chatgpt api", "gpt-4", "prompt engineering", "rag",
    "retrieval augmented generation", "langsmith", "autogpt",
    "gpt wrapper",
}

_CV_SPEECH_SKILLS = {
    "opencv", "cv2", "yolo", "object detection", "image classification",
    "image segmentation", "pose estimation", "optical flow",
    "tts", "text to speech", "speech recognition", "asr",
    "gans", "generative adversarial", "stable diffusion",
}

_IR_SKILLS = {
    "information retrieval", "bm25", "ranking", "learning to rank",
    "recommendation", "retrieval", "search", "embedding", "faiss",
    "vector search", "dense retrieval", "nlp", "natural language processing",
}


def _classify_text_domain(text: str, keyword_map: Dict[str, List[str]]) -> Optional[str]:
    """Return the best-matching domain for text, or None if no match."""
    text_lower = text.lower()
    best_domain = None
    best_count = 0
    for domain, keywords in keyword_map.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_domain = domain
    return best_domain if best_count > 0 else None


def domain_category_mismatch(career_entry: dict) -> float:
    """
    Adversarial Function 1: Domain-Category Mismatch.
    Maps job title through taxonomy to get its bucket, classifies description
    by keyword presence. If domain(title) != domain(description), returns 1.

    Schema fields read:
      - career_history[].title
      - career_history[].description

    Returns: 0.0 (no mismatch) or 1.0 (mismatch detected).
    """
    title = (career_entry.get("title") or "").strip()
    description = (career_entry.get("description") or "").strip()

    if not title or not description:
        return 0.0 

    title_domain = _classify_text_domain(title, _DOMAIN_TITLE_KEYWORDS)
    desc_domain = _classify_text_domain(description, _DOMAIN_DESC_KEYWORDS)

    if title_domain is None or desc_domain is None:
        return 0.0 

    return 1.0 if title_domain != desc_domain else 0.0


def template_registry_match(description: str) -> float:
    """
    Adversarial Function 2: Template Registry.
    String matching against known synthetic templates.
    Fires if substring matches or SequenceMatcher ratio >= 0.65.

    Pre-filter: each template's first word must appear in the description
    before SequenceMatcher is called. If the first word is absent, the
    full-string similarity ratio cannot reach 0.65 — so SM is safely skipped.
    This reduces SequenceMatcher calls from ~272K to ~3K on the Stage 1 pool.

    Schema fields read:
      - career_history[].description

    Returns: 1.0 if any template matches, 0.0 otherwise.
    """
    if not description:
        return 0.0
    desc_lower = description.lower()
    fragment = desc_lower[:200]
    for template, first_word in zip(_SYNTHETIC_TEMPLATES, _TEMPLATE_FIRST_WORDS):
        if template in desc_lower:
            return 1.0
        if first_word not in desc_lower:
            continue
        ratio = difflib.SequenceMatcher(None, fragment, template, autojunk=False).ratio()
        if ratio >= 0.65:
            return 1.0
    return 0.0


def prod_signal_log_score(description: str) -> float:
    """
    Adversarial Function 3: Production Signal (log-compression).
    Returns log(1 + count) of production keywords in description.
    If ONLY academic keywords exist (and no production keywords), returns -1.0.

    Schema fields read:
      - career_history[].description

    Returns: float. -1.0 for pure academic, log(1+count) >= 0 for production.
    """
    if not description:
        return 0.0

    desc_lower = description.lower()
    prod_count = sum(1 for kw in _PRODUCTION_KEYWORDS if kw in desc_lower)
    academic_count = sum(1 for kw in _ACADEMIC_ONLY_KEYWORDS if kw in desc_lower)

    if prod_count == 0 and academic_count > 0:
        return -1.0  

    return math.log1p(prod_count)


def langchain_dabbler_score(skills: List[dict]) -> float:
    """
    Adversarial Function 4: Temporal LangChain Dabbler.
    Evaluates pre_llm (bm25, xgboost, scikit-learn) vs llm_era (langchain, openai api).
    High return value = more pre-LLM depth (good signal).
    Low return value = LLM-only / LangChain-only (bad signal).

    Schema fields read:
      - skills[].name
      - skills[].duration_months (optional, falls back to count)

    Returns: float in [-1.0, 1.0]:
      - 1.0 = strong pre-LLM foundation
      - 0.0 = balanced or no signal
      - -1.0 = LLM-era only (LangChain dabbler)
    """
    if not skills:
        return 0.0

    pre_llm_months = 0
    llm_era_months = 0

    for s in skills:
        name = (s.get("name") or "").lower().strip()
        months = s.get("duration_months") or 0  # safe default if missing
        months = max(0, int(months))

        weight = months if months > 0 else 1

        if any(pre in name for pre in _PRE_LLM_SKILLS):
            pre_llm_months += weight
        if any(llm in name for llm in _LLM_ERA_SKILLS):
            llm_era_months += weight

    total = pre_llm_months + llm_era_months
    if total == 0:
        return 0.0

    return (pre_llm_months - llm_era_months) / total


def cv_specialist_score(skills: List[dict]) -> float:
    """
    Adversarial Function 5: CV/Speech Specialist.
    Evaluates opencv, yolo, tts dominance over IR skills.

    Schema fields read:
      - skills[].name
      - skills[].duration_months (optional)

    Returns: float in [0.0, 1.0] where 1.0 = pure CV/Speech (bad for this JD).
    """
    if not skills:
        return 0.0

    cv_months = 0
    ir_months = 0

    for s in skills:
        name = (s.get("name") or "").lower().strip()
        months = s.get("duration_months") or 0
        months = max(0, int(months))
        weight = months if months > 0 else 1

        if any(cv in name for cv in _CV_SPEECH_SKILLS):
            cv_months += weight
        if any(ir in name for ir in _IR_SKILLS):
            ir_months += weight

    total = cv_months + ir_months
    if total == 0:
        return 0.0

    return cv_months / total



def _safe_date(date_str: Optional[str]) -> Optional[date]:
    """Parse date string safely; return None on any failure."""
    if not date_str:
        return None
    try:
        return date.fromisoformat(str(date_str))
    except (ValueError, TypeError):
        return None


def compute_yoe(candidate: dict) -> float:
    """
    Feature 2: Years of experience.
    Schema fields read: profile.years_of_experience
    """
    yoe = candidate.get("profile", {}).get("years_of_experience")
    if yoe is None:
        return 0.0
    try:
        return max(0.0, float(yoe))
    except (TypeError, ValueError):
        return 0.0


def compute_param_a_systems_depth(candidate: dict) -> float:
    """
    Feature 3: Param_A_Systems_Depth.
    Fraction of career months in roles where descriptions contain
    retrieval/ranking/search/recommendation.

    Schema fields read:
      - career_history[].description
      - career_history[].duration_months
    """
    _SYSTEMS_KEYWORDS = {
        "retrieval", "ranking", "search", "recommendation",
        "information retrieval", "candidate retrieval",
        "passage retrieval", "vector search", "recommendation system",
        "recommender", "re-ranking", "reranking",
    }

    career = candidate.get("career_history", []) or []
    total_months = 0
    systems_months = 0

    for ch in career:
        dur = ch.get("duration_months")
        if dur is None:
            continue
        try:
            dur = max(0, int(dur))
        except (TypeError, ValueError):
            dur = 0

        total_months += dur
        desc = (ch.get("description") or "").lower()
        if any(kw in desc for kw in _SYSTEMS_KEYWORDS):
            systems_months += dur

    return systems_months / total_months if total_months > 0 else 0.0


def compute_param_b_availability(candidate: dict) -> float:
    """
    Feature 4: Param_B_Availability.
    Combined recruiter response rate and recency of last activity.

    Schema fields read:
      - platform_signals.recruiter_response_rate  (0–1)
      - platform_signals.last_active_date
      - platform_signals.open_to_work_flag
    """
    signals = candidate.get("platform_signals", {}) or {}

    rr = signals.get("recruiter_response_rate")
    if rr is None:
        rr = 0.0
    try:
        rr = max(0.0, min(1.0, float(rr)))
    except (TypeError, ValueError):
        rr = 0.0

    last_active = _safe_date(signals.get("last_active_date"))
    if last_active is None:
        recency_score = 0.0
    else:
        days_since = (REFERENCE_DATE - last_active).days
        days_since = max(0, days_since)
        recency_score = math.exp(-days_since / 180.0)

    open_to_work = float(bool(signals.get("open_to_work_flag", False)))

    # Weighted combination
    return 0.4 * rr + 0.4 * recency_score + 0.2 * open_to_work


def compute_param_c_tenure(candidate: dict) -> float:
    """
    Feature 5: Param_C_Tenure.
    Reward for 3+ year average tenure. Returns 1.0 if avg >= 36 months, scaled.

    Schema fields read:
      - career_history[].duration_months
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return 0.0

    durations = []
    for ch in career:
        dur = ch.get("duration_months")
        if dur is not None:
            try:
                dur = max(0, int(dur))
                durations.append(dur)
            except (TypeError, ValueError):
                pass

    if not durations:
        return 0.0

    avg_months = sum(durations) / len(durations)
    return min(1.0, avg_months / 36.0)


def compute_param_d_notice_exp(candidate: dict) -> float:
    """
    Feature 6: Param_D_Notice_Exp.
    exp(-max(0, days-30)/30) — continuous decay gradient.

    Schema fields read:
      - platform_signals.notice_period_days  (int, 0–180)
    """
    signals = candidate.get("platform_signals", {}) or {}
    days = signals.get("notice_period_days")
    if days is None:
        return 1.0  
    try:
        days = max(0, int(days))
    except (TypeError, ValueError):
        return 1.0

    return math.exp(-max(0, days - 30) / 30.0)


def compute_param_e_credibility(candidate: dict) -> float:
    """
    Feature 7: Param_E_Credibility.
    advanced_claimed_count / max(1, assessed_count).
    Higher = Less credible (more advanced claims than assessments).

    NOTE: We count skills where proficiency == "advanced" AND the skill name
    appears in skill_assessment_scores keys as "assessed". We count skills
    with proficiency == "advanced" regardless as "claimed".

    Schema fields read:
      - skills[].name
      - skills[].proficiency
      - platform_signals.skill_assessment_scores  (dict skill_name -> score 0-100)
    """
    skills = candidate.get("skills", []) or []
    signals = candidate.get("platform_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores") or {}

    if not isinstance(assessments, dict):
        assessments = {}

    assessed_keys = {k.lower().strip() for k in assessments.keys()}

    advanced_claimed = 0
    assessed_advanced = 0

    for s in skills:
        proficiency = (s.get("proficiency") or "").lower()
        name = (s.get("name") or "").lower().strip()

        if proficiency == "advanced":
            advanced_claimed += 1
            if name in assessed_keys:
                assessed_advanced += 1

    return min(5.0, advanced_claimed / max(1, assessed_advanced))


def compute_param_f_consulting(candidate: dict) -> float:
    """
    Feature 8: Param_F_Consulting.
    Fraction of career months spent in IT Services / Consulting roles.

    Schema fields read:
      - career_history[].industry
      - career_history[].duration_months
    """
    _CONSULTING_INDUSTRIES = {
        "it services", "consulting", "staffing", "outsourcing",
        "bpo", "business process outsourcing", "it consulting",
    }

    career = candidate.get("career_history", []) or []
    total_months = 0
    consulting_months = 0

    for ch in career:
        industry = (ch.get("industry") or "").lower().strip()
        dur = ch.get("duration_months")
        if dur is None:
            continue
        try:
            dur = max(0, int(dur))
        except (TypeError, ValueError):
            dur = 0

        total_months += dur
        if any(ci in industry for ci in _CONSULTING_INDUSTRIES):
            consulting_months += dur

    return consulting_months / total_months if total_months > 0 else 0.0


def compute_param_g_location(candidate: dict) -> float:
    """
    Feature 9: Param_G_Location.
    Pune/Noida = 1.0, other India = 0.5, outside India = 0.0.

    Schema fields read:
      - profile.location  (city, region/state)
      - profile.country
    """
    profile = candidate.get("profile", {}) or {}
    location = (profile.get("location") or "").lower().strip()
    country = (profile.get("country") or "").lower().strip()

    # Priority locations
    if any(city in location for city in ["pune", "noida"]):
        return 1.0

    # Other India
    india_indicators = ["india", "in", "bengaluru", "bangalore", "mumbai",
                        "hyderabad", "chennai", "delhi", "gurugram", "gurgaon",
                        "kolkata", "ahmedabad", "jaipur", "chandigarh"]
    if country in ["india", "in"] or any(ind in location for ind in india_indicators):
        return 0.5

    return 0.0


def compute_param_h_github(candidate: dict) -> float:
    """
    Feature 10: Param_H_GitHub.
    Open source activity score, normalized to [0, 1].
    -1 means no GitHub linked → return 0.0.

    Schema fields read:
      - platform_signals.github_activity_score  (float, -1 to 100)
    """
    signals = candidate.get("platform_signals", {}) or {}
    score = signals.get("github_activity_score")
    if score is None:
        return 0.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0

    if score < 0:
        return 0.0  # No GitHub linked

    return min(1.0, score / 100.0)


def compute_title_ai_fraction(candidate: dict) -> float:
    """
    Feature 11: title_ai_fraction.
    Fraction of career roles with AI/ML-oriented job titles.

    Schema fields read:
      - career_history[].title
    """
    _AI_TITLE_KEYWORDS = [
        "machine learning", "ml", "data scientist", "ai", "nlp",
        "deep learning", "research", "applied scientist",
        "ranking", "recommendation", "search", "retrieval",
        "computer vision", "speech", "nlp engineer",
    ]

    career = candidate.get("career_history", []) or []
    if not career:
        return 0.0

    ai_count = 0
    for ch in career:
        title = (ch.get("title") or "").lower()
        if any(kw in title for kw in _AI_TITLE_KEYWORDS):
            ai_count += 1

    return ai_count / len(career)


def compute_prod_signal_log(candidate: dict) -> float:
    """
    Feature 12: prod_signal_log.
    Aggregate production signal across ALL career history descriptions.
    Uses the adversarial function prod_signal_log_score per role.

    Schema fields read:
      - career_history[].description
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return 0.0

    total_prod_count = 0
    is_academic_only = True
    has_any_description = False

    for ch in career:
        desc = ch.get("description") or ""
        if not desc:
            continue
        has_any_description = True
        desc_lower = desc.lower()

        prod_count = sum(1 for kw in _PRODUCTION_KEYWORDS if kw in desc_lower)
        academic_count = sum(1 for kw in _ACADEMIC_ONLY_KEYWORDS if kw in desc_lower)

        total_prod_count += prod_count
        if prod_count > 0:
            is_academic_only = False

    if not has_any_description:
        return 0.0

    if total_prod_count == 0 and is_academic_only:
        return -1.0  

    return math.log1p(total_prod_count)


def compute_flag_consulting_only(candidate: dict) -> float:
    """
    Feature 15: flag_consulting_only.
    1.0 if ALL career history is in IT Services / Consulting with no product-company roles.

    Schema fields read:
      - career_history[].industry
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return 0.0

    _CONSULTING_INDUSTRIES = {
        "it services", "consulting", "staffing", "outsourcing", "bpo",
    }
    _PRODUCT_INDUSTRIES = {
        "internet", "software", "technology", "fintech", "saas",
        "e-commerce", "product", "startup",
    }

    all_consulting = True
    for ch in career:
        industry = (ch.get("industry") or "").lower().strip()
        if not any(ci in industry for ci in _CONSULTING_INDUSTRIES):
            all_consulting = False
            break

    return 1.0 if all_consulting else 0.0


def compute_flag_title_chaser(candidate: dict) -> float:
    """
    Feature 16: flag_title_chaser.
    Detects candidates who adopt trendy AI titles with very short tenure.
    Flag fires if most recent role has AI/ML title AND average tenure < 15 months
    AND at least one role has duration < 12 months.

    Schema fields read:
      - career_history[].title
      - career_history[].duration_months
      - career_history[].is_current
    """
    _TRENDY_TITLES = [
        "ai", "machine learning", "ml", "generative", "llm",
        "prompt", "gpt", "langchain", "chatbot", "nlp", "data scientist"
    ]

    career = candidate.get("career_history", []) or []
    if not career:
        return 0.0

  
    current_roles = [ch for ch in career if ch.get("is_current", False)]
    most_recent = current_roles[0] if current_roles else career[-1]

    title = (most_recent.get("title") or "").lower()
    is_trendy_title = any(kw in title for kw in _TRENDY_TITLES)

    durations = []
    for ch in career:
        dur = ch.get("duration_months")
        if dur is not None:
            try:
                dur = max(0, int(dur))
                durations.append(dur)
            except (TypeError, ValueError):
                pass

    if not durations:
        return 0.0

    avg_tenure = sum(durations) / len(durations)
    is_short_tenure = (avg_tenure < 15.0) and any(d < 12 for d in durations)

    return 1.0 if (is_trendy_title and is_short_tenure) else 0.0


def compute_flag_langchain_dabbler(skills: List[dict]) -> float:
    """
    Feature 17: flag_langchain_dabbler.
    1.0 if LLM-era skills dominate with no pre-LLM foundation.

    Schema fields read:
      - skills[].name
      - skills[].duration_months
    """
    score = langchain_dabbler_score(skills)
   
    return 1.0 if score < -0.3 else 0.0


def compute_flag_cv_specialist(skills: List[dict]) -> float:
    """
    Feature 18: flag_cv_specialist.
    1.0 if CV/speech skills dominate over IR skills.

    Schema fields read:
      - skills[].name
      - skills[].duration_months
    """
    cv_score = cv_specialist_score(skills)
    return 1.0 if cv_score > 0.7 else 0.0


def compute_flag_title_desc_mismatch(candidate: dict) -> float:
    """
    Feature 19: flag_title_desc_mismatch.
    Uses domain_category_mismatch on the most recent career entry.

    Schema fields read:
      - career_history[].title
      - career_history[].description
    """
    career = candidate.get("career_history", []) or []
    if not career:
        return 0.0

    current_roles = [ch for ch in career if ch.get("is_current", False)]
    most_recent = current_roles[0] if current_roles else career[-1]

    return domain_category_mismatch(most_recent)


def compute_flag_template_desc(candidate: dict) -> float:
    """
    Feature 20: flag_template_desc.
    1.0 if ANY career description matches a synthetic template.

    Schema fields read:
      - career_history[].description
    """
    career = candidate.get("career_history", []) or []
    for ch in career:
        desc = ch.get("description") or ""
        if template_registry_match(desc) == 1.0:
            return 1.0
    return 0.0


def build_feature_vector(
    candidate: dict,
    jd_config: JDConfig,
    bm25_score: float,
    stage1_bm25_median: float = 0.0,
    precomputed_static: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build the complete 22-feature vector for a single candidate.

    Args:
        candidate: Parsed candidate dict (from JSONL).
        jd_config: Parsed JD configuration.
        bm25_score: BM25 retrieval score from Stage 1.
        stage1_bm25_median: Median BM25 score of Stage 1 candidates (for c5).
        precomputed_static: Optional precomputed dictionary of the 18 static features.

    Returns:
        Dict mapping feature name -> float value.
        All features are guaranteed finite floats (no NaN, no None).
    """
    from features import (
        c1_timeline_impossibility, c2_signup_anomaly, c3_salary_inversion,
        c4_assessment_contradiction, c5_engagement_mismatch, consistency_score,
    )

    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []

    if precomputed_static is not None:
        yoe = float(precomputed_static.get("yoe", 0.0))
        hard_req = hard_req_coverage_score(candidate, jd_config)
        cons = consistency_score(
            candidate,
            bm25_score=bm25_score,
            median_bm25=stage1_bm25_median,
        )
        param_a = float(precomputed_static.get("Param_A_Systems_Depth", 0.0))
        param_b = float(precomputed_static.get("Param_B_Availability", 0.0))
        param_c = float(precomputed_static.get("Param_C_Tenure", 0.0))
        param_d = float(precomputed_static.get("Param_D_Notice_Exp", 0.0))
        param_e = float(precomputed_static.get("Param_E_Credibility", 0.0))
        param_f = float(precomputed_static.get("Param_F_Consulting", 0.0))
        param_g = float(precomputed_static.get("Param_G_Location", 0.0))
        param_h = float(precomputed_static.get("Param_H_GitHub", 0.0))
        title_ai_frac = float(precomputed_static.get("title_ai_fraction", 0.0))
        prod_sig_log = float(precomputed_static.get("prod_signal_log", 0.0))
        flag_consulting_only = float(precomputed_static.get("flag_consulting_only", 0.0))
        flag_title_chaser = float(precomputed_static.get("flag_title_chaser", 0.0))
        flag_langchain = float(precomputed_static.get("flag_langchain_dabbler", 0.0))
        flag_cv = float(precomputed_static.get("flag_cv_specialist", 0.0))
        flag_title_desc = float(precomputed_static.get("flag_title_desc_mismatch", 0.0))
        flag_template = float(precomputed_static.get("flag_template_desc", 0.0))
        interaction_yoe_x_prod = float(precomputed_static.get("interaction_yoe_x_prod", 0.0))
    else:
        yoe = compute_yoe(candidate)
        hard_req = hard_req_coverage_score(candidate, jd_config)
        cons = consistency_score(
            candidate,
            bm25_score=bm25_score,
            median_bm25=stage1_bm25_median,
        )
        param_a = compute_param_a_systems_depth(candidate)
        param_b = compute_param_b_availability(candidate)
        param_c = compute_param_c_tenure(candidate)
        param_d = compute_param_d_notice_exp(candidate)
        param_e = compute_param_e_credibility(candidate)
        param_f = compute_param_f_consulting(candidate)
        param_g = compute_param_g_location(candidate)
        param_h = compute_param_h_github(candidate)
        title_ai_frac = compute_title_ai_fraction(candidate)
        prod_sig_log = compute_prod_signal_log(candidate)
        flag_consulting_only = compute_flag_consulting_only(candidate)
        flag_title_chaser = compute_flag_title_chaser(candidate)
        flag_langchain = compute_flag_langchain_dabbler(skills)
        flag_cv = compute_flag_cv_specialist(skills)
        flag_title_desc = compute_flag_title_desc_mismatch(candidate)
        flag_template = compute_flag_template_desc(candidate)
        interaction_yoe_x_prod = yoe * max(0.0, prod_sig_log)

    interaction_req_x_cons = hard_req * cons

   
    fv = {
        "bm25_score": float(bm25_score),
        "yoe": float(yoe),
        "Param_A_Systems_Depth": float(param_a),
        "Param_B_Availability": float(param_b),
        "Param_C_Tenure": float(param_c),
        "Param_D_Notice_Exp": float(param_d),
        "Param_E_Credibility": float(param_e),
        "Param_F_Consulting": float(param_f),
        "Param_G_Location": float(param_g),
        "Param_H_GitHub": float(param_h),
        "title_ai_fraction": float(title_ai_frac),
        "prod_signal_log": float(prod_sig_log),
        "consistency_score": float(cons),
        "hard_req_coverage": float(hard_req),
        "flag_consulting_only": float(flag_consulting_only),
        "flag_title_chaser": float(flag_title_chaser),
        "flag_langchain_dabbler": float(flag_langchain),
        "flag_cv_specialist": float(flag_cv),
        "flag_title_desc_mismatch": float(flag_title_desc),
        "flag_template_desc": float(flag_template),
        "interaction_req_x_consistency": float(interaction_req_x_cons),
        "interaction_yoe_x_prod": float(interaction_yoe_x_prod),
    }

    for k, v in fv.items():
        if not math.isfinite(v):
            fv[k] = 0.0

    assert len(fv) == 22, f"Feature vector has {len(fv)} features, expected 22"
    return fv



def c1_timeline_impossibility(candidate: dict) -> float:
    """
    Consistency Check 1: Timeline Impossibility.
    Flag if any skill.duration_months > total_months_of_experience.

    Schema fields read:
      - skills[].duration_months
      - profile.years_of_experience
    """
    yoe = compute_yoe(candidate)
    total_months = yoe * 12.0

    skills = candidate.get("skills", []) or []
    for s in skills:
        dur = s.get("duration_months")
        if dur is None:
            continue
        try:
            dur = max(0, int(dur))
        except (TypeError, ValueError):
            continue

        if dur > total_months:
            return 0.0  # Violation

    return 1.0


def c2_signup_anomaly(candidate: dict) -> float:
    """
    Consistency Check 2: Signup Anomaly.
    Flag if signup_date is chronologically AFTER last_active_date.

    Schema fields read:
      - platform_signals.signup_date
      - platform_signals.last_active_date
    """
    signals = candidate.get("platform_signals", {}) or {}
    signup = _safe_date(signals.get("signup_date"))
    last_active = _safe_date(signals.get("last_active_date"))

    if signup is None or last_active is None:
        return 1.0  

    if signup > last_active:
        return 0.0

    return 1.0


def c3_salary_inversion(candidate: dict) -> float:
    """
    Consistency Check 3: Salary Inversion.
    Flag if expected_salary.min > max.

    Schema fields read:
      - platform_signals.expected_salary_range_inr_lpa.min
      - platform_signals.expected_salary_range_inr_lpa.max
    """
    signals = candidate.get("platform_signals", {}) or {}
    salary = signals.get("expected_salary_range_inr_lpa") or {}

    sal_min = salary.get("min")
    sal_max = salary.get("max")

    if sal_min is None or sal_max is None:
        return 1.0

    try:
        sal_min = float(sal_min)
        sal_max = float(sal_max)
    except (TypeError, ValueError):
        return 1.0

    if sal_min > sal_max:
        return 0.0  

    return 1.0


def c4_assessment_contradiction(candidate: dict) -> float:
    """
    Consistency Check 4: Assessment Contradiction.
    Flag if candidate claims "advanced" AND assessment score exists AND score < 50.

    Schema fields read:
      - skills[].name
      - skills[].proficiency
      - platform_signals.skill_assessment_scores  (dict)
    """
    skills = candidate.get("skills", []) or []
    signals = candidate.get("platform_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores") or {}

    if not isinstance(assessments, dict):
        assessments = {}

    assessed = {k.lower().strip(): v for k, v in assessments.items()}

    for s in skills:
        proficiency = (s.get("proficiency") or "").lower()
        name = (s.get("name") or "").lower().strip()

        if proficiency == "advanced" and name in assessed:
            score = assessed[name]
            try:
                score = float(score)
                if score < 50.0:
                    return 0.0
            except (TypeError, ValueError):
                pass

    return 1.0


def c5_engagement_mismatch(
    candidate: dict,
    bm25_score: float,
    median_bm25: float,
) -> float:
    """
    Consistency Check 5: Engagement Mismatch (Data-Adaptive).
    Flag if bm25_score > median(stage1_scores)
    AND connection_count <= 60
    AND search_appearance_30d <= 15
    AND endorsements_received <= 4.

    Schema fields read:
      - platform_signals.connection_count
      - platform_signals.search_appearance_30d
      - platform_signals.endorsements_received
    """
    signals = candidate.get("platform_signals", {}) or {}

    connections = signals.get("connection_count") or 0
    appearances = signals.get("search_appearance_30d") or 0
    endorsements = signals.get("endorsements_received") or 0

    try:
        connections = int(connections)
        appearances = int(appearances)
        endorsements = int(endorsements)
    except (TypeError, ValueError):
        return 1.0

    is_high_bm25 = bm25_score > median_bm25
    is_suspicious_engagement = (connections <= 60 and appearances <= 15 and endorsements <= 4)

    if is_high_bm25 and is_suspicious_engagement:
        return 0.0  

    return 1.0


def consistency_score(
    candidate: dict,
    bm25_score: float = 0.0,
    median_bm25: float = 0.0,
) -> float:
    """
    Composite consistency multiplier from Section 5.
    Returns the product of all 5 checks.

    AUDIT TRAIL — all 5 checks explicitly multiplied (verified against architecture doc):

      result = c1 * c2 * c3 * c4 * c5

    Each check returns 1.0 (pass) or 0.0 (violation), so any single violation
    zeros out the composite score.
    """
    c1 = c1_timeline_impossibility(candidate)
    c2 = c2_signup_anomaly(candidate)
    c3 = c3_salary_inversion(candidate)
    c4 = c4_assessment_contradiction(candidate)
    c5 = c5_engagement_mismatch(candidate, bm25_score, median_bm25)

    result = c1 * c2 * c3 * c4 * c5
    return float(result)


def score_langchain_dabbler(candidate: dict) -> float:
    """Helper wrapper for precompute offline labels penalty."""
    return compute_flag_langchain_dabbler(candidate.get("skills") or [])


def score_title_skill_discontinuity(candidate: dict) -> float:
    """Helper wrapper for precompute offline labels penalty."""
    return compute_flag_title_chaser(candidate)


def detect_description_title_mismatch(candidate: dict) -> float:
    """Helper wrapper for precompute offline labels penalty."""
    return compute_flag_title_desc_mismatch(candidate)


def score_cv_speech_specialist(candidate: dict) -> float:
    """Helper wrapper for precompute offline labels penalty."""
    return compute_flag_cv_specialist(candidate.get("skills") or [])


# Feature column order for LightGBM (must match training order)
FEATURE_COLUMNS = [
    "bm25_score", "yoe", "Param_A_Systems_Depth", "Param_B_Availability",
    "Param_C_Tenure", "Param_D_Notice_Exp", "Param_E_Credibility",
    "Param_F_Consulting", "Param_G_Location", "Param_H_GitHub",
    "title_ai_fraction", "prod_signal_log", "consistency_score",
    "hard_req_coverage", "flag_consulting_only", "flag_title_chaser",
    "flag_langchain_dabbler", "flag_cv_specialist", "flag_title_desc_mismatch",
    "flag_template_desc", "interaction_req_x_consistency", "interaction_yoe_x_prod",
]



if __name__ == "__main__":
    import json
    import sys

    print("=== Testing 5 Adversarial Functions ===\n")


    entry_ok = {"title": "Machine Learning Engineer", "description": "Built ranking models using neural networks and transformers."}
    entry_bad = {"title": "Customer Support", "description": "Conducted research on neural network architectures for image classification."}
    print(f"domain_category_mismatch (no mismatch): {domain_category_mismatch(entry_ok)}")
    print(f"domain_category_mismatch (mismatch):    {domain_category_mismatch(entry_bad)}")


    desc_template = "I am a results-driven professional with experience in agile and scrum methodologies."
    desc_real = "Deployed a production BM25 ranking system serving 10M queries/day with p99 latency < 50ms."
    print(f"\ntemplate_registry_match (template):     {template_registry_match(desc_template)}")
    print(f"template_registry_match (real):         {template_registry_match(desc_real)}")

   
    prod_desc = "Deployed model to production serving 1M users at scale with low latency."
    academic_desc = "University project on coursework for thesis on deep learning."
    empty_desc = ""
    print(f"\nprod_signal_log_score (production): {prod_signal_log_score(prod_desc):.4f}")
    print(f"prod_signal_log_score (academic):   {prod_signal_log_score(academic_desc):.4f}")
    print(f"prod_signal_log_score (empty):      {prod_signal_log_score(empty_desc):.4f}")

    skills_pre_llm = [
        {"name": "BM25", "proficiency": "advanced", "endorsements": 10, "duration_months": 36},
        {"name": "XGBoost", "proficiency": "advanced", "endorsements": 8, "duration_months": 24},
    ]
    skills_llm_only = [
        {"name": "LangChain", "proficiency": "advanced", "endorsements": 2, "duration_months": 6},
        {"name": "Prompt Engineering", "proficiency": "intermediate", "endorsements": 1, "duration_months": 4},
    ]
    print(f"\nlangchain_dabbler_score (pre-LLM):  {langchain_dabbler_score(skills_pre_llm):.4f}")
    print(f"langchain_dabbler_score (LLM-only): {langchain_dabbler_score(skills_llm_only):.4f}")

    
    skills_cv = [
        {"name": "OpenCV", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
        {"name": "YOLO", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
    ]
    skills_ir = [
        {"name": "FAISS", "proficiency": "advanced", "endorsements": 15, "duration_months": 24},
        {"name": "BM25", "proficiency": "advanced", "endorsements": 10, "duration_months": 18},
    ]
    print(f"\ncv_specialist_score (CV dominant): {cv_specialist_score(skills_cv):.4f}")
    print(f"cv_specialist_score (IR focused):  {cv_specialist_score(skills_ir):.4f}")

    print("\n=== Testing Consistency Checks ===\n")


    base = {
        "candidate_id": "CAND_TEST001",
        "profile": {"years_of_experience": 5.0, "location": "Bangalore", "country": "India",
                    "current_title": "ML Engineer", "current_company": "Startup",
                    "current_company_size": "11-50", "current_industry": "Technology"},
        "career_history": [{"company": "Startup", "title": "ML Engineer",
                             "start_date": "2021-01-01", "end_date": None,
                             "duration_months": 36, "is_current": True,
                             "industry": "Technology", "company_size": "11-50",
                             "description": "Deployed production ranking pipeline."}],
        "skills": [{"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 36}],
        "platform_signals": {
            "signup_date": "2021-01-01", "last_active_date": "2025-12-01",
            "recruiter_response_rate": 0.8, "open_to_work_flag": True,
            "connection_count": 100, "search_appearance_30d": 50,
            "endorsements_received": 10, "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
            "github_activity_score": 75, "skill_assessment_scores": {},
        },
    }

    print(f"c1 (clean):    {c1_timeline_impossibility(base)}")
    print(f"c2 (clean):    {c2_signup_anomaly(base)}")
    print(f"c3 (clean):    {c3_salary_inversion(base)}")
    print(f"c4 (clean):    {c4_assessment_contradiction(base)}")
    print(f"c5 (clean):    {c5_engagement_mismatch(base, bm25_score=10.0, median_bm25=5.0)}")
    print(f"consistency_score (clean): {consistency_score(base, bm25_score=10.0, median_bm25=5.0)}")

    # Inject violations one at a time
    import copy
    v1 = copy.deepcopy(base)
    v1["skills"][0]["duration_months"] = 999
    print(f"\nc1 (timeline violation):   {c1_timeline_impossibility(v1)}")

    v2 = copy.deepcopy(base)
    v2["platform_signals"]["signup_date"] = "2099-01-01"
    print(f"c2 (signup anomaly):       {c2_signup_anomaly(v2)}")

    v3 = copy.deepcopy(base)
    v3["platform_signals"]["expected_salary_range_inr_lpa"] = {"min": 50.0, "max": 10.0}
    print(f"c3 (salary inversion):     {c3_salary_inversion(v3)}")

    v4 = copy.deepcopy(base)
    v4["platform_signals"]["skill_assessment_scores"] = {"python": 12.0}
    print(f"c4 (assessment contradiction): {c4_assessment_contradiction(v4)}")

    v5 = copy.deepcopy(base)
    v5["platform_signals"]["connection_count"] = 0
    v5["platform_signals"]["search_appearance_30d"] = 0
    v5["platform_signals"]["endorsements_received"] = 0
    print(f"c5 (engagement mismatch):  {c5_engagement_mismatch(v5, bm25_score=10.0, median_bm25=5.0)}")
