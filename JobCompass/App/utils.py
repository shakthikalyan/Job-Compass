
# App/utils.py

import os
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from App.models import OccupationSkillRelation, Skill
from App import ai_client
from App.utilsMain import call_claude_parse


log = logging.getLogger(__name__)


RESUME_PARSE_SYSTEM = """
You are a resume parsing engine.

STRICT RULES:
- Output MUST be a single valid JSON object
- DO NOT add explanations, comments, or text outside JSON
- DO NOT use markdown or ``` blocks
- If information is missing, use empty strings or empty arrays
- DO NOT infer or guess skills
- DO NOT apologize or explain limitations

If you violate any rule, the output is invalid.
"""

RESUME_PARSE_USER = """
Extract structured information from the resume text below.

Return ONLY a JSON object in EXACTLY this schema:

{
  "summary": string,
  "technical_skills": [string],
  "soft_skills": [string],
  "experience": [
    {
      "title": string,
      "organization": string,
      "duration": string,
      "description": string
    }
  ],
  "projects": [
    {
      "title": string,
      "description": string,
      "technologies": [string]
    }
  ],
  "education": [string]
}

Rules:
- Use empty arrays if sections are not found
- Do NOT add text outside JSON
- Do NOT use markdown

Resume text:
{{RESUME_TEXT}}
"""



JD_PARSE_SYSTEM = """
You are an expert job description analyst.
Classify skills strictly based on job requirements.
Return VALID JSON ONLY.
"""

JD_PARSE_USER = """
Parse the following job description and return structured JSON.

Schema:
{
  "job_title": string,
  "role_category": string,
  "critical_skills": [string],
  "beneficial_skills": [string],
  "responsibilities": [string],
  "years_required": number | null
}

Rules:
- Critical skills = mandatory requirements
- Beneficial skills = preferred or nice-to-have
- Do not hallucinate skills

Job Description:
{{JOB_TEXT}}
"""

GAP_REASONING_SYSTEM = """
You are a career coach.
You give precise, non-generic resume advice.
Return VALID JSON ONLY.
"""

GAP_REASONING_USER = """
Given the following information:

Job Critical Skills:
{{JOB_SKILLS}}

Resume Skills:
{{RESUME_SKILLS}}

Missing or Weak Skills:
{{GAPS}}

Return JSON in this format:

{
  "recommendations": [
    {
      "skill": string,
      "tip": string
    }
  ]
}

Rules:
- Give 1–2 concise resume-strengthening tips per skill
- Tips must be actionable (project, certification, bullet rewrite)
"""



# -------------------- MODELS --------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_llm_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
import re

def safe_json_loads(text: str):
    """
    Safely extract and parse JSON from LLM output.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    text = text.strip()

    # If model wrapped JSON in markdown
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text)
        text = re.sub(r"```$", "", text)

    # Extract JSON block if extra text exists
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    return json.loads(text)

def safe_json_parse(text: str) -> dict:
    """
    Safely parse JSON from LLM output.
    Returns empty dict on failure instead of crashing.
    """
    if not text or not text.strip():
        return {}

    text = text.strip()

    # Remove markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text)
        text = re.sub(r"```$", "", text)

    # Extract JSON object if extra text exists
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except Exception:
        log.error("Invalid JSON from LLM:\n%s", text[:500])
        return {}

# -------------------- EMBEDDINGS --------------------

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    vecs = _llm_embedding_model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

def cosine_similarity(a, b):
    return np.dot(a, b.T)

# -------------------- LLM CALL WRAPPER --------------------

def call_llama_raw(system, user, temperature=0.2, max_tokens=600):
    response = ai_client.call_llm(
        system=system,
        prompt=user,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.get("text", "")

def call_llama_json(system, user, temperature=0.2, max_tokens=600):
    raw = call_llama_raw(system, user, temperature, max_tokens)
    return safe_json_parse(raw)


# -------------------- PARSING --------------------

MAX_INPUT_CHARS = 8000  # ~2000 tokens, safe for Groq
SKILL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.# ]{1,40}")

def extract_skills_from_text(text: str) -> list[str]:
    """
    Deterministic skill extraction grounded in DB.
    """
    if not text:
        return []

    text = text.lower()
    tokens = set(SKILL_TOKEN_RE.findall(text))

    db_skills = Skill.objects.all().values_list("normalized", flat=True)

    matched = []
    for sk in db_skills:
        if sk and sk in text:
            matched.append(sk)

    return list(set(matched))


def sanitize_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\x00", "")

def truncate_text(text: str) -> str:
    return (text or "")[:MAX_INPUT_CHARS]


def parse_job_llm(job_text):
    safe_text = truncate_text(job_text)
    parsed = call_llama_json(
        JD_PARSE_SYSTEM,
        JD_PARSE_USER.replace("{{JOB_TEXT}}", safe_text),
        max_tokens=500
    )
    parsed.setdefault("critical_skills", [])
    parsed.setdefault("beneficial_skills", [])
    parsed.setdefault("responsibilities", [])
    return parsed


# -------------------- SEMANTIC MATCHING --------------------
def get_weighted_job_skills(occupation):
    """
    Returns {skill_name: weight_0_to_1}
    """
    if not occupation:
        return {}

    rels = OccupationSkillRelation.objects.filter(
        occupation=occupation,
        importance__gt=0
    ).select_related("skill")

    weights = {}
    for r in rels:
        weights[r.skill.name] = r.importance / 100.0

    return weights

def experience_score(resume, job):
    exp = resume.get("experience", [])
    if not exp:
        return 0.0

    job_text = " ".join(job.get("responsibilities", [])).lower()

    hits = 0
    for e in exp:
        desc = (e.get("description") or "").lower()
        if any(k in desc for k in job_text.split()):
            hits += 1

    return min(hits / len(exp), 1.0)

def project_score(resume, job):
    projects = resume.get("projects", [])
    if not projects:
        return 0.0

    job_skills = job.get("critical_skills", [])
    if not job_skills:
        return 0.0

    hits = 0
    for p in projects:
        tech = " ".join(p.get("technologies", [])).lower()
        if any(s.lower() in tech for s in job_skills):
            hits += 1

    return min(hits / len(projects), 1.0)

def semantic_match_score(resume, job, occupation=None):
    resume_skills = resume.get("technical_skills", [])
    job_skills = job.get("critical_skills", [])

    if not resume_skills or not job_skills:
        return {"score": 0, "rating": "Weak", "breakdown": {}}

    r_emb = embed(resume_skills)
    j_emb = embed(job_skills)

    sims = cosine_similarity(j_emb, r_emb)
    best = sims.max(axis=1)

    # 🔹 O*NET weights
    weights = get_weighted_job_skills(occupation)

    weighted_score = 0.0
    total_weight = 0.0

    for i, skill in enumerate(job_skills):
        w = weights.get(skill, 0.5)  # fallback
        total_weight += w
        weighted_score += w * max(best[i], 0)

    skill_component = (weighted_score / max(total_weight, 1)) * 0.60

    exp_component = experience_score(resume, job) * 0.25
    proj_component = project_score(resume, job) * 0.15

    final_score = round((skill_component + exp_component + proj_component) * 10, 1)

    rating = (
        "Strong" if final_score >= 8 else
        "Moderate" if final_score >= 6 else
        "Fair" if final_score >= 4 else
        "Weak"
    )

    return {
        "score": final_score,
        "rating": rating,
        "breakdown": {
            "skills": round(skill_component * 10, 1),
            "experience": round(exp_component * 10, 1),
            "projects": round(proj_component * 10, 1)
        }
    }


# -------------------- GAP ANALYSIS --------------------

def gap_analysis(resume, job):
    resume_skills = resume.get("technical_skills", [])
    job_skills = job.get("critical_skills", [])

    gaps = []

    if not job_skills:
        return gaps

    if not resume_skills:
        return [
            {
                "skill": skill,
                "type": "missing",
                "importance": 1.0,
                "suggestion": f"Add evidence of {skill} through projects or experience."
            }
            for skill in job_skills
        ]

    r_emb = embed(resume_skills)
    j_emb = embed(job_skills)

    if r_emb.size == 0 or j_emb.size == 0:
        return gaps

    sims = cosine_similarity(j_emb, r_emb)
    best = sims.max(axis=1)

    for i, skill in enumerate(job_skills):
        if best[i] < 0.65:
            gaps.append({
                "skill": skill,
                "type": "missing",
                "importance": round(1.0 - float(best[i]), 2),
                "suggestion": f"Add evidence of {skill} through projects or experience."
            })

    return gaps


# -------------------- RECOMMENDATIONS --------------------

def generate_recommendations(resume, job, gaps):
    if not gaps:
        return []

    gap_skills = [g["skill"] for g in gaps[:3]]

    payload = GAP_REASONING_USER \
        .replace("{{JOB_SKILLS}}", ", ".join(job.get("critical_skills", [])[:10])) \
        .replace("{{RESUME_SKILLS}}", ", ".join(resume.get("technical_skills", [])[:15])) \
        .replace("{{GAPS}}", ", ".join(gap_skills))

    result = call_llama_json(
        GAP_REASONING_SYSTEM,
        payload,
        max_tokens=400
    )

    return result.get("recommendations", [])


# -------------------- NL CHATBOT --------------------

def nl_query(query, resume, job, score, gaps):
    system = "You answer career questions using limited context."

    context_summary = f"""
Job Title: {job.get('job_title')}
Critical Skills: {', '.join(job.get('critical_skills', [])[:10])}
Resume Skills: {', '.join(resume.get('technical_skills', [])[:15])}
Match Score: {score.get('score', 'N/A')}
Missing Skills: {', '.join(gaps[:5])}
"""

    user = f"""
Context:
{context_summary}

User Question:
{query}
"""

    return call_llama_raw(system, user, temperature=0.3, max_tokens=400)
