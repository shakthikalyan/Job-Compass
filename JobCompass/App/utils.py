"""
Core logic implementations:
- resume_text_extraction (simple),
- call_claude_parse (placeholder),
- job parsing,
- semantic scoring,
- gap analysis,
- recommendation logic,
- natural language query handling (local simple reasoning using parsed data)
"""

import re
from collections import defaultdict
from django.conf import settings
import json
import math

# ---------- Helpers ----------
def simple_text_from_file(file_obj):
    """
    Minimal text extractor: if PDF you'd normally use pdfminer / tika; here we read bytes and attempt utf-8.
    In production, swap with robust extractor.
    """
    try:
        b = file_obj.read()
        try:
            return b.decode('utf-8', errors='ignore')
        except Exception:
            return str(b)
    finally:
        file_obj.seek(0)

# ---------- 1. Resume Parsing Logic ----------
def build_resume_prompt(raw_text):
    """
    Build a structured JSON prompt to send to Claude (or other LLM) to extract resume fields.
    """
    prompt = {
        "task": "Parse resume into structured JSON",
        "input_text": raw_text,
        "output_schema": {
            "technical_skills": "list of strings",
            "soft_skills": "list of strings",
            "experience": [{"company": "string", "role": "string", "start": "YYYY-MM", "end": "YYYY-MM or Present", "achievements": ["string"]}],
            "projects": [{"name":"string", "technologies":["string"], "metrics":"string", "description":"string"}],
            "education": [{"degree":"string","institution":"string","year":"string"}],
            "certifications": ["string"]
        },
        "instructions": "Be concise and return only valid JSON matching the schema."
    }
    return prompt

def call_claude_parse(prompt_json):
    """
    Placeholder: send prompt_json to Claude and return parsed_json (dict).
    Replace with actual client call.
    """
    # Example pseudocode for actual implementation (do NOT run here):
    # headers = {"Authorization": f"Bearer {settings.CLAUDE_API_KEY}", "Content-Type":"application/json"}
    # resp = requests.post("https://api.claude.ai/v1/parse", json=prompt_json, headers=headers, timeout=30)
    # return resp.json()
    # For now: fallback simple heuristic extraction
    raw = prompt_json.get("input_text","")
    tech = sorted(list(set(re.findall(r'\b(?:Python|Django|Flask|React|React.js|Docker|Kubernetes|AWS|GCP|SQL|PostgreSQL|MySQL|REST|GraphQL|Machine Learning|TensorFlow|PyTorch)\b', raw, flags=re.I))), key=str.lower)
    parsed = {
        "technical_skills": tech,
        "soft_skills": ["communication", "teamwork"] if "team" in raw.lower() else ["communication"],
        "experience": [],
        "projects": [],
        "education": [],
        "certifications": []
    }
    return parsed

def parse_and_store_resume(resume_obj):
    text = simple_text_from_file(resume_obj.file)
    resume_obj.raw_text = text
    prompt = build_resume_prompt(text)
    parsed = call_claude_parse(prompt)
    resume_obj.parsed = parsed
    resume_obj.save()
    return parsed

# ---------- 2. Job Description Parsing Logic ----------
def parse_job_description_text(raw_text):
    """
    Heuristic parser: find Requirements / Preferred / Responsibilities sections.
    In production, use an LLM like Claude (similar to resume parsing).
    Output:
    {
      "title": "...",
      "critical_skills": [...],
      "beneficial_skills": [...],
      "experience_level": "Senior/Mid/Junior",
      "years_required": 3,
      "responsibilities": [...],
      "domain_keywords": [...]
    }
    """
    text = raw_text or ""
    lower = text.lower()
    critical = []
    beneficial = []
    responsibilities = []
    # find lines under headings
    def extract_section(name):
        pattern = rf'({name}[:]?)(.*?)(\n\n|\Z)'
        m = re.search(pattern, text, flags=re.I | re.S)
        return m.group(2).strip() if m else ""
    requirements = extract_section("requirements") or extract_section("must have") or ""
    preferred = extract_section("preferred") or extract_section("nice to have") or ""
    resp = extract_section("responsibilities") or extract_section("responsibility") or ""
    def skills_from_text(t):
        return list(set(re.findall(r'\b[A-Za-z\+\#\.\-]{2,20}\b', t)))
    critical = skills_from_text(requirements)[:40]
    beneficial = skills_from_text(preferred)[:40]
    responsibilities = [line.strip() for line in resp.splitlines() if line.strip()][:30]
    years = None
    m = re.search(r'(\d+)\+?\s+years?', text, flags=re.I)
    if m:
        years = int(m.group(1))
    level = "Unknown"
    if 'senior' in lower: level='Senior'
    elif 'mid' in lower or 'experienced' in lower: level='Mid'
    elif 'junior' in lower or 'fresher' in lower: level='Junior'
    domain_keywords = list(set(re.findall(r'\b(fintech|healthcare|e-commerce|cloud|security|ml|data science|embedded|iot)\b', lower)))
    return {
        "title": (text.splitlines()[0][:200] if text.splitlines() else "Job"),
        "critical_skills": critical,
        "beneficial_skills": beneficial,
        "experience_level": level,
        "years_required": years,
        "responsibilities": responsibilities,
        "domain_keywords": domain_keywords
    }

def parse_and_store_job(job_obj):
    text = job_obj.raw_text or ""
    if job_obj.file and not text.strip():
        text = simple_text_from_file(job_obj.file)
    job_obj.raw_text = text
    parsed = parse_job_description_text(text)
    job_obj.parsed = parsed
    job_obj.title = parsed.get('title') or job_obj.title
    job_obj.save()
    return parsed

# ---------- 3. Semantic Match Score Logic ----------
# We will implement the weighted scoring you specified:
# critical skills 60%, beneficial 20%, experience 15%, projects 5%
# scoring scale 0-10. We'll produce breakdown.

# normalization map / equivalences
EQUIVALENCES = {
    "docker": ["containerization", "containers"],
    "react": ["react.js", "reactjs"],
    "python": ["py"],
    "flask": ["rest api", "rest"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud"],
    "postgresql": ["postgres"]
}

def normalize_token(tok):
    t = tok.lower().strip()
    if t.endswith('s'):
        t = t[:-1]
    return t

def is_equivalent(skill, candidate_skills):
    s = normalize_token(skill)
    cand = [normalize_token(c) for c in candidate_skills]
    if s in cand:
        return True, 1.0  # exact
    for k,v in EQUIVALENCES.items():
        if s==k:
            for alt in v:
                if alt in cand:
                    return True, 1.0
    # check transferable (substring)
    for c in cand:
        if s in c or c in s:
            return True, 0.5
    return False, 0.0

def semantic_match_score(resume_parsed, job_parsed):
    # Compute points then scale to 0-10
    resume_skills = resume_parsed.get("technical_skills", []) + resume_parsed.get("soft_skills", [])
    resume_projects = resume_parsed.get("projects", [])
    job_crit = job_parsed.get("critical_skills", [])
    job_benef = job_parsed.get("beneficial_skills", [])
    points = 0.0
    max_points = 0.0
    breakdown = {"critical":{}, "beneficial":{}, "experience":{}, "projects":{}}
    # Critical (60%)
    crit_weight = 0.6
    for s in job_crit:
        max_points += crit_weight * 1.5
        matched, score = is_equivalent(s, resume_skills)
        if matched and score==1.0:
            points += crit_weight * 1.5
            breakdown['critical'][s] = "+1.5 (exact)"
        elif matched and score==0.5:
            points += crit_weight * 0.5
            breakdown['critical'][s] = "+0.5 (transferable)"
        else:
            points += crit_weight * -1.0
            breakdown['critical'][s] = "-1 (missing)"
    # Beneficial (20%)
    ben_weight = 0.2
    for s in job_benef:
        max_points += ben_weight * 0.5
        matched,score = is_equivalent(s, resume_skills)
        if matched:
            points += ben_weight * 0.5
            breakdown['beneficial'][s] = "+0.5"
        else:
            breakdown['beneficial'][s] = "0"
    # Experience (15%)
    exp_weight = 0.15
    years_req = job_parsed.get("years_required")
    exp_pts = 0.0
    # naive: if experience in resume contains number of years? fallback to role count
    resume_exps = resume_parsed.get("experience", [])
    resume_years = None
    # attempt to compute years from durations if available
    def extract_years_from_date(s):
        if not s: return None
        m = re.match(r'(\d{4})', s)
        if m:
            return int(m.group(1))
        return None
    # fallback: number of roles as proxy
    if years_req:
        resume_years = 0
        for r in resume_exps:
            start = extract_years_from_date(r.get('start'))
            end = extract_years_from_date(r.get('end'))
            if start and end:
                resume_years += max(0, end - start)
        if resume_years >= years_req:
            exp_pts += exp_weight * 1.0
            breakdown['experience'] = f"Meets years required ({resume_years} >= {years_req}) +1"
        elif resume_years >= max(0, years_req-1):
            exp_pts += exp_weight * 0.5
            breakdown['experience'] = f"Close match ({resume_years} ~ {years_req}) +0.5"
        else:
            exp_pts -= exp_weight * 0.5
            breakdown['experience'] = f"Under-experienced ({resume_years} < {years_req}) -0.5"
    else:
        # match on experience level label
        req_level = job_parsed.get("experience_level","Unknown").lower()
        resume_level_guess = "unknown"
        # crude heuristic: number of roles
        cnt = len(resume_exps)
        if cnt >=4:
            resume_level_guess='senior'
        elif cnt>=2:
            resume_level_guess='mid'
        else:
            resume_level_guess='junior'
        if resume_level_guess == req_level:
            exp_pts += exp_weight * 1.0
            breakdown['experience'] = f"Level matches ({resume_level_guess}) +1"
        elif resume_level_guess in ('mid','senior') and req_level in ('junior','mid'):
            exp_pts += exp_weight * 0.5
            breakdown['experience'] = f"Close enough ({resume_level_guess} vs {req_level}) +0.5"
        else:
            exp_pts -= exp_weight * 0.5
            breakdown['experience'] = f"Level mismatch ({resume_level_guess} vs {req_level}) -0.5"
    points += exp_pts
    max_points += exp_weight * 1.0
    # Projects (5%)
    proj_weight = 0.05
    proj_pts = 0.0
    for p in resume_projects:
        # check relevance by technology overlap with job critical
        techs = p.get('technologies', []) if isinstance(p, dict) else []
        overlap = 0
        for s in job_crit:
            matched,_ = is_equivalent(s, techs)
            if matched:
                overlap += 1
        if overlap >= 2:
            proj_pts += proj_weight * 1.0
            breakdown['projects'][p.get('name','project')] = "Highly relevant +1"
        elif overlap == 1:
            proj_pts += proj_weight * 0.5
            breakdown['projects'][p.get('name','project')] = "Somewhat relevant +0.5"
        else:
            breakdown['projects'][p.get('name','project')] = "Not relevant 0"
    points += proj_pts
    max_points += proj_weight * 1.0
    # Now scale to 0-10. Score = (points / max_points) * 10 but clamp.
    if max_points <= 0:
        score = 0.0
    else:
        score = (points / max_points) * 10.0
    # clamp
    score = max(0.0, min(10.0, score))
    # Rating
    if score >= 8.0:
        rating = "Excellent"
    elif score >= 6.0:
        rating = "Strong"
    elif score >= 4.0:
        rating = "Moderate"
    else:
        rating = "Weak"
    return {"score": round(score,2), "rating": rating, "breakdown": breakdown, "raw_points": points, "max_points": max_points}

# ---------- 4. Gap Analysis Logic ----------
def gap_analysis(resume_parsed, job_parsed, top_n=5):
    resume_skills = resume_parsed.get("technical_skills", []) + resume_parsed.get("soft_skills", [])
    critical = job_parsed.get("critical_skills", [])[:50]
    beneficial = job_parsed.get("beneficial_skills", [])[:50]
    gaps = []
    for s in critical:
        matched,score = is_equivalent(s, resume_skills)
        if not matched:
            # check if transferable exists (via equivalences)
            transferable = False
            for key,alts in EQUIVALENCES.items():
                if s.lower() == key and any(normalize_token(a) in [normalize_token(x) for x in resume_skills] for a in alts):
                    transferable = True
            # importance: critical -> 1.0
            importance = 1.0
            typ = 'transferable' if transferable else 'missing'
            suggestion = ""
            if transferable:
                suggestion = f"Your related experience (e.g., {', '.join(alts)}) covers {s}. Highlight cloud/container experience."
            else:
                suggestion = f"Consider learning {s}. Short actionable: 2-6 hours tutorial + small project."
            gaps.append({"skill": s, "type": typ, "importance": importance, "suggestion": suggestion})
    for s in beneficial:
        matched,_ = is_equivalent(s, resume_skills)
        if not matched:
            # treat as learnable lower importance
            gaps.append({"skill": s, "type": "learnable", "importance": 0.4, "suggestion": f"Optional but helpful: learn {s} through a short course."})
    # Prioritize by importance then learnability (we sort: importance desc)
    gaps_sorted = sorted(gaps, key=lambda x: (-x['importance'], x['type']))
    return gaps_sorted[:top_n]

# ---------- 5. Personalized Recommendation Logic ----------
def generate_personalized_recommendation(resume_parsed, job_parsed, gaps):
    """
    Choose ONE recommendation (A or B preferred).
    Use rules: If a project can be reframed (has metrics), produce Type A.
    If transferable skills exist, produce Type B.
    Otherwise propose Type C or Type D.
    """
    # Try Type A: reframe existing best project
    projects = resume_parsed.get("projects", [])
    # pick the project with metrics or most techs
    best = None
    for p in projects:
        if isinstance(p, dict) and p.get('metrics'):
            best = p; break
    if not best and projects:
        best = projects[0]
    if best:
        name = best.get('name','Project')
        techs = best.get('technologies',[])
        metrics = best.get('metrics') or "improved performance / delivered feature"
        text = f"Type A - Reframe: Change '{name}' description to 'Developed {name} using {', '.join(techs)} — {metrics}.' Emphasize metrics and responsibilities aligned to the job's top requirements."
        return {"kind":"A", "text": text}
    # Try Type B: highlight transferable
    # find a high-importance gap that mentions cloud/aws and see if resume contains GCP
    for g in gaps:
        if g['type']=='transferable':
            text = f"Type B - Highlight transferable: {g['suggestion']}. Add explicit mention in skills/summary."
            return {"kind":"B","text":text}
    # Type C: add unstated skill
    if gaps:
        s = gaps[0]['skill']
        text = f"Type C - Add unstated skill: If you have used tools related to {s}, list them explicitly (e.g., 'Version Control (Git)') to satisfy quick filters."
        return {"kind":"C", "text": text}
    # Type D: learn gap
    text = "Type D - Learn gap: Complete a short tutorial (2-8 hours) on a missing but critical technology and add the resulting small project to resume."
    return {"kind":"D", "text": text}

# ---------- 6. Natural Language Query Logic ----------
def handle_nl_query(session_parsed_resume, session_parsed_job, last_turns, query_text):
    """
    Simple local NL handler: classify intent and provide targeted answer using stored parsed data.
    We don't call LLM here; we synthesize deterministic responses.
    """
    q = query_text.lower()
    # classify
    if "what skills" in q or "which skills" in q or q.strip().startswith("skills"):
        # skill learning roadmap
        # base: check if asks for 'data science' or 'web'
        if 'data' in q and 'science' in q:
            have_python = any(normalize_token(s)=="python" for s in (session_parsed_resume.get('technical_skills',[]) if session_parsed_resume else []))
            roadmap = []
            if have_python:
                roadmap = [
                    "Pandas (2 weeks — small projects with CSV datasets)",
                    "Scikit-learn (3 weeks — classification/regression models)",
                    "Feature engineering & EDA (2 weeks)",
                    "Small end-to-end Kaggle-style project (1 month)"
                ]
            else:
                roadmap = ["Python (2-4 weeks)", "Then follow: Pandas (2 weeks), Scikit-learn (3 weeks) ..."]
            return {"intent":"skill_learning", "answer": roadmap}
        else:
            # generic skill list: show missing top criticals
            missing = []
            if session_parsed_job:
                gaps = gap_analysis(session_parsed_resume or {}, session_parsed_job, top_n=5)
                missing = [g['skill'] for g in gaps]
            return {"intent":"skill_learning", "answer": missing or ["Specify domain for tailored roadmap."]}
    if "ready" in q or "fit" in q or "apply" in q:
        # readiness check -> compute match result
        if not session_parsed_resume or not session_parsed_job:
            return {"intent":"readiness", "answer":"Need both resume and job in session for readiness check."}
        mm = semantic_match_score(session_parsed_resume, session_parsed_job)
        explanation = f"Score {mm['score']} ({mm['rating']}). Top breakdown: { {k:len(v) for k,v in mm['breakdown'].items()} }"
        return {"intent":"readiness", "answer": explanation, "detail": mm}
    if "confidence" in q:
        mm = semantic_match_score(session_parsed_resume or {}, session_parsed_job or {})
        return {"intent":"application_advice", "answer": f"Confidence rating {mm['rating']} with score {mm['score']}. Breakdown: {mm['breakdown']}"}
    # fallback
    return {"intent":"unknown", "answer":"I can (1) give a skill roadmap, (2) readiness check, (3) specific skill relevance. Try: 'Am I ready for this job?' or 'What skills for data science?'"}
