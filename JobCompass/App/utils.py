# App/utils.py
import re
import os
import numpy as np
from django.db.models import Q
from django.conf import settings

# optional libs
try:
    import PyPDF2
except Exception:
    PyPDF2 = None


try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# Lazy import Skill to avoid circular import at migration time
try:
    from .models import Skill
except Exception:
    Skill = None


#
try:
    import pdfplumber
except Exception:
    pdfplumber = None

COMMON_SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving", "adaptability",
    "time management", "collaboration", "presentation", "mentoring", "project management",
    "critical thinking", "research", "empathy", "creativity", "work ethic", "conflict resolution",
    "decision making", "interpersonal skills", "organization", "attention to detail",
    "flexibility", "active listening", "negotiation", "stress management", "public speaking",
    "customer service", "coaching", "networking", "strategic planning", "multitasking", "motivation",
    "team leadership", "innovation", "analytical thinking", "dependability", "patience", "positivity",
    "self-motivation", "collaborative problem solving",

]

SECTION_HEADINGS = [
    r'(?i)experience', r'(?i)work experience', r'(?i)professional experience',
    r'(?i)skills', r'(?i)technical skills', r'(?i)design tools', r'(?i)design skills',
    r'(?i)research skills',r'(?i)key skills', r'(?i)projects', r'(?i)education', r'(?i)certifications',
    r'(?i)volunteer', r'(?i)awards', r'(?i)summary', r'(?i)profile', r'(?i)contact'
]

def simple_text_from_file(file_obj):
    """
    Improved text extraction:
    - use pdfplumber when available (better on modern/resume PDFs),
    - fallback to PyPDF2 (existing),
    - fallback to raw bytes decode.
    """
    try:
        # reset pointer
        file_obj.seek(0)
    except Exception:
        pass

    text = ""
    # Try pdfplumber first (better extraction for design PDFs)
    if pdfplumber and hasattr(file_obj, 'name') and file_obj.name.lower().endswith('.pdf'):
        try:
            file_obj.seek(0)
            with pdfplumber.open(file_obj) as pdf:
                pages = []
                for p in pdf.pages:
                    try:
                        p_text = p.extract_text() or ""
                        pages.append(p_text)
                    except Exception:
                        pages.append("")
                text = "\n\n".join(pages)
            text = text.replace('\x00', '')
            if text.strip():
                return text
        except Exception:
            # fall through to other methods
            pass

    # fallback to PyPDF2 if available
    if hasattr(file_obj, 'name') and file_obj.name.lower().endswith('.pdf') and PyPDF2:
        try:
            file_obj.seek(0)
            reader = PyPDF2.PdfReader(file_obj)
            pages = []
            for page in reader.pages:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:
                    pages.append("")
            text = "\n\n".join(pages)
            text = text.replace('\x00', '')
            if text.strip():
                return text
        except Exception:
            pass

    # fallback: read bytes and decode
    try:
        file_obj.seek(0)
        raw = file_obj.read()
        try:
            text = raw.decode('utf-8', errors='ignore')
        except Exception:
            text = str(raw)
        text = text.replace('\x00', '')
    except Exception:
        text = ""
    return text

def normalize_token(tok):
    t = (tok or "").lower().strip()
    if t.endswith('s'):
        t = t[:-1]
    return t

# ---------- KB helpers (ESCO seeded Skill table) ----------
def kb_lookup(token):
    """
    Return Skill object, canonical name, and matched synonyms if any.
    Returns (None, None, []) if not found or DB unavailable.
    """
    if Skill is None:
        return None, None, []
    t = normalize_token(token)
    skill = Skill.objects.filter(Q(normalized=t) | Q(name__iexact=token)).first()
    if skill:
        syns = skill.synonyms or []
        matched_syns = [s for s in syns if normalize_token(s) == t or t in normalize_token(s) or normalize_token(s) in t]
        return skill, skill.name, matched_syns
    # Try synonyms contains lookup (best-effort)
    try:
        skill = Skill.objects.filter(synonyms__icontains=[token]).first()
        if skill:
            return skill, skill.name, [token]
    except Exception:
        # Linear fallback (limited)
        for s in Skill.objects.all()[:10000]:
            for syn in s.synonyms or []:
                if normalize_token(syn) == t or t in normalize_token(syn) or normalize_token(syn) in t:
                    return s, s.name, [syn]
    return None, None, []

# local fallback equivalences (kept for dev fallback)
EQUIVALENCES = {
    "docker": ["containerization", "containers"],
    "react": ["react.js", "reactjs"],
    "python": ["py"],
    "flask": ["rest api", "rest"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud"],
    "postgresql": ["postgres"]
}

def is_equivalent(skill, candidate_skills):
    """
    KB-aware equivalence check. Returns (matched:bool, score:float)
    score: 1.0 exact (KB or exact), 0.8 KB-synonym partial, 0.5 transferable substring, 0.0 none.
    """
    s_norm = normalize_token(skill)
    cand = [normalize_token(c) for c in candidate_skills if c]

    # KB lookup
    kb_skill, canonical, matched_syns = kb_lookup(skill)
    if kb_skill:
        variants = {normalize_token(canonical)}
        for syn in (kb_skill.synonyms or []):
            variants.add(normalize_token(syn))
        for c in cand:
            if c in variants:
                return True, 1.0
        # partial via synonyms
        for c in cand:
            for v in variants:
                if v in c or c in v:
                    return True, 0.8

    # local equivalences fallback
    for key, alts in EQUIVALENCES.items():
        if s_norm == normalize_token(key):
            for alt in alts:
                if normalize_token(alt) in cand:
                    return True, 1.0

    # substring transferable
    for c in cand:
        if s_norm in c or c in s_norm:
            return True, 0.5

    return False, 0.0

# ---------- embedding helpers ----------
EMBEDDING_MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_MODEL = None
if SentenceTransformer is not None:
    try:
        EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        EMBEDDING_MODEL = None

SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.65))

def embed_texts(texts):
    if EMBEDDING_MODEL is None:
        return None
    if texts is None:
        return None
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, EMBEDDING_MODEL.get_sentence_embedding_dimension()))
    return EMBEDDING_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def cosine_sim(a, b):
    if a is None or b is None:
        return None
    try:
        return util.cos_sim(a, b).cpu().numpy()
    except Exception:
        # numpy fallback
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

# ---------- parsing helpers ----------

def build_resume_prompt(raw_text):
    # keep a simple wrapper for potential LLM calls.
    return {
        "task": "Parse resume into structured JSON",
        "input_text": raw_text or ""
    }

def _split_sections_by_headings(text):
    """
    Naive section splitter: returns ordered dict {heading: content}
    It finds known headings in the text and slices by their positions.
    """
    import collections
    s_text = text.replace('\r', '\n')
    # normalize multiple blank lines
    s_text = re.sub(r'\n{2,}', '\n\n', s_text)
    # find headings (case-insensitive) by scanning lines
    lines = s_text.splitlines()
    indexes = []
    for i, line in enumerate(lines):
        clean = line.strip()
        for h in SECTION_HEADINGS:
            # use regex search
            if re.fullmatch(h + r'[:\s]*', clean, flags=re.I):
                indexes.append((i, clean))
                break
        # also handle lines that are ALL CAPS and short (like company headings) - not used for section split
    # if no explicit headings, return full text as 'body'
    if not indexes:
        return {"body": s_text}
    # build sections by slicing between heading indices
    sections = collections.OrderedDict()
    for idx, (line_idx, heading_line) in enumerate(indexes):
        start = line_idx + 1
        end = indexes[idx + 1][0] if idx + 1 < len(indexes) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        # normalize heading name
        heading_norm = re.sub(r'[:\s]+$', '', heading_line).strip()
        sections[heading_norm] = content
    return sections

def _extract_contact_info(text):
    """
    Extract email, phone, website, name heuristically.
    Returns dict.
    """
    contact = {}
    # email
    em = re.search(r'([A-Za-z0-9.\-_+]+@[A-Za-z0-9\-_]+\.[A-Za-z0-9.\-_]+)', text)
    if em:
        contact['email'] = em.group(1)
    # phone (simple patterns - international optional)
    ph = re.search(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})', text)
    if ph:
        contact['phone'] = "".join([g for g in ph.groups() if g]).strip()
    # website
    web = re.search(r'(https?://[^\s,;]+|www\.[^\s,;]+)', text)
    if web:
        contact['website'] = web.group(1)
    # name heuristic: first non-empty line at top that looks like a name (2 words capitalized)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        top = lines[0]
        # if top contains a known label like "resume" skip
        if len(top.split()) <= 4 and re.search(r'[A-Z][a-z]+', top):
            contact['name'] = top
        else:
            # try second line
            if len(lines) > 1 and len(lines[1].split()) <= 4:
                contact['name'] = lines[1]
    return contact

def _lines_to_skills(content):
    """
    Turn a block of content into a list of skills by splitting on commas, bullets and slashes.
    Keep tokens of reasonable length and deduplicate.
    """
    if not content:
        return []
    # replace bullet characters
    content = content.replace('•', '\n').replace('·', '\n').replace('–', '-')
    # split into candidate lines
    cand = []
    for part in re.split(r'[\n;/\u2022]', content):
        part = part.strip()
        if not part:
            continue
        # if comma-separated in same line, also split
        parts = [p.strip() for p in re.split(r',\s*', part) if p.strip()]
        cand.extend(parts)
    # filter tokens that are too long/short, remove generic words
    skills = []
    seen = set()
    for token in cand:
        # remove sentences
        if len(token) < 2 or len(token) > 80:
            continue
        # remove lines that look like sentences
        if re.search(r'\s{3,}', token):
            continue
        # strip trailing punctuation
        token = token.strip(' .;:')
        tnorm = token.lower()
        if tnorm in ('skills', 'experience', 'education', 'projects'):
            continue
        if tnorm in seen:
            continue
        seen.add(tnorm)
        skills.append(token)
    return skills

def call_claude_parse(prompt_json):
    """
    Improved heuristic parser that looks for common resume sections and extracts:
    - contact info
    - technical_skills (list)
    - soft_skills (list)
    - experience (list of dicts : company/role/start-end/description)
    - projects (list)
    - education (list)
    - certifications (list)
    """
    raw = (prompt_json.get("input_text") or "").strip()
    parsed = {
        "contact": {},
        "technical_skills": [],
        "soft_skills": [],
        "experience": [],
        "projects": [],
        "education": [],
        "certifications": [],
        "summary": ""
    }
    if not raw:
        return parsed

    # normalize spacing
    text = re.sub(r'\r\n', '\n', raw)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # extract contact info as top-level
    contact = _extract_contact_info(text)
    parsed['contact'] = contact

    # split into sections by known headings (best-effort)
    sections = _split_sections_by_headings(text)

    # If explicit skills sections exist, gather them
    skills_candidates = []
    for k, v in sections.items():
        k_low = k.lower()
        if 'skill' in k_low or 'design tools' in k_low or 'research' in k_low:
            skills_candidates.append(v)

    # If no explicit skills heading, try to find a block that starts with "Skills" or "Design Tools"
    if not skills_candidates:
        m = re.search(r'(skills|design tools|technical skills|key skills)[:\s]*\n(.{20,400}?)\n\n', text, flags=re.I | re.S)
        if m:
            skills_candidates.append(m.group(2))

    # collect technical skills from candidates
    tech_skills = []
    for block in skills_candidates:
        tech_skills += _lines_to_skills(block)

    # Additional heuristic: scan for "Design Tools:" / "Design Skills:" exact labels and extract
    for label in ['Design Tools', 'Design Skills', 'Research Skills', 'Skills', 'Tools']:
        rx = re.search(r'(?i)'+re.escape(label)+r'[:\s]*\n?(.{1,400})\n\n', text, flags=re.I | re.S)
        if rx:
            tech_skills += _lines_to_skills(rx.group(1))

    # dedupe preserving order
    seen = set()
    final_tech = []
    for t in tech_skills:
        tn = t.lower().strip()
        if tn and tn not in seen:
            final_tech.append(t.strip())
            seen.add(tn)

    parsed['technical_skills'] = final_tech

    # soft skills: look for commonly named soft skills anywhere OR in small bullets under Volunteer/lead/President
    soft_found = []
    for ss in COMMON_SOFT_SKILLS:
        if re.search(r'\b' + re.escape(ss) + r'\b', text, flags=re.I):
            soft_found.append(ss)
    # also check for role keywords like "lead", "president", "managed"
    if re.search(r'\blead(er|ing)?\b|\bpresident\b|\bmanage(d|r)?\b', text, flags=re.I):
        if 'leadership' not in soft_found:
            soft_found.append('leadership')

    parsed['soft_skills'] = soft_found

    # Experience extraction: find blocks under headings or infer by company-like lines with dates
    exps = []
    # 1) If 'Experience' section exists, split that block by double newlines into items
    exp_block = None
    for h in sections:
        if re.search(r'(?i)experience', h):
            exp_block = sections[h]
            break
    if exp_block:
        # split into items by blank lines (a typical resume uses a blank line between roles)
        pieces = [p.strip() for p in re.split(r'\n{1,3}', exp_block) if p.strip()]
        # heuristics: group lines where a line has a date range or company name
        current = {}
        buffer_lines = []
        for line in pieces:
            # if line contains a date range or hyphen + year, treat as separator
            if re.search(r'(\bJan|\bFeb|\bMar|\bApr|\bMay|\bJun|\bJul|\bAug|\bSep|\bOct|\bNov|\bDec|\d{4})', line) and (len(buffer_lines) > 0):
                # flush previous
                buffer_lines.append(line)
                # create an experience entry from buffer_lines
                block_text = " ".join(buffer_lines)
                # try to parse role/company and dates
                role = None; company = None; start=None; end=None
                # heuristic: if line has ' - ' or '–' with dates
                mdate = re.search(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\.?\s*\d{2,4}|\d{4})\s*[-–to]+\s*(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\.?\s*\d{2,4}|\d{4}|present)', block_text, flags=re.I)
                if mdate:
                    # record approximate dates
                    start = mdate.group(1)
                    end = mdate.group(2)
                # company detection: a line with Title Case + maybe a comma
                lines_in_buf = block_text.split('. ')
                if len(lines_in_buf) > 0:
                    first = buffer_lines[0]
                    # if first line contains commas and mixed caps treat as company/role
                    parts = re.split(r'\s{2,}|\s-\s|,', first)
                    if len(parts) >= 2:
                        company = parts[-1].strip()
                        role = parts[0].strip()
                    else:
                        # fallback: try to split by ' - ' or ' | '
                        if ' - ' in first:
                            role, company = [p.strip() for p in first.split(' - ', 1)]
                        elif '|' in first:
                            role, company = [p.strip() for p in first.split('|', 1)]
                        else:
                            # fallback: set whole block as description
                            role = first.strip()
                exps.append({
                    "role": role or "",
                    "company": company or "",
                    "start": start,
                    "end": end,
                    "description": block_text
                })
                buffer_lines = []
            else:
                buffer_lines.append(line)
        # flush leftover
        if buffer_lines:
            block_text = " ".join(buffer_lines)
            exps.append({
                "role": buffer_lines[0],
                "company": "",
                "start": None,
                "end": None,
                "description": block_text
            })
    else:
        # Fallback: scan whole document for company-like lines followed by bullets; look for uppercase company names then dates
        cand = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            ln = lines[i]
            # if line includes ' - ' and a year, likely role+date
            if re.search(r'\d{4}', ln) and (len(ln.split()) < 12):
                # take next 2 lines as description
                desc = []
                j = i+1
                while j < len(lines) and len(lines[j]) < 200:
                    if re.search(r'\d{4}', lines[j]):
                        break
                    desc.append(lines[j])
                    j += 1
                cand.append({
                    "role": ln,
                    "company": "",
                    "start": None,
                    "end": None,
                    "description": " ".join(desc)
                })
                i = j
            else:
                i += 1
        exps = cand

    parsed['experience'] = exps

    # Projects: look for 'Projects' section or paragraphs containing 'project' word
    projects = []
    for h, block in sections.items():
        if re.search(r'(?i)project', h):
            # split by double newlines into project entries
            for pr in re.split(r'\n{1,3}', block):
                pr = pr.strip()
                if pr:
                    # try to find technologies inside parentheses or after 'using'
                    techs = re.findall(r'\b(?:using|with|built with|tech:)\s+([A-Za-z0-9,\s\-\+\.#&/]+)', pr, flags=re.I)
                    techs_list = []
                    if techs:
                        techs_list = _lines_to_skills(techs[0])
                    projects.append({
                        "name": pr.split('\n')[0][:120],
                        "description": pr,
                        "technologies": techs_list
                    })
    # If no explicit projects, try to infer from experience descriptions that mention 'project'
    if not projects:
        for e in parsed['experience']:
            if 'project' in (e.get('description') or "").lower():
                # use role as project name fallback
                projects.append({
                    "name": e.get('role') or e.get('company') or "Project",
                    "description": e.get('description') or "",
                    "technologies": []
                })
    parsed['projects'] = projects

    # Education: look for Education section & capture lines
    edu = []
    for h, block in sections.items():
        if re.search(r'(?i)education', h):
            # split by lines
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                edu.append(line)
    parsed['education'] = edu

    # Certifications:
    certs = []
    for h, block in sections.items():
        if re.search(r'(?i)certif', h):
            certs += _lines_to_skills(block)
    parsed['certifications'] = certs

    # Short summary: first paragraph or 'summary' section
    if 'summary' in sections:
        parsed['summary'] = sections['summary'][:800]
    else:
        # first non-empty paragraph of the resume (first 300 chars)
        first_para = ""
        for p in text.split('\n\n'):
            if p.strip():
                first_para = p.strip()
                break
        parsed['summary'] = first_para[:800]

    # ensure arrays are lists of strings where appropriate
    parsed['technical_skills'] = parsed.get('technical_skills') or []
    parsed['soft_skills'] = parsed.get('soft_skills') or []
    parsed['projects'] = parsed.get('projects') or []
    parsed['experience'] = parsed.get('experience') or []
    parsed['education'] = parsed.get('education') or []
    parsed['certifications'] = parsed.get('certifications') or []

    return parsed

def parse_and_store_resume(resume_obj):
    """
    Robust wrapper: extract text from file or use raw_text, call parser,
    ensure parsed is JSON-serializable, save to model and return parsed dict.
    """
    text = ""
    try:
        if getattr(resume_obj, "file", None):
            # simple_text_from_file should reset file pointer internally
            text = simple_text_from_file(resume_obj.file) or ""
    except Exception as e:
        # fallback gracefully and keep exception info in raw_text for debugging
        text = resume_obj.raw_text or ""
        # optional: log(e)

    text = text or (resume_obj.raw_text or "")
    # normalize whitespace
    text = re.sub(r'\r\n', '\n', text or "")
    text = text.strip()
    resume_obj.raw_text = text

    # If empty, save minimal parsed and return
    if not text:
        parsed = {
            "contact": {},
            "technical_skills": [],
            "soft_skills": [],
            "experience": [],
            "projects": [],
            "education": [],
            "certifications": [],
            "summary": ""
        }
        resume_obj.parsed = parsed
        resume_obj.save()
        return parsed

    # Build prompt / payload for parser
    prompt = build_resume_prompt(text)
    try:
        parsed = call_claude_parse(prompt) or {}
    except Exception as e:
        # In case parser throws, return a safe structure and store error note
        parsed = {"error": f"parser error: {str(e)}"}
    # Guarantee JSON-serializable basic structure
    safe_parsed = {
        "contact": parsed.get("contact", {}) if isinstance(parsed.get("contact", {}), dict) else {},
        "technical_skills": list(parsed.get("technical_skills", []) or []),
        "soft_skills": list(parsed.get("soft_skills", []) or []),
        "experience": parsed.get("experience", []) if isinstance(parsed.get("experience", []), list) else [],
        "projects": parsed.get("projects", []) if isinstance(parsed.get("projects", []), list) else [],
        "education": parsed.get("education", []) if isinstance(parsed.get("education", []), list) else [],
        "certifications": parsed.get("certifications", []) if isinstance(parsed.get("certifications", []), list) else [],
        "summary": parsed.get("summary", "") or ""
    }

    resume_obj.parsed = safe_parsed
    # optionally save a parser timestamp for auditing
    try:
        resume_obj.parsed_at = getattr(resume_obj, "parsed_at", None) or None
        resume_obj.save()
    except Exception:
        # last-resort: try saving minimally
        resume_obj.save()
    return safe_parsed


def parse_and_store_job(job_obj):
    """
    Robustly parse job object text/file and populate job_obj.parsed and title.
    """
    text = (job_obj.raw_text or "").strip()
    try:
        # Only extract from file if raw_text empty
        if getattr(job_obj, "file", None) and not text:
            text = simple_text_from_file(job_obj.file) or ""
    except Exception:
        text = job_obj.raw_text or ""
    text = re.sub(r'\r\n', '\n', text or "").strip()
    job_obj.raw_text = text

    if not text:
        parsed = {
            "title": job_obj.title or "Job",
            "critical_skills": [],
            "beneficial_skills": [],
            "experience_level": "Unknown",
            "years_required": None,
            "responsibilities": [],
            "domain_keywords": []
        }
        job_obj.parsed = parsed
        job_obj.save()
        return parsed

    parsed = parse_job_description_text(text) or {}
    # normalize title fallback
    parsed_title = parsed.get("title") or (text.splitlines()[0].strip() if text.splitlines() else job_obj.title or "Job")
    job_obj.parsed = parsed
    job_obj.title = parsed_title
    job_obj.save()
    return parsed

def parse_job_description_text(raw_text):
    """
    More robust JD parser:
    - more flexible section extraction (headings + bullet detection)
    - cleaner skill extraction with stopwords filtering
    - improved years detection
    - returns structured dict
    """
    text = (raw_text or "").strip()
    lower = text.lower()

    # helper: find section by heading variants and return block
    def extract_section_variants(names):
        # try many heading variants, return combined paragraphs following them
        for name in names:
            # matches "Name:", "Name\n", "Name -"
            regex = rf'(?ms)^\s*(?:{name})\s*[:\-]?\s*\n(.*?)(?=\n^[A-Z0-9][^\n]{0,120}\s*[:\-]?\s*$|\Z)'
            m = re.search(regex, text, flags=re.I | re.M)
            if m:
                return m.group(1).strip()
        return ""

    requirements = extract_section_variants(["requirements", "required qualifications", "must have", "qualifications", "you must have", "you have"])
    preferred = extract_section_variants(["preferred", "nice to have", "desired qualifications", "nice-to-have"])
    responsibilities = extract_section_variants(["responsibilities", "what you'll do", "what you will do", "role", "you will"])
    # fallback: search for lines starting with '-', '•' or bullet-like lists
    if not requirements:
        # collect lines that appear inside a nearby 'Requirements' block or contain skill-like tokens
        req_lines = []
        for line in text.splitlines():
            if re.match(r'^\s*[-•\*]\s+', line):
                req_lines.append(line.lstrip('-•* \t'))
        requirements = "\n".join(req_lines)

    # cleaner skill extraction
    STOPWORDS = set([
        "and", "the", "with", "or", "to", "for", "of", "in", "on", "experience", "years",
        "year", "knowledge", "skills", "ability", "strong", "working", "understanding", "candidate"
    ])
    def skills_from_text(t):
        if not t:
            return []
        # split by commas, slashes, semicolons, bullets, and newlines
        parts = re.split(r'[,/;•\n\-–\|]+', t)
        out = []
        seen = set()
        for p in parts:
            token = p.strip()
            if not token:
                continue
            # drop long sentences
            if len(token.split()) > 6:
                # try to pull obvious tech tokens inside parentheses
                in_parens = re.findall(r'\(([^)]+)\)', token)
                if in_parens:
                    for ip in in_parens:
                        for sub in re.split(r'[,\|/;]+', ip):
                            sub = sub.strip()
                            if 1 < len(sub) <= 60:
                                key = sub.lower()
                                if key not in STOPWORDS and key not in seen:
                                    out.append(sub)
                                    seen.add(key)
                continue
            # remove leading verbs like "experience in"
            token = re.sub(r'^(experience in|expert in|familiar with|knowledge of)\s+', '', token, flags=re.I).strip()
            # filter stopwords-only tokens and numeric tokens
            tnorm = token.lower().strip()
            if not re.search(r'[a-zA-Z]', tnorm):
                continue
            if tnorm in STOPWORDS:
                continue
            # short word filter
            if len(tnorm) < 2:
                continue
            if tnorm not in seen:
                seen.add(tnorm)
                out.append(token.strip())
        return out

    critical = skills_from_text(requirements)[:80]
    beneficial = skills_from_text(preferred)[:80]

    # responsibilities lines
    resp_list = []
    if responsibilities:
        for line in re.split(r'[\n;•\-]+', responsibilities):
            l = line.strip()
            if l and len(l) < 600:
                resp_list.append(l)
    else:
        # fallback: collect first N long lines that look like duties
        for line in text.splitlines():
            if re.search(r'\b(responsible|lead|design|develop|own|manage)\b', line, flags=re.I):
                resp_list.append(line.strip())
    # years detection (more robust)
    years = None
    m = re.search(r'(?i)(?:minimum|at least)?\s*(\d+)\+?\s*(?:years|yrs)\b', text)
    if m:
        try:
            years = int(m.group(1))
        except Exception:
            years = None
    # experience level heuristics
    level = "Unknown"
    if re.search(r'\bsenior\b|\bsr\.\b', lower):
        level = "Senior"
    elif re.search(r'\b(mid|experienced|lead)\b', lower):
        level = "Mid"
    elif re.search(r'\b(junior|entry|graduate|fresher)\b', lower):
        level = "Junior"

    # domain keywords (expandable)
    domain_keywords = list(set(re.findall(r'\b(fintech|healthcare|e-?commerce|cloud|security|machine learning|ml|data science|embedded|iot|devops|platform)\b', lower, flags=re.I)))

    # title: prefer an explicit "Title:" or the first non-empty line
    title = ""
    # try to find "title" on top lines or "position"
    first_lines = [l.strip() for l in text.splitlines() if l.strip()][:5]
    if first_lines:
        # often JD title is the first line
        title = first_lines[0]
    return {
        "title": title[:200],
        "critical_skills": critical,
        "beneficial_skills": beneficial,
        "experience_level": level,
        "years_required": years,
        "responsibilities": resp_list[:50],
        "domain_keywords": domain_keywords
    }

# ---------- 3. Semantic Match Score Logic (hybrid) ----------
def semantic_match_score(resume_parsed, job_parsed):
    # gather resume signals
    resume_skills = resume_parsed.get("technical_skills", []) + resume_parsed.get("soft_skills", [])
    resume_projects = resume_parsed.get("projects", [])
    resume_exps = resume_parsed.get("experience", [])

    # build resume sentences for embedding-based semantic matching
    resume_sentences = []
    for r in resume_exps:
        if isinstance(r, dict):
            resume_sentences += [r.get('role',''), r.get('description','')] + (r.get('achievements') or [])
        elif isinstance(r, str):
            resume_sentences.append(r)
    for p in resume_projects:
        if isinstance(p, dict):
            resume_sentences += [p.get('name',''), p.get('description','')] + (p.get('technologies') or [])
        elif isinstance(p, str):
            resume_sentences.append(p)
    resume_sentences += [s for s in resume_skills if isinstance(s, str)]

    job_crit = job_parsed.get("critical_skills", []) or []
    job_benef = job_parsed.get("beneficial_skills", []) or []
    job_reqs = job_crit + job_benef

    # embeddings (if available)
    resume_emb = embed_texts(resume_sentences) if EMBEDDING_MODEL is not None else None
    job_emb = embed_texts(job_reqs) if EMBEDDING_MODEL is not None else None
    sim_matrix = cosine_sim(job_emb, resume_emb) if (job_emb is not None and resume_emb is not None) else None

    # weights
    crit_weight = 0.6
    ben_weight = 0.2
    exp_weight = 0.15
    proj_weight = 0.05

    points = 0.0
    max_points = 0.0
    breakdown = {"critical": {}, "beneficial": {}, "experience": {}, "projects": {}}

    # critical
    for idx, s in enumerate(job_crit):
        max_points += crit_weight * 1.5
        matched, rule_score = is_equivalent(s, resume_skills)
        sem_best = 0.0
        if sim_matrix is not None and idx < sim_matrix.shape[0]:
            row = sim_matrix[idx]
            sem_best = float(row.max()) if row.size > 0 else 0.0
        if matched and rule_score == 1.0:
            points += crit_weight * 1.5
            breakdown['critical'][s] = "+1.5 (exact)"
        elif sem_best >= SIM_THRESHOLD:
            points += crit_weight * 1.2
            breakdown['critical'][s] = f"+1.2 (semantic {sem_best:.2f})"
        elif sem_best >= (SIM_THRESHOLD - 0.15):
            points += crit_weight * 0.7
            breakdown['critical'][s] = f"+0.7 (partial semantic {sem_best:.2f})"
        elif matched and rule_score == 0.5:
            points += crit_weight * 0.5
            breakdown['critical'][s] = "+0.5 (transferable)"
        else:
            points += crit_weight * -1.0
            breakdown['critical'][s] = "-1 (missing)"

    # beneficial
    for i, s in enumerate(job_benef):
        job_idx = len(job_crit) + i
        max_points += ben_weight * 0.5
        matched, rule_score = is_equivalent(s, resume_skills)
        sem_best = 0.0
        if sim_matrix is not None and job_idx < sim_matrix.shape[0]:
            row = sim_matrix[job_idx]
            sem_best = float(row.max()) if row.size > 0 else 0.0
        if matched and rule_score >= 1.0:
            points += ben_weight * 0.5
            breakdown['beneficial'][s] = "+0.5 (exact)"
        elif sem_best >= SIM_THRESHOLD:
            points += ben_weight * 0.4
            breakdown['beneficial'][s] = f"+0.4 (semantic {sem_best:.2f})"
        elif matched and rule_score == 0.5:
            points += ben_weight * 0.25
            breakdown['beneficial'][s] = "+0.25 (transferable)"
        else:
            breakdown['beneficial'][s] = "0 (missing)"

    # experience
    years_req = job_parsed.get("years_required")
    exp_pts = 0.0
    if years_req:
        resume_years = 0
        for r in resume_exps:
            if isinstance(r, dict):
                try:
                    start = int(re.search(r'(\d{4})', str(r.get('start') or '')) .group(1))
                    end = int(re.search(r'(\d{4})', str(r.get('end') or '')) .group(1))
                    resume_years += max(0, end - start)
                except Exception:
                    pass
        if resume_years >= years_req:
            exp_pts += exp_weight * 1.0
            breakdown['experience'] = f"Meets years ({resume_years} >= {years_req})"
        elif resume_years >= max(0, years_req - 1):
            exp_pts += exp_weight * 0.5
            breakdown['experience'] = f"Close ({resume_years} ~ {years_req})"
        else:
            exp_pts -= exp_weight * 0.5
            breakdown['experience'] = f"Under ({resume_years} < {years_req})"
    else:
        req_level = (job_parsed.get("experience_level") or "Unknown").lower()
        cnt = len(resume_exps)
        guess = 'senior' if cnt >= 4 else ('mid' if cnt >= 2 else 'junior')
        if guess == req_level:
            exp_pts += exp_weight * 1.0
            breakdown['experience'] = f"Level matches ({guess})"
        elif guess in ('mid','senior') and req_level in ('junior','mid'):
            exp_pts += exp_weight * 0.5
            breakdown['experience'] = f"Close ({guess} vs {req_level})"
        else:
            exp_pts -= exp_weight * 0.5
            breakdown['experience'] = f"Mismatch ({guess} vs {req_level})"
    points += exp_pts
    max_points += exp_weight * 1.0

    # projects
    proj_pts = 0.0
    for p in resume_projects:
        proj_name = p.get('name', 'project') if isinstance(p, dict) else (p or 'project')
        techs = p.get('technologies', []) if isinstance(p, dict) else []
        overlap = 0
        for s in job_crit:
            m, _ = is_equivalent(s, techs)
            if m:
                overlap += 1
            else:
                # semantic check between job requirement and project text if embeddings available
                if EMBEDDING_MODEL is not None:
                    try:
                        job_emb_single = embed_texts(s)
                        proj_text = ""
                        if isinstance(p, dict):
                            proj_text = " ".join([p.get('name',''), p.get('description',''), " ".join(techs)])
                        else:
                            proj_text = p
                        proj_emb_single = embed_texts(proj_text)
                        if job_emb_single is not None and proj_emb_single is not None:
                            sim = cosine_sim(job_emb_single, proj_emb_single)
                            if sim is not None and sim.size > 0 and float(sim.max()) >= SIM_THRESHOLD:
                                overlap += 1
                    except Exception:
                        pass
        if overlap >= 2:
            proj_pts += proj_weight * 1.0
            breakdown['projects'][proj_name] = "Highly relevant"
        elif overlap == 1:
            proj_pts += proj_weight * 0.5
            breakdown['projects'][proj_name] = "Somewhat relevant"
        else:
            breakdown['projects'][proj_name] = "Not relevant"

    points += proj_pts
    max_points += proj_weight * 1.0

    # scale to 0-10
    score = 0.0
    if max_points > 0:
        score = (points / max_points) * 10.0
    score = max(0.0, min(10.0, score))
    rating = "Weak"
    if score >= 8.0:
        rating = "Excellent"
    elif score >= 6.0:
        rating = "Strong"
    elif score >= 4.0:
        rating = "Moderate"

    return {
        "score": round(score, 2),
        "rating": rating,
        "breakdown": breakdown,
        "raw_points": round(points, 4),
        "max_points": round(max_points, 4)
    }

# ---------- 4. Gap Analysis ----------
def gap_analysis(resume_parsed, job_parsed, top_n=5):
    resume_skills = resume_parsed.get("technical_skills", []) + resume_parsed.get("soft_skills", [])
    critical = job_parsed.get("critical_skills", [])[:50]
    beneficial = job_parsed.get("beneficial_skills", [])[:50]
    gaps = []
    for s in critical:
        matched, score = is_equivalent(s, resume_skills)
        if not matched:
            transferable = False
            related_alts = []
            for key, alts in EQUIVALENCES.items():
                if s.lower() == key:
                    for a in alts:
                        if normalize_token(a) in [normalize_token(x) for x in resume_skills]:
                            transferable = True
                            related_alts.append(a)
            typ = 'transferable' if transferable else 'missing'
            suggestion = (f"Your related experience (e.g., {', '.join(related_alts)}) covers {s}. Highlight it."
                          if transferable else
                          f"Consider learning {s}. Short actionable: 2-6 hours tutorial + a small project.")
            gaps.append({"skill": s, "type": typ, "importance": 1.0, "suggestion": suggestion})
    for s in beneficial:
        matched, _ = is_equivalent(s, resume_skills)
        if not matched:
            gaps.append({"skill": s, "type": "learnable", "importance": 0.4, "suggestion": f"Optional but helpful: learn {s} through a short course."})
    gaps_sorted = sorted(gaps, key=lambda x: (-x['importance'], x['type']))
    return gaps_sorted[:top_n]

# ---------- 5. Recommendation generator ----------
def generate_personalized_recommendation(resume_parsed, job_parsed, gaps):
    projects = resume_parsed.get("projects", [])
    # Type A: reframe best project
    best = None
    for p in projects:
        if isinstance(p, dict) and p.get('metrics'):
            best = p; break
    if not best and projects:
        best = projects[0]
    if best:
        name = best.get('name','Project')
        techs = best.get('technologies', [])
        metrics = best.get('metrics') or "improved performance / delivered feature"
        text = f"Type A - Reframe: 'Developed {name} using {', '.join(techs)} — {metrics}.' Emphasize metrics and responsibilities aligned to the job."
        return {"kind": "A", "text": text}
    # Type B: transferable
    for g in gaps:
        if g['type'] == 'transferable':
            return {"kind":"B", "text": f"Type B - Highlight transferable: {g['suggestion']}. Add explicit mention in skills/summary."}
    if gaps:
        s = gaps[0]['skill']
        return {"kind":"C", "text": f"Type C - Add unstated skill: If you have used tools related to {s}, list them explicitly on your resume."}
    return {"kind":"D", "text":"Type D - Learn gap: Complete a short tutorial (2-8 hours) and add a small project demonstrating it."}

# ---------- 6. Simple NL handler ----------
def handle_nl_query(session_parsed_resume, session_parsed_job, last_turns, query_text):
    q = (query_text or "").lower()
    if "skills" in q or q.strip().startswith("skills"):
        if 'data' in q and 'science' in q:
            have_python = any(normalize_token(s) == "python" for s in (session_parsed_resume.get('technical_skills',[]) if session_parsed_resume else []))
            roadmap = ["Python (2-4 weeks)", "Pandas (2 weeks)", "Scikit-learn (3 weeks)"] if not have_python else ["Pandas (2 weeks)", "Scikit-learn (3 weeks)", "Feature engineering (2 weeks)"]
            return {"intent":"skill_learning", "answer": roadmap}
        gaps = gap_analysis(session_parsed_resume or {}, session_parsed_job or {}, top_n=5)
        return {"intent":"skill_learning", "answer": [g['skill'] for g in gaps] or ["Specify domain for tailored roadmap."]}
    if "ready" in q or "apply" in q:
        if not session_parsed_resume or not session_parsed_job:
            return {"intent":"readiness", "answer":"Need both resume and job in session for readiness check."}
        mm = semantic_match_score(session_parsed_resume, session_parsed_job)
        return {"intent":"readiness", "answer": f"Score {mm['score']} ({mm['rating']}).", "detail": mm}
    return {"intent":"unknown", "answer":"I can: (1) give skill roadmap, (2) readiness check, (3) specific skill relevance."}


def _load_skill_embeddings(limit=None):
    """
    Load skills with embeddings into memory.
    Returns (skill_objs_list, embeddings_array (n,d), names_list)
    """
    qs = Skill.objects.exclude(embedding__isnull=True).exclude(embedding__exact=b'')
    if limit:
        qs = qs[:limit]
    skills = list(qs)
    if not skills:
        return [], None, []
    # convert bytes -> numpy float32 arrays
    vecs = []
    names = []
    for s in skills:
        try:
            arr = np.frombuffer(s.embedding, dtype=np.float32)
            vecs.append(arr)
            names.append(s.name)
        except Exception:
            vecs.append(None)
            names.append(s.name)
    # filter out any None
    idx_valid = [i for i,v in enumerate(vecs) if v is not None]
    if not idx_valid:
        return [], None, []
    vecs_arr = np.stack([vecs[i] for i in idx_valid], axis=0)
    skills_valid = [skills[i] for i in idx_valid]
    names_valid = [names[i] for i in idx_valid]
    return skills_valid, vecs_arr, names_valid

def nearest_skill_by_text(text, top_k=3, threshold=0.65):
    """
    Given a text (JD sentence or resume bullet), find nearest canonical skills by precomputed embeddings.
    Returns list of tuples: [(skill_obj, similarity), ...] filtered by threshold.
    """
    if not EMBEDDING_MODEL:
        return []
    emb = EMBEDDING_MODEL.encode([text], convert_to_numpy=True)
    skills, skill_vecs, names = _load_skill_embeddings()
    if skill_vecs is None or skill_vecs.size == 0:
        return []
    # normalize and compute cosine quickly
    # ensure row vectors
    a = emb  # shape (1,d)
    b = skill_vecs  # (n,d)
    # cosine via dot product of normalized vectors
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    sims = (a_norm @ b_norm.T)[0]  # shape (n,)
    # get top_k indices
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        sim = float(sims[i])
        if sim >= threshold:
            results.append((skills[i], sim))
    return results