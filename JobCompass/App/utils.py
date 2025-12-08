import re
import os
import numpy as np
from django.db.models import Q
from django.conf import settings
import logging
import time

log = logging.getLogger(__name__)

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
    r'(?i)research skills', r'(?i)key skills', r'(?i)projects', r'(?i)education', r'(?i)certifications',
    r'(?i)volunteer', r'(?i)awards', r'(?i)summary', r'(?i)profile', r'(?i)contact'
]

# ---------- small helpers ----------
def simple_text_from_file(file_obj):
    try:
        file_obj.seek(0)
    except Exception:
        pass

    text = ""
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
            log.debug("pdfplumber extraction failed", exc_info=True)

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
            log.debug("PyPDF2 extraction failed", exc_info=True)

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


# ---------- KB helpers (Skill table) ----------
def kb_lookup(token):
    if Skill is None:
        return None, None, []
    t = normalize_token(token)
    skill = Skill.objects.filter(Q(normalized=t) | Q(name__iexact=token)).first()
    if skill:
        syns = skill.synonyms or []
        matched_syns = [s for s in syns if normalize_token(s) == t or t in normalize_token(s) or normalize_token(s) in t]
        return skill, skill.name, matched_syns
    try:
        skill = Skill.objects.filter(synonyms__icontains=[token]).first()
        if skill:
            return skill, skill.name, [token]
    except Exception:
        if Skill is not None:
            for s in Skill.objects.all()[:10000]:
                for syn in s.synonyms or []:
                    if normalize_token(syn) == t or t in normalize_token(syn) or normalize_token(syn) in t:
                        return s, s.name, [syn]
    return None, None, []

EQUIVALENCES = {
    "docker": ["containerization", "containers", "containerized"],
    "react": ["react.js", "reactjs"],
    "python": ["py"],
    "flask": ["rest api", "rest"],
    "aws": ["amazon web services", "azure", "cloud"],
    "gcp": ["google cloud"],
    "postgresql": ["postgres", "postgresql"],
}

def _extract_keywords_from_phrase(phrase, max_terms=6):
    if not phrase:
        return []
    txt = re.sub(r'[^A-Za-z0-9\+\#\.\s\-]', ' ', phrase)
    parts = [p.strip().lower() for p in re.split(r'[\s\-]+', txt) if p.strip()]
    parts = [p for p in parts if p not in _SHORT_STOPWORDS and len(p) > 1]
    return parts[:max_terms]

# Slightly more permissive thresholds for multi-word tokens
_JACCARD_HIGH = 0.5
_JACCARD_MED = 0.25

def is_equivalent(skill, candidate_skills):
    s = (skill or "").strip()
    if not s:
        return False, 0.0
    cand = [c for c in (candidate_skills or []) if c]
    cand_norm = [normalize_token(c) for c in cand]

    kb_skill, canonical, matched_syns = kb_lookup(skill)
    if kb_skill:
        variants = {normalize_token(canonical)}
        for syn in (kb_skill.synonyms or []):
            variants.add(normalize_token(syn))
        for c in cand_norm:
            if c in variants:
                return True, 1.0
        for c in cand_norm:
            for v in variants:
                if v in c or c in v:
                    return True, 0.9

    s_norm = normalize_token(s)
    for key, alts in EQUIVALENCES.items():
        if s_norm == normalize_token(key):
            for alt in alts:
                if normalize_token(alt) in cand_norm:
                    return True, 0.9

    for c in cand_norm:
        if s_norm in c or c in s_norm:
            return True, 0.5

    s_tokens = set(_extract_keywords_from_phrase(s))
    best_overlap = 0.0
    for c in cand:
        c_tokens = set(_extract_keywords_from_phrase(c))
        if not s_tokens or not c_tokens:
            continue
        inter = s_tokens & c_tokens
        union = s_tokens | c_tokens
        overlap = (len(inter) / len(union)) if union else 0
        if overlap > best_overlap:
            best_overlap = overlap
    if best_overlap >= _JACCARD_HIGH:
        return True, 0.6
    if best_overlap >= _JACCARD_MED:
        return True, 0.5

    model = get_embedding_model()
    try:
        if model is not None and cand:
            s_emb = embed_texts([s])
            cand_emb = embed_texts(cand)
            if s_emb is not None and cand_emb is not None:
                sims = cosine_sim(s_emb, cand_emb)
                if sims is not None and sims.size > 0:
                    if float(sims.max()) >= SIM_THRESHOLD:
                        return True, 0.85
    except Exception:
        log.debug("embedding fallback in is_equivalent failed", exc_info=True)

    return False, 0.0


# ---------- embedding helpers ----------
from functools import lru_cache
import time

EMBEDDING_MODEL= os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.65))

@lru_cache(maxsize=1)
def get_embedding_model():
    if SentenceTransformer is None:
        log.debug("SentenceTransformer not installed or failed import earlier.")
        return None
    try:
        t0 = time.time()
        model = SentenceTransformer(EMBEDDING_MODEL)
        t1 = time.time()
        log.info(f"Loaded embedding model '{EMBEDDING_MODEL}' in {t1-t0:.2f}s")
        return model
    except Exception:
        log.exception("Failed to load SentenceTransformer model")
        return None

def embed_texts(texts):
    model = get_embedding_model()
    if model is None:
        return None
    if texts is None:
        return None
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    try:
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb
    except Exception:
        log.exception("embed_texts failed")
        return None


def cosine_sim(a, b):
    if a is None or b is None:
        return None
    try:
        model = get_embedding_model()
        if util is not None and model is not None:
            try:
                return util.cos_sim(a, b).cpu().numpy()
            except Exception:
                pass
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
    except Exception:
        log.exception("cosine_sim fallback failed")
        return None


# ---------- parsing helpers ----------
def build_resume_prompt(raw_text):
    # keep a simple wrapper for potential LLM calls.
    return {
        "task": "Parse resume into structured JSON",
        "input_text": raw_text or ""
    }

def _split_sections_by_headings(text):
    import collections
    s_text = text.replace('\r', '\n')
    s_text = re.sub(r'\n{2,}', '\n\n', s_text)
    lines = s_text.splitlines()
    indexes = []
    for i, line in enumerate(lines):
        clean = line.strip()
        for h in SECTION_HEADINGS:
            if re.fullmatch(h + r'[:\s]*', clean, flags=re.I):
                indexes.append((i, clean))
                break
    if not indexes:
        return {"body": s_text}
    sections = collections.OrderedDict()
    for idx, (line_idx, heading_line) in enumerate(indexes):
        start = line_idx + 1
        end = indexes[idx + 1][0] if idx + 1 < len(indexes) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        heading_norm = re.sub(r'[:\s]+$', '', heading_line).strip()
        sections[heading_norm] = content
    return sections

def _extract_contact_info(text):
    contact = {}
    em = re.search(r'([A-Za-z0-9.\-_+]+@[A-Za-z0-9\-_]+\.[A-Za-z0-9.\-_]+)', text)
    if em:
        contact['email'] = em.group(1)
    ph = re.search(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4})', text)
    if ph:
        contact['phone'] = "".join([g for g in ph.groups() if g]).strip()
    web = re.search(r'(https?://[^\s,;]+|www\.[^\s,;]+)', text)
    if web:
        contact['website'] = web.group(1)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        top = lines[0]
        if len(top.split()) <= 4 and re.search(r'[A-Z][a-z]+', top):
            contact['name'] = top
        else:
            if len(lines) > 1 and len(lines[1].split()) <= 4:
                contact['name'] = lines[1]
    return contact


# ---------- stricter skill tokenizer ----------
VERB_KEYWORDS = [
    'design', 'designed', 'designing', 'develop', 'developed', 'build', 'built', 'lead', 'led', 'managed',
    'work', 'worked', 'ship', 'shipped', 'present', 'presented', 'research', 'conducted', 'implemented',
    'evangelized', 'improved', 'created', 'co-led', 'co lead', 'organize', 'organized', 'manage', 'managing'
]

def _looks_like_sentence(s):
    if not s or len(s.strip()) == 0:
        return True
    if len(s.split()) > 6:
        return True
    low = s.lower()
    for v in VERB_KEYWORDS:
        if re.search(r'\b' + re.escape(v) + r'\b', low):
            return True
    if re.search(r'\b(19|20)\d{2}\b', s):
        return True
    if len(re.findall(r'[.,;:]', s)) >= 2:
        return True
    return False

def _lines_to_skills(content, strict=False):
    if not content:
        return []
    content = content.replace('•', '\n').replace('·', '\n').replace('–', '-').replace('\r', '\n')
    parts = []
    for part in re.split(r'[\n;/\u2022]', content):
        part = part.strip()
        if not part:
            continue
        for sub in re.split(r',\s*', part):
            sub = sub.strip()
            if sub:
                parts.append(sub)

    skills = []
    seen = set()
    for token in parts:
        token = token.strip(' .;:')
        if not token:
            continue
        if token.lower() in ('skills', 'experience', 'education', 'projects', 'tech stack', 'techstack', 'tech:'):
            continue
        if _looks_like_sentence(token):
            continue
        if strict and len(token.split()) > 3:
            continue
        if len(token) < 2 or len(token) > 60:
            continue
        tnorm = token.lower()
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

    # --- TECHNICAL SKILLS: only extract from clear places ---
    skills_candidates = []
    for k, v in sections.items():
        k_low = k.lower()
        if any(x in k_low for x in ('skill', 'tool', 'tech', 'key skill', 'technical', 'design tools', 'research skills')):
            skills_candidates.append((v, False))

    # If no explicit skills heading, try a strict extraction from the top-of-document header
    if not skills_candidates:
        header_snippet = text.split('\n\n', 1)[0][:800]
        header_tokens = _lines_to_skills(header_snippet, strict=True)
        if header_tokens:
            skills_candidates.append((", ".join(header_tokens), True))

    tech_skills = []
    for block, strict_flag in skills_candidates:
        tech_skills += _lines_to_skills(block, strict=strict_flag)

    # ALSO: parse explicit "Tech Stack" inline labels (common in project bullets)
    for m in re.finditer(r'(?i)(?:tech(?: ?stack)?\s*[:\-])\s*(.+)', text):
        cap = m.group(1).strip()
        tech_skills += _lines_to_skills(cap, strict=False)

    # dedupe preserving order
    seen = set()
    final_tech = []
    for t in tech_skills:
        tn = t.lower().strip()
        if tn and tn not in seen:
            final_tech.append(t.strip())
            seen.add(tn)

    # FILTER: remove tokens that are obviously roles/projects/dates
    final_tech = [t for t in final_tech if not re.search(r'\b(project|product|intern|manager|lead|experience|develop|worked)\b', t, flags=re.I)]
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
    exp_block = None
    for h in sections:
        if re.search(r'(?i)experience', h):
            exp_block = sections[h]
            break
    if exp_block:
        pieces = [p.strip() for p in re.split(r'\n{1,3}', exp_block) if p.strip()]
        buffer_lines = []
        for line in pieces:
            if re.search(r'(\bJan|\bFeb|\bMar|\bApr|\bMay|\bJun|\bJul|\bAug|\bSep|\bOct|\bNov|\bDec|\d{4})', line) and (len(buffer_lines) > 0):
                buffer_lines.append(line)
                block_text = " ".join(buffer_lines)
                role = None; company = None; start=None; end=None
                mdate = re.search(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\.?\s*\d{2,4}|\d{4})\s*[-–to]+\s*(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\.?\s*\d{2,4}|\d{4}|present)', block_text, flags=re.I)
                if mdate:
                    start = mdate.group(1)
                    end = mdate.group(2)
                lines_in_buf = block_text.split('. ')
                if len(lines_in_buf) > 0:
                    first = buffer_lines[0]
                    parts = re.split(r'\s{2,}|\s-\s|,', first)
                    if len(parts) >= 2:
                        company = parts[-1].strip()
                        role = parts[0].strip()
                    else:
                        if ' - ' in first:
                            role, company = [p.strip() for p in first.split(' - ', 1)]
                        elif '|' in first:
                            role, company = [p.strip() for p in first.split('|', 1)]
                        else:
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
        cand = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        i = 0
        while i < len(lines):
            ln = lines[i]
            if re.search(r'\d{4}', ln) and (len(ln.split()) < 12):
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

    # ---------- Projects: improved extraction (better paragraph expansion) ----------
    projects = []

    # helper: detect tech list inside text (unchanged)
    def _extract_tech_list_from_text(txt):
        if not txt:
            return []
        # common labels + comma/pipe separated tech lists and parentheses
        m = re.search(r'(?i)(?:tech(?: ?stack)?|technologies|tools?)[:\s]*(.+)', txt)
        if m:
            cand = m.group(1)
        else:
            # try parenthetical lists like "(React, Node.js, Docker)"
            par = re.search(r'\(([^)]+\b(?:js|python|docker|react|node|aws|kubernetes|django|flask|sql)[^)]+)\)', txt, flags=re.I)
            cand = par.group(1) if par else ""
        # split by commas, pipes, slashes and ampersand
        items = [p.strip() for p in re.split(r'[,\|/;&]+', cand) if p.strip()]
        # normalize short tokens via existing skill tokenizer
        techs = []
        for it in items:
            techs += _lines_to_skills(it, strict=True)  # strict: keep only short tokens
        return techs

    # helper: decide if a sentence looks like a project description (unchanged)
    def _is_project_sentence(sent):
        if not sent or len(sent.strip()) < 20:
            return False
        low = sent.lower()
        # must contain either project/action verbs or explicit project cue
        project_cues = ['project', 'prototype', 'alpha', 'beta', 'launched', 'ship', 'shipped',
                        'developed', 'built', 'designed', 'led', 'implemented', 'research', 'revamped',
                        'created', 'developed', 'launched', 'delivered', 'pilot']
        if any(re.search(r'\b' + re.escape(c) + r'\b', low) for c in project_cues):
            return True
        # also consider sentences that include 'Tech Stack' or technology lists
        if re.search(r'(?i)tech(?: ?stack)?[:\-\s]|\breact\b|\bnode\b|\bpython\b|\bdjango\b|\bkubernetes\b|\baws\b', low):
            return True
        # presence of metrics often indicates project: %, improved by, users, customers, million
        if re.search(r'\b(%|improv|increase|reduce|users|customers|million|k users|x%|x million|improved by|reduced|resulting in)\b', low):
            return True
        return False

    # helper: expand the matched sentence to a paragraph (prefer double-newline paragraph), else include neighbor sentences
    def _expand_to_paragraph(sentence, full_text):
        if not sentence or not full_text:
            return sentence or ""
        # Normalize spacing
        normalized = full_text.replace('\r\n', '\n')
        # split into paragraphs (double newlines)
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', normalized) if p.strip()]
        # find which paragraph contains the sentence substring
        for p in paragraphs:
            if sentence.strip() in p:
                return p  # full paragraph found
        # fallback: sentence not in any paragraph (maybe no blank lines). Split by sentence punctuation and expand +/-2 sentences
        # Split into crude sentences by punctuation markers
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', normalized) if s.strip()]
        # find index of matching sentence (best-effort using substring)
        idx = None
        for i, s in enumerate(sents):
            if sentence.strip() in s:
                idx = i
                break
        if idx is None:
            # try fuzzy match by starting words
            start = " ".join(sentence.strip().split()[:6]).lower()
            for i, s in enumerate(sents):
                if start and s.lower().startswith(start):
                    idx = i
                    break
        if idx is None:
            # ultimate fallback: return original sentence (no expansion)
            return sentence
        # expand up to 2 sentences before and after (bounded)
        start_i = max(0, idx - 2)
        end_i = min(len(sents), idx + 3)
        expanded = " ".join(sents[start_i:end_i])
        return expanded

    # 1) explicit Projects section (authoritative)
    for h, block in sections.items():
        if re.search(r'(?i)project', h):
            for pr in re.split(r'\n{1,3}', block):
                pr = pr.strip()
                if not pr:
                    continue
                techs = _extract_tech_list_from_text(pr)
                projects.append({
                    "name": pr.split('\n')[0][:200],
                    "description": pr,
                    "technologies": techs,
                    "source": h
                })

    # 2) infer projects from Experience entries (paragraph-level expansion)
    for exp in parsed.get('experience', []):
        if not isinstance(exp, dict):
            continue
        desc_full = (exp.get('description') or "")  # DO NOT truncate here; preserve full description
        if not desc_full or not desc_full.strip():
            continue
        # split into candidate sentences / clauses (allow multi-line)
        cand_sents = [s.strip() for s in re.split(r'[.\n;•\-]{1,}\s*', desc_full) if s.strip()]
        for sent in cand_sents:
            if _is_project_sentence(sent):
                # expand to full paragraph (preferred) or surrounding sentences
                expanded = _expand_to_paragraph(sent, desc_full)
                techs = _extract_tech_list_from_text(expanded)
                # form a better project name: prefer a short heading if present, else first sentence/clause (max 200 chars)
                # attempt to find a short heading-like clause in expanded (before ":" or "-" or "|" )
                heading_match = re.split(r'[:\-\|]\s*', expanded.strip(), 1)[0]
                # pick heading if it's reasonably short, else take beginning of expanded
                if 6 <= len(heading_match.split()) <= 12:
                    name = heading_match.strip()[:200]
                else:
                    # first 200 chars of expanded paragraph
                    name = expanded.strip().split('\n', 1)[0][:200]
                # check duplicates by comparing expanded paragraph content
                dup = False
                for p in projects:
                    # check if expanded paragraph or its start is already present
                    if expanded.strip()[:120].lower() in p.get('description','').lower() or p.get('description','').strip()[:120].lower() in expanded.strip().lower():
                        dup = True
                        p_techs = set(p.get('technologies', []))
                        for t in techs:
                            if t and t not in p_techs:
                                p['technologies'].append(t)
                        break
                if not dup:
                    projects.append({
                        "name": name,
                        "description": expanded,
                        "technologies": techs,
                        "source": exp.get('company') or exp.get('role') or 'experience'
                    })

    # 3) fallback: scan whole document for 'Tech Stack:' occurrences (project-like)
    for m in re.finditer(r'(?i)(?:tech(?: ?stack)?|technologies|tools)[:\s]*(.+)', text):
        cap = m.group(1).strip()
        techs = _lines_to_skills(cap, strict=True)
        if techs:
            name = "Tech stack"
            exists = any(set(techs) <= set(p.get('technologies', [])) for p in projects)
            if not exists:
                projects.append({
                    "name": name,
                    "description": cap,
                    "technologies": techs,
                    "source": "tech-stack"
                })

    # final normalization: dedupe but prefer longer descriptions when duplicates exist
    seen_map = {}  # key -> index in final_projects
    final_projects = []
    for p in projects:
        key = (p.get('name','').strip().lower() + '|' + (p.get('description','')[:120] or '').strip().lower())
        if key in seen_map:
            # prefer the one with longer description: replace if current is longer
            idx = seen_map[key]
            existing = final_projects[idx]
            if len(p.get('description','') or "") > len(existing.get('description','') or ""):
                final_projects[idx] = p
        else:
            seen_map[key] = len(final_projects)
            # cleanup technology tokens: dedupe + short-only
            techs = []
            tseen = set()
            for t in (p.get('technologies') or []):
                tn = t.strip()
                if not tn or len(tn.split()) > 6:
                    continue
                tl = tn.lower()
                if tl in tseen:
                    continue
                tseen.add(tl)
                techs.append(tn)
            p['technologies'] = techs
            final_projects.append(p)

    parsed['projects'] = final_projects
    # -------------------------------------------------------------------------

    # Education: look for Education section & capture lines
    edu = []
    for h, block in sections.items():
        if re.search(r'(?i)education', h):
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
            certs += _lines_to_skills(block, strict=False)
    parsed['certifications'] = certs

    # Short summary: first paragraph or 'summary' section
    if 'summary' in sections:
        parsed['summary'] = sections['summary'][:800]
    else:
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
            text = simple_text_from_file(resume_obj.file) or ""
    except Exception as e:
        text = resume_obj.raw_text or ""
        log.debug("file extraction error", exc_info=True)

    text = text or (resume_obj.raw_text or "")
    text = re.sub(r'\r\n', '\n', text or "")
    text = text.strip()
    resume_obj.raw_text = text

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

    prompt = build_resume_prompt(text)
    try:
        parsed = call_claude_parse(prompt) or {}
    except Exception as e:
        parsed = {"error": f"parser error: {str(e)}"}
        log.exception("call_claude_parse failed")

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
    try:
        resume_obj.parsed_at = getattr(resume_obj, "parsed_at", None) or None
        resume_obj.save()
    except Exception:
        resume_obj.save()
    return safe_parsed


def parse_and_store_job(job_obj):
    """
    Robustly parse job object text/file and populate job_obj.parsed and title.
    """
    text = (job_obj.raw_text or "").strip()
    try:
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
    parsed_title = parsed.get("title") or (text.splitlines()[0].strip() if text.splitlines() else job_obj.title or "Job")
    job_obj.parsed = parsed
    job_obj.title = parsed_title
    job_obj.save()
    return parsed

def parse_job_description_text(raw_text):
    """
    Robust JD parser:
    - finds headings by scanning lines,
    - slices sections reliably between headings,
    - extracts requirement/preferred candidate strings (bullets + parenthetical lists),
    - uses the advanced auto_normalize_skills() to produce clean skill tokens,
    - extracts responsibilities from explicit responsibilities sections or short action bullets,
    - returns structured dict.
    """
    text = (raw_text or "").strip()
    lower = text.lower()

    # --- 1) Lines + heading detection (lightweight) ---
    lines = [ln.rstrip() for ln in text.splitlines()]
    heading_keywords = [
        "requirements", "required qualifications", "must have", "qualifications",
        "minimum qualifications", "preferred qualifications", "preferred", "nice to have",
        "responsibilities", "responsibility", "what you'll do", "what you will do", "role",
        "you will", "about the role", "job responsibilities", "minimum requirements", "minimum"
    ]

    heading_positions = []
    for i, ln in enumerate(lines):
        clean = ln.strip()
        if not clean:
            continue
        cmp = clean.rstrip(':').strip().lower()
        # exact known headings or startswith
        if any(cmp == hk or cmp.startswith(hk) for hk in heading_keywords):
            heading_positions.append((i, clean.rstrip(':')))
            continue
        # heuristic: a short, title-like line
        if 1 <= len(clean.split()) <= 6 and re.match(r'^[A-Za-z0-9 \-&()\/]+:?\s*$', clean):
            heading_positions.append((i, clean.rstrip(':')))

    # build sections
    sections = {}
    if heading_positions:
        heading_positions = sorted(heading_positions, key=lambda x: x[0])
        for idx, (pos, heading) in enumerate(heading_positions):
            start = pos + 1
            end = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
            content = "\n".join(lines[start:end]).strip()
            sections[heading.strip()] = content
    else:
        sections = {"body": text}

    # helper: match section by name
    def get_section_by_names(names):
        for k, v in sections.items():
            kl = k.lower()
            for name in names:
                if name in kl or kl.startswith(name):
                    return v
        return ""

    # Extract common sections
    requirements = get_section_by_names(["requirement", "minimum qualification", "must have", "qualification"])
    preferred = get_section_by_names(["preferred", "nice to have", "desired qualification"])
    responsibilities = get_section_by_names(["responsibil", "what you'll do", "role", "you will", "about the role"])

    # --- 2) Build skill candidate strings (prefer bullets + parenthesis lists) ---
    def _bullets_from_block(block_text):
        if not block_text:
            return []
        cands = []
        for ln in re.split(r'[\r\n]+', block_text):
            if not ln or not ln.strip():
                continue
            s = ln.strip()
            s = re.sub(r'^\s*[-•\*\u2022]\s*', '', s)
            cands.append(s.strip())
        return cands

    requirement_candidates = []
    if requirements:
        requirement_candidates.append(requirements)
        requirement_candidates += _bullets_from_block(requirements)
        # capture parenthetical lists inside requirements like "(e.g. Java, Python)"
        for par in re.findall(r'\(([^)]+)\)', requirements):
            requirement_candidates.append(par)
    else:
        # fallback: scan bullets across entire document for likely requirement lines
        all_bullets = []
        for ln in lines:
            if re.match(r'^\s*[-•\*\u2022]\s+', ln):
                all_bullets.append(re.sub(r'^\s*[-•\*\u2022]\s+', '', ln).strip())
        for b in all_bullets:
            if re.search(r'\b(experience|years|experience in|knowledge of|experience with|degree|must|minimum|required|qualification|work authorization)\b', b, flags=re.I):
                requirement_candidates.append(b)
        # also add parenthetical lists from whole text
        for par in re.findall(r'\(([^)]+)\)', text):
            requirement_candidates.append(par)

    preferred_candidates = []
    if preferred:
        preferred_candidates.append(preferred)
        preferred_candidates += _bullets_from_block(preferred)
        for par in re.findall(r'\(([^)]+)\)', preferred):
            preferred_candidates.append(par)

    # fallback: extract short strict tokens from top-of-document header if nothing found
    if not requirement_candidates:
        header_snippet = text.split('\n\n', 1)[0][:800]
        strict_header = _lines_to_skills(header_snippet, strict=True)
        if strict_header:
            requirement_candidates += strict_header

    # --- 3) Normalize skills using automatic normalizer (auto_normalize_skills) ---
    # NOTE: auto_normalize_skills is defined below in the file; calling it here is fine at runtime.
    try:
        critical = auto_normalize_skills(requirement_candidates)[:80]
    except Exception:
        # fallback to older behavior if something goes wrong
        critical = []
        for r in requirement_candidates:
            critical += _lines_to_skills(r, strict=True)
        # dedupe
        seen_c = set()
        tmp = []
        for c in critical:
            k = c.lower().strip()
            if k not in seen_c:
                seen_c.add(k)
                tmp.append(c)
        critical = tmp[:80]

    try:
        beneficial = auto_normalize_skills(preferred_candidates)[:80]
    except Exception:
        beneficial = []
        for p in preferred_candidates:
            beneficial += _lines_to_skills(p, strict=True)
        seen_b = set()
        tmpb = []
        for b in beneficial:
            k = b.lower().strip()
            if k not in seen_b:
                seen_b.add(k)
                tmpb.append(b)
        beneficial = tmpb[:80]

    # --- 4) Responsibilities extraction: prefer explicit section; otherwise short action bullets ---
    resp_list = []
    if responsibilities:
        for part in re.split(r'[\n;•\-–]+', responsibilities):
            p = part.strip()
            if not p:
                continue
            # drop long paragraphs (likely an intro/summary)
            if len(p) > 300:
                continue
            # keep short-ish lines and those starting with action verbs
            if re.search(r'^\s*(develop|design|implement|build|manage|lead|produce|work independently|work on|master|deliver|maintain|test|optimi[sz]e|create|demonstrate)\b', p, flags=re.I) or len(p.split()) <= 40:
                resp_list.append(p)
    else:
        # fallback: scan bullets across the whole doc but filter out qualification/authorization lines
        for ln in lines:
            if re.match(r'^\s*[-•\*\u2022]\s+', ln):
                candidate = re.sub(r'^\s*[-•\*\u2022]\s+', '', ln).strip()
            else:
                candidate = ln.strip()
            if not candidate:
                continue
            if len(candidate.split()) > 40:
                continue
            if re.search(r'^\s*(develop|design|implement|build|manage|lead|produce|work|master|deliver|maintain|test|create|demonstrate)\b', candidate, flags=re.I) or len(candidate.split()) <= 20:
                # drop qualification lines (degrees, must obtain, work authorization)
                if re.search(r'\b(degree|bachelor|master|phd|must obtain|work authorization|obtain work authorization|must have|required|minimum qualifications?)\b', candidate, flags=re.I):
                    continue
                resp_list.append(candidate)

    # dedupe responsibilities preserving order
    seen_r = set()
    final_resp = []
    for r in resp_list:
        k = r.lower().strip()
        if k in seen_r:
            continue
        seen_r.add(k)
        final_resp.append(r)
    resp_list = final_resp[:50]

    # --- 5) Years and experience level heuristics ---
    years = None
    m = re.search(r'(?i)(?:minimum|at least)?\s*(\d+)\+?\s*(?:years|yrs)\b', text)
    if m:
        try:
            years = int(m.group(1))
        except Exception:
            years = None

    level = "Unknown"
    if re.search(r'\bsenior\b|\bsr\.\b', lower):
        level = "Senior"
    elif re.search(r'\b(mid|experienced|lead)\b', lower):
        level = "Mid"
    elif re.search(r'\b(junior|entry|graduate|fresher|university grad)\b', lower):
        level = "Junior"

    # domain keywords heuristic
    domain_keywords = list(set(re.findall(
        r'\b(fintech|healthcare|e-?commerce|cloud|security|machine learning|ml|data science|embedded|iot|devops|platform)\b',
        lower, flags=re.I)))

    # Title extraction: prefer first non-empty line, but prefer header-like lines
    first_lines = [l.strip() for l in text.splitlines() if l.strip()][:5]
    title = ""
    if first_lines:
        if re.search(r' - | — | – ', first_lines[0]) or len(first_lines[0].split()) <= 12:
            title = first_lines[0]
        else:
            title = first_lines[0]

    return {
        "title": title[:200],
        "critical_skills": critical,
        "beneficial_skills": beneficial,
        "experience_level": level,
        "years_required": years,
        "responsibilities": resp_list,
        "domain_keywords": domain_keywords
    }


# ---------- 3. Semantic Match Score Logic (hybrid) ----------
def semantic_match_score(resume_parsed, job_parsed):
    resume_skills = (resume_parsed.get("technical_skills", []) or []) + (resume_parsed.get("soft_skills", []) or [])
    resume_projects = resume_parsed.get("projects", []) or []
    resume_exps = resume_parsed.get("experience", []) or []

    job_crit = job_parsed.get("critical_skills", []) or []
    job_benef = job_parsed.get("beneficial_skills", []) or []

    CRIT_WEIGHT = 0.50
    BEN_WEIGHT = 0.20
    EXP_WEIGHT = 0.15
    PROJ_WEIGHT = 0.15

    _emb_cache = {}
    def get_emb_once(text):
        if not text:
            return None
        key = text if isinstance(text, str) else str(text)
        if key in _emb_cache:
            return _emb_cache[key]
        emb = embed_texts(text)
        _emb_cache[key] = emb
        return emb

    resume_sentences = []
    for p in resume_projects:
        if isinstance(p, dict):
            if p.get('name'):
                resume_sentences.append(p.get('name'))
            if p.get('description'):
                resume_sentences.append(p.get('description'))
            for t in (p.get('technologies') or []):
                if t:
                    resume_sentences.append(t)
    for r in resume_exps:
        if isinstance(r, dict):
            if r.get('role'):
                resume_sentences.append(r.get('role'))
            if r.get('description'):
                resume_sentences.append(r.get('description'))
    resume_sentences += [s for s in resume_skills if isinstance(s, str) and s.strip()]

    job_reqs = [s for s in job_crit if s] + [s for s in job_benef if s]

    model = get_embedding_model()
    resume_emb = None
    job_emb = None
    sim_matrix = None
    try:
        if model is not None:
            if resume_sentences:
                # embed list directly (cached)
                resume_emb = get_emb_once(tuple(resume_sentences)) or embed_texts(resume_sentences)
            if job_reqs:
                job_emb = get_emb_once(tuple(job_reqs)) or embed_texts(job_reqs)
            if job_emb is not None and resume_emb is not None:
                if getattr(job_emb, "size", 0) and getattr(resume_emb, "size", 0):
                    sim_matrix = cosine_sim(job_emb, resume_emb)
            log.debug("semantic_match_score: model_loaded=%s, job_emb_shape=%s, resume_emb_shape=%s",
                      (model is not None),
                      None if job_emb is None else getattr(job_emb, 'shape', None),
                      None if resume_emb is None else getattr(resume_emb, 'shape', None))
    except Exception:
        log.exception("Embedding precompute failed in semantic_match_score; continuing with rule-based checks.")

    num_crit = max(1, len(job_crit))
    num_ben = max(1, len(job_benef))

    breakdown = {"critical": {}, "beneficial": {}, "experience": {}, "projects": {}}
    weighted_sum = 0.0

    def compute_skill_credit(skill_text, idx_in_job_reqs=None, cand_resume_list=None):
        cand_resume_list = cand_resume_list or resume_skills
        matched, rule_score = is_equivalent(skill_text, cand_resume_list)
        sem_best = 0.0
        if sim_matrix is not None and idx_in_job_reqs is not None:
            try:
                if 0 <= idx_in_job_reqs < sim_matrix.shape[0]:
                    row = sim_matrix[idx_in_job_reqs]
                    if row is not None and getattr(row, "size", 0):
                        sem_best = float(row.max())
            except Exception:
                log.debug("sim_matrix indexing error for idx %s", idx_in_job_reqs, exc_info=True)
        if matched and rule_score >= 1.0:
            return 1.0, f"1.00 (exact rule)"
        if sem_best >= SIM_THRESHOLD:
            return 0.9, f"{0.9:.2f} (semantic {sem_best:.2f})"
        if sem_best >= (SIM_THRESHOLD - 0.15):
            return 0.6, f"{0.6:.2f} (partial semantic {sem_best:.2f})"
        if matched and rule_score >= 0.45:
            return 0.5, f"{0.5:.2f} (transferable)"
        return 0.0, "0 (missing)"

    for i, s in enumerate(job_crit):
        credit, reason = compute_skill_credit(s, idx_in_job_reqs=i, cand_resume_list=resume_skills)
        contrib = (credit * CRIT_WEIGHT) / num_crit
        weighted_sum += contrib
        breakdown['critical'][s] = reason

    base_idx = len(job_crit)
    for j, s in enumerate(job_benef):
        idx = base_idx + j
        credit, reason = compute_skill_credit(s, idx_in_job_reqs=idx, cand_resume_list=resume_skills)
        contrib = (credit * BEN_WEIGHT) / num_ben
        weighted_sum += contrib
        breakdown['beneficial'][s] = reason

    # Experience computation: structured years then summary then fallback
    resume_years = 0
    for r in resume_exps:
        if isinstance(r, dict):
            s = r.get('start') or ""
            e = r.get('end') or ""
            ys = re.findall(r'(\d{4})', str(s))
            ye = re.findall(r'(\d{4})', str(e))
            try:
                if ys and ye:
                    start_y = int(ys[0])
                    end_y = int(ye[-1])
                    resume_years += max(0, end_y - start_y)
                elif ys and (not ye):
                    resume_years += max(0, (int(time.strftime("%Y")) - int(ys[0])))
            except Exception:
                pass

    summary = (resume_parsed.get("summary") or "")
    # capture "3+ years" or "3 years"
    m = re.search(r'(\d+)\+?\s+years', summary, flags=re.I)
    if m:
        try:
            years_val = int(m.group(1))
            resume_years = max(resume_years, years_val)
        except Exception:
            pass
    else:
        # also try small word numbers like "three years"
        WORD_NUMS = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        for word, val in WORD_NUMS.items():
            if re.search(r'\b' + re.escape(word) + r'(?:\+)?\s+years?\b', summary, flags=re.I):
                resume_years = max(resume_years, val)
                break

    years_req = job_parsed.get("years_required")
    exp_credit = 0.0
    if years_req:
        try:
            years_req_i = int(years_req)
        except Exception:
            years_req_i = None
        if years_req_i:
            if resume_years >= years_req_i:
                exp_credit = 1.0
                breakdown['experience'] = f"Meets years ({resume_years} >= {years_req_i})"
            elif resume_years >= max(0, years_req_i - 1):
                exp_credit = 0.6
                breakdown['experience'] = f"Close ({resume_years} ~ {years_req_i})"
            else:
                exp_credit = 0.0
                breakdown['experience'] = f"Under ({resume_years} < {years_req_i})"
        else:
            # fallback as below
            pass
    if not years_req:
        cnt = len([e for e in resume_exps if isinstance(e, dict)])
        if resume_years >= 5 or cnt >= 4:
            exp_credit = 1.0
            breakdown['experience'] = "Senior inferred"
        elif resume_years >= 2 or cnt >= 2:
            exp_credit = 0.6
            breakdown['experience'] = "Mid inferred"
        else:
            exp_credit = 0.2
            breakdown['experience'] = f"Early career inferred ({resume_years} yrs)"

    weighted_sum += exp_credit * EXP_WEIGHT

    proj_credit_total = 0.0
    if resume_projects:
        for p in resume_projects:
            if not isinstance(p, dict):
                continue
            techs = p.get('technologies') or []
            # normalize tech tokens
            techs_norm = []
            for t in techs:
                if t and isinstance(t, str):
                    n = _normalize_token(t)
                    if n:
                        techs_norm.append(n)
            # If after normalization techs empty, try inference from description
            if not techs_norm:
                desc = (p.get('description') or "") + " " + (p.get('name') or "")
                inferred = []
                # quick heuristic: find occurrences of common tech words
                common_techs = r'\b(python|flask|fastapi|docker|kubernetes|aws|azure|sql|postgresql|postgres|mysql|node\.js|react|django|java|c\+\+|c#|go|golang)\b'
                for m in re.finditer(common_techs, desc, flags=re.I):
                    tok = _normalize_token(m.group(1))
                    if tok and tok not in inferred:
                        inferred.append(tok)
                techs_norm = inferred
            overlap_cnt = 0
            for s in job_crit:
                m, _ = is_equivalent(s, techs_norm)
                if m:
                    overlap_cnt += 1
            proj_text = " ".join(filter(None, [p.get('name',''), p.get('description',''), " ".join(techs_norm)]))
            proj_emb = get_emb_once(proj_text) if model is not None else None
            sem_score = 0.0
            if proj_emb is not None and job_emb is not None:
                try:
                    sims = cosine_sim(job_emb, proj_emb)
                    if sims is not None and sims.size:
                        sem_score = float(sims.max())
                except Exception:
                    sem_score = 0.0
            token_score = min(1.0, overlap_cnt / max(1.0, len(job_crit)))
            proj_relevance = max(token_score * 0.7, sem_score * 0.9)
            proj_credit_total += proj_relevance
            breakdown['projects'][p.get('name','project')] = f"rel:{proj_relevance:.2f} (tokens:{overlap_cnt},sem:{sem_score:.2f})"
        proj_credit = (proj_credit_total / len(resume_projects))
    else:
        proj_credit = 0.0
    weighted_sum += proj_credit * PROJ_WEIGHT

    final_score = max(0.0, min(1.0, weighted_sum)) * 10.0
    rating = "Weak"
    if final_score >= 8.0:
        rating = "Excellent"
    elif final_score >= 6.0:
        rating = "Strong"
    elif final_score >= 4.0:
        rating = "Moderate"

    return {
        "score": round(final_score, 2),
        "rating": rating,
        "breakdown": breakdown,
        "raw_points": round(weighted_sum, 4),
        "max_points": 1.0
    }


# ---------- 4. Gap Analysis ----------
def gap_analysis(resume_parsed, job_parsed, top_n=8):
    resume_skills = (resume_parsed.get("technical_skills", []) or []) + (resume_parsed.get("soft_skills", []) or [])
    raw_critical = job_parsed.get("critical_skills", []) or []
    beneficial = job_parsed.get("beneficial_skills", []) or []

    crit_candidates = []
    for item in raw_critical:
        if not item:
            continue
        if _looks_like_sentence(item) or len(item.split()) > 5:
            kws = _extract_keywords_from_phrase(item, max_terms=6)
            # join kws into small phrases where possible before normalizing
            if kws:
                # attempt grouping into short multi-word tokens (two-word phrases)
                twos = []
                for i in range(len(kws)-1):
                    twos.append(kws[i] + " " + kws[i+1])
                candidates = kws + twos
            else:
                candidates = [item]
            norm = auto_normalize_skills(candidates)
            if norm:
                crit_candidates.extend(norm)
            else:
                short = _lines_to_skills(item, strict=True)
                if short:
                    crit_candidates.extend(auto_normalize_skills(short))
                else:
                    # fallback: keep first 3 words if they look ok
                    first = " ".join(item.strip().split()[:3])
                    norm2 = auto_normalize_skills([first])
                    if norm2:
                        crit_candidates.extend(norm2)
        else:
            norm = auto_normalize_skills([item])
            if norm:
                crit_candidates.extend(norm)
            else:
                crit_candidates.append(item.strip())

    seen = set()
    critical = []
    for c in crit_candidates:
        if not c:
            continue
        k = c.lower().strip()
        if k not in seen:
            seen.add(k)
            critical.append(c)

    gaps = []
    for s in critical:
        matched, score = is_equivalent(s, resume_skills)
        if matched:
            continue
        transferable = False
        related = []
        for key, alts in EQUIVALENCES.items():
            if normalize_token(s) == normalize_token(key):
                for a in alts:
                    if normalize_token(a) in [normalize_token(x) for x in resume_skills]:
                        transferable = True
                        related.append(a)
        s_tokens = set(_extract_keywords_from_phrase(s))
        for rs in resume_skills:
            r_tokens = set(_extract_keywords_from_phrase(rs))
            if s_tokens and r_tokens:
                ov = len(s_tokens & r_tokens) / float(len(s_tokens | r_tokens))
                if ov >= 0.20 and ov < _JACCARD_HIGH:
                    transferable = True
                    related.append(rs)
        if transferable:
            suggestion = f"Highlight related experience: {', '.join(list(dict.fromkeys(related))[:3])}. Move those bullets to top of experience or summary."
            gaps.append({"skill": s, "type": "transferable", "importance": 1.0, "suggestion": suggestion})
        else:
            suggestion = f"This role requires '{s}' but your resume doesn't mention it. Consider adding it or a short micro-project demonstrating it (e.g., small Dockerized API)."
            gaps.append({"skill": s, "type": "missing", "importance": 1.0, "suggestion": suggestion})

    ben_norm = []
    for b in beneficial:
        norm = auto_normalize_skills([b]) or [b]
        ben_norm.extend(norm)
    for s in ben_norm:
        matched, _ = is_equivalent(s, resume_skills)
        if not matched:
            suggestion = f"Optional but helpful: learn '{s}' and add a demonstrable artifact or short project."
            gaps.append({"skill": s, "type": "learnable", "importance": 0.4, "suggestion": suggestion})

    gaps_sorted = sorted(gaps, key=lambda x: (-x['importance'], x['type']))
    return gaps_sorted[:top_n]


# ---------- 5. Recommendation generator ----------
def generate_personalized_recommendation(resume_parsed, job_parsed, gaps, use_llm=False, llm_fn=None):
    """
    Produce a recommendation tailored to resume_parsed and job_parsed using gaps.
    Returns a dict with keys:
      - kind: one of "A","B","C","D"
      - text: human-facing recommendation
      - example_bullet: (optional) a copy-paste bullet the candidate can put on resume
      - micro_project: (optional) dict describing steps when relevant
      - project: (optional) best_project dict if Type A
    """
    projects = resume_parsed.get("projects", []) or []
    # normalize any declared techs
    for p in projects:
        techs = p.get('technologies') or []
        if techs:
            norm = []
            for t in techs:
                if not t:
                    continue
                n = _normalize_token(t)
                if n:
                    norm.append(n)
            p['technologies'] = norm

    # quick helper to infer tech tokens from text
    def infer_techs_from_text(txt):
        if not txt:
            return []
        common_techs_re = r'\b(python|flask|fastapi|django|docker|kubernetes|aws|azure|gcp|sql|sqlite|postgresql|postgres|mysql|node\.js|react|java|c\+\+|c#|go|golang)\b'
        found = []
        for m in re.finditer(common_techs_re, txt, flags=re.I):
            tok = _normalize_token(m.group(1))
            if tok and tok not in found:
                found.append(tok)
        return found

    # Embedding utilities (safe cached)
    model = get_embedding_model()
    _emb_cache = {}
    def get_emb_once_local(text):
        if not text:
            return None
        key = text if isinstance(text, str) else str(text)
        if key in _emb_cache:
            return _emb_cache[key]
        emb = embed_texts(text)
        _emb_cache[key] = emb
        return emb

    # Score projects for relevance (token overlap + semantic)
    job_crit = job_parsed.get("critical_skills", []) or []
    best_project = None
    best_score = 0.0
    for p in projects:
        name = p.get('name','').strip()
        techs = p.get('technologies') or []
        # infer if empty
        if not techs:
            techs = infer_techs_from_text((p.get('description') or "") + " " + name)
        # token overlap metric
        token_overlap = 0.0
        if techs and job_crit:
            overlaps = 0
            for jc in job_crit:
                matched, _ = is_equivalent(jc, techs)
                if matched:
                    overlaps += 1
            token_overlap = overlaps / max(1.0, len(job_crit))
        # semantic similarity
        sem_score = 0.0
        try:
            if model is not None:
                proj_text = " ".join(filter(None, [name, p.get('description',''), " ".join(techs)]))
                p_emb = get_emb_once_local(proj_text)
                if p_emb is not None and job_crit:
                    j_emb = get_emb_once_local(tuple(job_crit)) or embed_texts(job_crit)
                    if j_emb is not None:
                        sims = cosine_sim(j_emb, p_emb)
                        if sims is not None and getattr(sims, "size", 0):
                            sem_score = float(sims.max())
        except Exception:
            sem_score = 0.0
        relevance = max(token_overlap * 0.75, sem_score * 0.9)
        # small boost to projects with explicit techs
        if techs:
            relevance += 0.05
        if relevance > best_score:
            best_score = relevance
            best_project = p

    # If we found a strongly relevant project, produce Type A (reframe)
    if best_project and best_score >= 0.45:
        name = best_project.get('name','Project')
        techs = best_project.get('technologies') or []
        if not techs:
            techs = infer_techs_from_text((best_project.get('description') or "") + " " + name)
        desc = (best_project.get('description') or "").strip()
        short_desc = (desc[:180] + '...') if len(desc) > 180 else desc
        example_bullet = f"Developed {name} using {', '.join(techs[:4])} — {short_desc}" if techs else f"Developed {name} — {short_desc}"
        text_template = (
            f"Type A — Reframe an existing project: Move the project '{name}' to the top of your experience. "
            f"Open the project bullet with technologies ({', '.join(techs[:4])}) and a one-line impact/delivery statement. "
            f"Example bullet: \"{example_bullet}\""
        )
        return {
            "kind": "A",
            "text": text_template,
            "example_bullet": example_bullet,
            "project": best_project
        }

    # No strong project: analyze gaps for transferable vs missing
    missing = [g for g in (gaps or []) if g.get('type') == 'missing']
    transferable = [g for g in (gaps or []) if g.get('type') == 'transferable']
    learnable = [g for g in (gaps or []) if g.get('type') == 'learnable']

    # If transferable gaps exist, give a Type B recommendation (how to reframe)
    if transferable:
        top = transferable[0]
        related = top.get('suggestion') or ""
        text = (
            f"Type B — Highlight transferable experience: {related}. "
            f"Suggested action: Move the related bullets to the top of your experience section and add a one-line summary in your resume summary mentioning "
            f"collaboration with engineering or instrumentation (e.g., \"Instrumented product prototype and collaborated with engineers on deployment\")."
        )
        example_bullet = None
        # attempt to construct one using project names or descriptions
        if projects:
            sample = projects[0]
            pname = sample.get('name') or "Project"
            ptech = sample.get('technologies') or infer_techs_from_text((sample.get('description') or "") + " " + pname)
            if ptech:
                example_bullet = f"Collaborated on {pname} (used {', '.join(ptech[:4])}) to instrument and validate product flows with engineering partners."
            else:
                example_bullet = f"Collaborated on {pname} to instrument prototype and validate product flows with engineering partners."
        return {"kind": "B", "text": text, "example_bullet": example_bullet}

    # If critical missing skills exist, build a concrete Type C micro-project plan
    if missing:
        # prioritize top 3 missing criticals
        top_missing = [m['skill'] for m in missing[:3]]
        # map missing tokens to a concrete micro-project stack
        stack = []
        if any(re.search(r'\bdocker\b', s, flags=re.I) for s in top_missing):
            stack.append('Docker')
        if any(re.search(r'\bpython\b|\bflask\b|\bfastapi\b', s, flags=re.I) for s in top_missing):
            stack.append('Python (Flask or FastAPI)')
        if any(re.search(r'\bsql\b|\bpostgres\b|\bmysql\b|\bsqlite\b', s, flags=re.I) for s in top_missing):
            stack.append('SQLite / SQL')
        if any(re.search(r'\baws\b|\bazure\b|\bgcp\b|\bcloud\b', s, flags=re.I) for s in top_missing):
            stack.append('Cloud (Deploy to Render/Heroku/AWS Free Tier)')
        if any(re.search(r'\bapi\b|\bbackend\b|\bserver\b|\bmicroservice\b', s, flags=re.I) for s in top_missing):
            if 'Python (Flask or FastAPI)' not in stack:
                stack.append('Python (Flask or FastAPI)')

        # default fallback stack if stack empty but missing exists
        if not stack:
            stack = ['Python (Flask)', 'SQLite', 'Docker', 'Deploy (Render/Heroku)']

        # construct micro-project steps tightly mapped to missing skills
        micro_steps = [
            f"1) Build a minimal REST API implementing one resource (e.g., inventory endpoint) using {stack[0]}.",
            f"2) Persist sample data using {stack[1] if len(stack)>1 else 'SQLite'} (CRUD endpoints).",
            f"3) Add basic unit tests for the API endpoints (pytest/unittest).",
            f"4) Create a Dockerfile to containerize the service and verify it runs locally in a container.",
            f"5) Deploy the container to a free host (Render / Heroku / AWS free tier) and add a short README describing the architecture & deployment steps.",
            f"6) Add a single-line measurable demo in README (e.g., sample curl requests) and include the GitHub link in your resume."
        ]
        micro_project = {
            "title": f"Micro-project targeted to: {', '.join(top_missing)}",
            "stack": stack,
            "steps": micro_steps
        }

        # example resume bullet tailored to the micro-project & job title
        job_title = job_parsed.get('title') or "the role"
        example_bullet = (
            f"Built a {job_title}-relevant prototype (Flask API + SQLite + Docker); containerized and deployed the prototype to Render, "
            "documented deployment steps and integration notes to demonstrate end-to-end functionality."
        )

        text = (
            f"Type C — Missing critical skills: {', '.join(top_missing)}. "
            f"Suggested micro-project (4–8 hours) using: {', '.join(stack)}. "
            "Follow the steps below to produce an artifact you can link on your resume and highlight in your experience."
        )

        return {
            "kind": "C",
            "text": text,
            "micro_project": micro_project,
            "example_bullet": example_bullet
        }

    # If we reach here, nothing clear missing/transferable — produce a contextual Type D
    job_title = job_parsed.get('title') or job_parsed.get('role') or "this role"
    ben = job_parsed.get('beneficial_skills') or []
    ben_short = ", ".join(auto_normalize_skills(ben[:4])) if ben else "relevant beneficial skills"
    text = (
        f"Type D — No clear project to reframe and no critical skills detected as missing. "
        f"Suggested: pick a short micro-project that shows backend basics tailored to {job_title}: focus on {ben_short} and one backend stack (Python + Docker + SQL)."
    )
    micro_project = {
        "title": f"Starter micro-project for {job_title}",
        "stack": ["Python (Flask)", "SQLite", "Docker", "Deploy (Render)"],
        "steps": [
            "1) Create a simple Flask API that returns JSON for one endpoint.",
            "2) Store and retrieve data using SQLite.",
            "3) Add a Dockerfile and run locally as a container.",
            "4) Deploy to Render/Heroku and add README with instructions."
        ]
    }
    example_bullet = "Built a small Flask + Docker prototype demonstrating an end-to-end API; documented deployment steps and linked GitHub."
    return {"kind": "D", "text": text, "micro_project": micro_project, "example_bullet": example_bullet}

import json

# Allowed tech tokens / short allowlist (extend as needed)
_ALLOWED_TECH_TOKENS = {
    'python','flask','fastapi','django','docker','kubernetes','aws','azure','gcp',
    'sql','sqlite','postgresql','postgres','mysql','node.js','react','java','c++',
    'c#','go','golang','powerbi','tableau','excel','salesforce','hubspot'
}

def _llm_validate_and_parse(json_text, resume_parsed, job_parsed):
    """
    Validate LLM JSON output conservatively:
    - Must be valid JSON with required keys
    - example_bullet must not invent numbers or metrics
    - any mentioned tech must be in resume or job or in allowlist
    Returns (valid:bool, parsed_obj/dict or error_msg)
    """
    try:
        obj = json.loads(json_text)
    except Exception as e:
        return False, f"invalid_json: {e}"

    # expected schema keys
    allowed_keys = {'kind','text','example_bullet','micro_project','project'}
    if not isinstance(obj, dict):
        return False, "json_not_object"
    if not obj.get('kind') or obj['kind'] not in {'A','B','C','D'}:
        return False, "missing_or_invalid_kind"

    # validate text presence
    if 'text' not in obj or not isinstance(obj['text'], str):
        return False, "missing_text"

    # no invented numbers: example_bullet must not contain % or numeric claims unless present in inputs
    eb = obj.get('example_bullet') or ""
    if any(re.search(r'\b\d{1,3}%|\b\d{1,4}\b', token) for token in re.findall(r'\S+', eb)):
        # if numbers appear in resume_parsed or job_parsed, allow those specific numbers
        inputs_combined = json.dumps(resume_parsed) + json.dumps(job_parsed)
        numbers_in_input = set(re.findall(r'\b\d{1,4}\b|\b\d{1,3}%\b', inputs_combined))
        numbers_in_eb = set(re.findall(r'\b\d{1,4}\b|\b\d{1,3}%\b', eb))
        if not numbers_in_eb.issubset(numbers_in_input):
            return False, "invented_numbers_in_example_bullet"

    # tech token validation: ensure technologies mentioned appear in resume/job or allowed set
    mentioned_techs = set([t.lower() for t in re.findall(r'\b([A-Za-z0-9\+\#\.]{2,30})\b', eb)])
    # build a small corpus of allowed tokens from resume/job
    corpus = set()
    for k in ('technical_skills','projects','experience','summary'):
        v = resume_parsed.get(k) or ""
        corpus.update(re.findall(r'\b[a-zA-Z0-9\+\#\-\.]{2,30}\b', str(v)))
    for k in ('critical_skills','beneficial_skills','title'):
        v = job_parsed.get(k) or ""
        corpus.update(re.findall(r'\b[a-zA-Z0-9\+\#\-\.]{2,30}\b', str(v)))
    corpus = set([c.lower() for c in corpus])
    # If a token looks like a tech (quick heuristic) and is not in corpus or allowlist, fail
    suspicious = [t for t in mentioned_techs if t.isalpha() and len(t) <= 30 and t not in corpus and t not in _ALLOWED_TECH_TOKENS]
    if suspicious:
        # allow common words that are not techs; filter by lowercase common english words is optional
        # to be safe, reject when suspicious tech-like tokens found
        return False, f"unvalidated_tokens:{','.join(suspicious)}"

    # micro_project check (if present) must be a dict and contain 'steps' list
    if 'micro_project' in obj:
        mp = obj['micro_project']
        if not isinstance(mp, dict):
            return False, "micro_project_not_object"
        if 'steps' not in mp or not isinstance(mp['steps'], list):
            return False, "micro_project_steps_missing"

    return True, obj

def _call_llm_for_recommendation(resume_parsed, job_parsed, gaps, llm_fn):
    """
    Build a constrained prompt and call llm_fn(prompt, temperature=0.0).
    Expect the model to return strict JSON. Return parsed dict or None.
    """
    # Build short facts block
    def shortify(obj, keys, max_chars=1000):
        out = []
        for k in keys:
            v = obj.get(k)
            if not v:
                continue
            s = str(v)
            if len(s) > max_chars:
                s = s[:max_chars] + "..."
            out.append(f"{k}: {s}")
        return "\n".join(out)

    resume_facts = shortify(resume_parsed, ["summary","technical_skills","projects"])
    job_facts = shortify(job_parsed, ["title","critical_skills","beneficial_skills","years_required"])
    gaps_small = json.dumps(gaps[:6], ensure_ascii=False)

    prompt = f"""
You are a concise career assistant. INPUTS (only use the facts below; do NOT invent facts or numbers):
----- RESUME FACTS -----
{resume_facts}

----- JOB FACTS -----
{job_facts}

----- TOP GAPS (from earlier analysis) -----
{gaps_small}

TASK:
Produce a single JSON object (no surrounding text) with these keys:
- kind: one of "A","B","C","D"
- text: a short recommendation sentence (<= 300 chars)
- example_bullet: a single resume bullet candidate (<= 200 chars). Do NOT invent numeric claims (percentages, counts); only reuse numbers present in the inputs if any.
- micro_project: optional object with "title" (string) and "steps" (list of short step strings). Steps should be concrete, practical, and match the missing skills.

Important constraints:
1) Use only facts in the RESUME or JOB blocks. If you need to suggest a tech, prefer technologies already in resume/projects or in job criticals; otherwise suggest conservative, common stacks (e.g., Python/Flask, SQLite, Docker).
2) Do NOT invent metrics (no "%", no "reduced X by Y%" unless the number appears in the inputs).
3) Output must be valid JSON only. No extra explanation.
4) Keep temperature/creativity low; be pragmatic and concise.

Return only the JSON object.
"""

    # call LLM with low temperature; your llm_fn should accept temperature param
    try:
        # adapt to the llm_fn signature available in your environment
        try:
            raw = llm_fn(prompt, temperature=0.0)
        except TypeError:
            # fallback if llm_fn doesn't accept temperature
            raw = llm_fn(prompt)
    except Exception as e:
        log.exception("LLM call failed: %s", e)
        return None

    # try parsing and validating
    valid, parsed_or_err = _llm_validate_and_parse(raw, resume_parsed, job_parsed)
    if not valid:
        log.warning("LLM recommendation validation failed: %s. Raw response: %s", parsed_or_err, raw[:200])
        return None
    return parsed_or_err


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


def _load_skill_embeddings(limit=None, expected_dim=None):
    """
    Load skills with embeddings and validate shapes/dtype.
    Returns (skill_objs_list, embeddings_array (n,d), names_list)
    """
    if Skill is None:
        return [], None, []
    qs = Skill.objects.exclude(embedding__isnull=True).exclude(embedding__exact=b'')
    if limit:
        qs = qs[:limit]
    skills = list(qs)
    if not skills:
        return [], None, []
    vecs = []
    names = []
    bad = []
    for s in skills:
        try:
            arr = np.frombuffer(s.embedding, dtype=np.float32)
            # If expected_dim provided, check
            if expected_dim and arr.size % expected_dim != 0:
                bad.append((s, arr.shape))
                continue
            if arr.ndim == 1:
                vecs.append(arr)
                names.append(s.name)
            else:
                bad.append((s, arr.shape))
        except Exception:
            bad.append((s, None))
    if not vecs:
        return [], None, []
    vecs_arr = np.stack(vecs, axis=0).astype(np.float32)
    return skills[:len(vecs)], vecs_arr, names

def nearest_skill_by_text(text, top_k=3, threshold=0.65):
    """
    Use the lazy model + precomputed skill embeddings.
    """
    model = get_embedding_model()
    if model is None:
        return []
    # embed the text
    emb = embed_texts([text])
    if emb is None:
        return []
    # load skill embeddings and validate dimension
    skills, skill_vecs, names = _load_skill_embeddings(expected_dim=model.get_sentence_embedding_dimension())
    if skill_vecs is None or skill_vecs.size == 0:
        return []
    # normalize and compute sims
    a = emb.astype(np.float32)
    b = skill_vecs.astype(np.float32)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sims = (a_norm @ b_norm.T)[0]
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        sim = float(sims[i])
        if sim >= threshold:
            results.append((skills[i], sim))
    return results

_GENERIC_SKILL_TOKENS = {
    'relevant', 'relevance', 'qualification', 'qualifications', 'practical', 'computer', 'science',
    'experience', 'skills', 'summary', 'responsibilities', 'responsibility', 'requirements', 'preferred'
}


# common short acronyms we want uppercase

# blacklist phrases or tokens that are not skills
_SKILL_STOP_PATTERNS = [
    r"currently has", r"in the process of obtaining", r"bachelor", r"master", r"degree",
    r"must obtain", r"work authorization", r"must have", r"minimum qualifications?",
    r"preferred qualifications?", r"responsibilities?", r"summary", r"experience coding",
    r"experience from", r"work experience", r"publications", r"coding competitions",
    r"relevant technical field", r"experience with the", r"knowledge of the"
]
_SKILL_STOP_RE = re.compile("|".join(_SKILL_STOP_PATTERNS), flags=re.I)
_ACRONYM_SET = {"aws", "gcp", "sql", "html", "css", "api", "ml", "nlp", "ci", "cd", "cli", "os", "ios", "android"}
_SHORT_STOPWORDS = {"and", "or", "the", "with", "in", "on", "for", "of", "to", "experience", "knowledge"}
_SPLIT_DELIM_RE = re.compile(r'[,/;•\n\|]+')
_PARENS_RE = re.compile(r'\((.*?)\)')
_PRESERVE_PATTERN = re.compile(r'^(c\+\+|c#|\.net|node\.js|nodejs|node|js|javascript|typescript|ts|java|python|go(lang)?|golang|rust|kotlin|swift)$', flags=re.I)


def _is_acronym(word):
    w = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
    return w in _ACRONYM_SET or (word.isupper() and len(word) <= 5)

def _clean_whitespace_and_punct(s):
    # collapse strange punctuation, remove extra spaces
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    # remove leading/trailing punctuation
    s = re.sub(r'^[\s\-\–\—\•\*]+', '', s)
    s = re.sub(r'[\s\-\–\—\•\*]+$', '', s)
    return s

def _normalize_token(tok):
    if not tok:
        return None
    t = tok.strip()
    # drop if contains stop phrases
    if _SKILL_STOP_RE.search(t):
        return None
    # remove stray leading bullets / punctuation
    t = _clean_whitespace_and_punct(t)
    # remove explanatory trailing clause
    t = re.sub(r'(?i)\b(e\.g\.|eg:|for example:|such as:|such as)\b.*$', '', t).strip()
    if t.startswith('(') and t.endswith(')'):
        t = t[1:-1].strip()
    t = re.sub(r'^(experience in|experience with|familiar with|knowledge of|proficient in|proficiency in)\s+', '', t, flags=re.I).strip()
    if re.search(r'\b(degree|bachelor|masters|phd|university|graduat(e|ing))\b', t, flags=re.I):
        return None
    if len(t.split()) > 8:
        return None
    if t.lower() in _GENERIC_SKILL_TOKENS:
        return None
    m = re.match(r'^(c\+\+|c#|\.net)$', t, flags=re.I)
    if m:
        normalized = m.group(1)
        if normalized.lower() == 'c++':
            return 'C++'
        if normalized.lower() == 'c#':
            return 'C#'
        return '.NET'
    lower = t.lower()
    if _is_acronym(t):
        return t.upper()
    if re.search(r'[._]', t):
        if re.match(r'node(\.|js|js$)', lower):
            return 'Node.js'
        if re.match(r'^\.?net$', lower) or lower == '.net':
            return '.NET'
        t2 = re.sub(r'^[\W_]+|[\W_]+$', '', t)
        return t2 if len(t2) <= 60 else None
    if re.match(r'^[a-z0-9\+\#\-]+$', t, flags=re.I):
        if re.match(_PRESERVE_PATTERN, t):
            low = t.lower()
            if low in ('js', 'javascript'):
                return 'JavaScript'
            if low in ('py', 'python'):
                return 'Python'
            if low in ('cpp', 'c++'):
                return 'C++'
            if low in ('c#', 'csharp'):
                return 'C#'
            if low in ('golang', 'go'):
                return 'Go'
            if low in ('ts', 'typescript'):
                return 'TypeScript'
            if low in ('node', 'nodejs'):
                return 'Node.js'
            if low in ('sql',):
                return 'SQL'
            return t.capitalize()
        if len(t) <= 4 and t.islower() and t in _ACRONYM_SET:
            return t.upper()
        return t.capitalize()
    parts = re.split(r'[\s\-_]+', t)
    normalized_parts = []
    for p in parts:
        if _is_acronym(p):
            normalized_parts.append(p.upper())
        else:
            if re.search(r'[A-Z][a-z]', p):
                normalized_parts.append(p)
            else:
                normalized_parts.append(p.capitalize())
    normalized = " ".join(normalized_parts).strip()
    if len(normalized) == 0 or len(normalized) > 80:
        return None
    normalized = re.sub(r'[,:;]+$', '', normalized)
    if normalized.lower() in _GENERIC_SKILL_TOKENS:
        return None
    return normalized


def auto_normalize_skills(raw_skill_candidates):
    if not raw_skill_candidates:
        return []
    seen = set()
    out = []
    for item in raw_skill_candidates:
        if not item:
            continue
        pieces = re.split(_SPLIT_DELIM_RE, item)
        parens = _PARENS_RE.findall(item)
        if parens:
            for p in parens:
                pieces.extend(re.split(_SPLIT_DELIM_RE, p))
        expanded = []
        for p in pieces:
            if p and re.search(r'\band\b', p, flags=re.I) and len(p.split(',')) == 1 and len(p.split()) <= 6:
                for sub in re.split(r'\band\b', p, flags=re.I):
                    expanded.append(sub)
            else:
                expanded.append(p)
        for piece in expanded:
            p = piece.strip()
            if not p:
                continue
            p = re.sub(r'^\s*[:\-\u2022\*]+\s*', '', p)
            norm = _normalize_token(p)
            if not norm:
                continue
            nk = norm.lower()
            if nk not in seen:
                seen.add(nk)
                out.append(norm)
    return out