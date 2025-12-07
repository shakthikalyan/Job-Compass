# App/utils.py
import re
import os
import numpy as np
from django.db.models import Q
from django.conf import settings
import logging

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


# ---------- basic helpers ----------
def simple_text_from_file(file_obj):
    """
    Improved text extraction:
    - use pdfplumber when available (better on modern/resume PDFs),
    - fallback to PyPDF2,
    - fallback to raw bytes decode.
    """
    try:
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
            log.debug("pdfplumber extraction failed", exc_info=True)

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
            log.debug("PyPDF2 extraction failed", exc_info=True)

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


# ---------- KB helpers (Skill table) ----------
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
        if Skill is not None:
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


# ---------- stricter skill tokenizer ----------
VERB_KEYWORDS = [
    'design', 'designed', 'designing', 'develop', 'developed', 'build', 'built', 'lead', 'led', 'managed',
    'work', 'worked', 'ship', 'shipped', 'present', 'presented', 'research', 'conducted', 'implemented',
    'evangelized', 'improved', 'created', 'co-led', 'co lead', 'organize', 'organized', 'manage', 'managing'
]


def _looks_like_sentence(s):
    """Return True if s looks like a sentence / description rather than a short skill token."""
    if not s or len(s.strip()) == 0:
        return True
    # Many words -> likely a sentence/description
    if len(s.split()) > 6:
        return True
    # Contains verbs (heuristic)
    low = s.lower()
    for v in VERB_KEYWORDS:
        if re.search(r'\b' + re.escape(v) + r'\b', low):
            return True
    # Contains dates or year tokens -> not a skill
    if re.search(r'\b(19|20)\d{2}\b', s):
        return True
    # multiple punctuation (likely a sentence)
    if len(re.findall(r'[.,;:]', s)) >= 2:
        return True
    return False


def _lines_to_skills(content, strict=False):
    """
    Produce short, canonical skill tokens from a content block.
    strict=True: only accept very short tokens (1-3 words), used for fallback header extraction.
    """
    if not content:
        return []
    # normalize bullet characters and newlines
    content = content.replace('•', '\n').replace('·', '\n').replace('–', '-').replace('\r', '\n')
    parts = []
    # split on newlines and some separators, then on commas
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
        # discard obvious headings
        if token.lower() in ('skills', 'experience', 'education', 'projects', 'tech stack', 'techstack', 'tech:'):
            continue
        # drop sentence-like tokens
        if _looks_like_sentence(token):
            continue
        # strict mode: only accept at most 3 words
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

    resume_emb = embed_texts(resume_sentences) if EMBEDDING_MODEL is not None else None
    job_emb = embed_texts(job_reqs) if EMBEDDING_MODEL is not None else None
    sim_matrix = cosine_sim(job_emb, resume_emb) if (job_emb is not None and resume_emb is not None) else None

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
    for s in skills:
        try:
            arr = np.frombuffer(s.embedding, dtype=np.float32)
            vecs.append(arr)
            names.append(s.name)
        except Exception:
            vecs.append(None)
            names.append(s.name)
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
    a = emb
    b = skill_vecs
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    sims = (a_norm @ b_norm.T)[0]
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        sim = float(sims[i])
        if sim >= threshold:
            results.append((skills[i], sim))
    return results


# common short acronyms we want uppercase
_ACRONYM_SET = {"aws", "gcp", "sql", "html", "css", "api", "ml", "nlp", "ci", "cd", "cli", "os", "ios", "android"}

# blacklist phrases or tokens that are not skills
_SKILL_STOP_PATTERNS = [
    r"currently has", r"in the process of obtaining", r"bachelor", r"master", r"degree",
    r"must obtain", r"work authorization", r"must have", r"minimum qualifications?",
    r"preferred qualifications?", r"responsibilities?", r"summary", r"experience coding",
    r"experience from", r"work experience", r"publications", r"coding competitions",
    r"relevant technical field", r"experience with the", r"knowledge of the"
]
_SKILL_STOP_RE = re.compile("|".join(_SKILL_STOP_PATTERNS), flags=re.I)

# words that are not helpful as skills by themselves
_SHORT_STOPWORDS = {"and", "or", "the", "with", "in", "on", "for", "of", "to", "experience", "knowledge"}

# delimiters to split multi-skill tokens
_SPLIT_DELIM_RE = re.compile(r'[,/;•\n\|]+')

# detect languages / tech tokens that should preserve special chars (C++, C#, .NET)
_PRESERVE_PATTERN = re.compile(r'^(c\+\+|c#|\.net|node\.js|nodejs|node|js|javascript|typescript|ts|java|python|go(lang)?|golang|rust|kotlin|swift)$', flags=re.I)

# detect things that look like "X (Y, Z)" or "X: Y, Z" while splitting
_PARENS_RE = re.compile(r'\((.*?)\)')

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
    """
    Normalize a single token heuristically.
    Returns normalized token or None if token should be discarded.
    """
    if not tok:
        return None
    t = tok.strip()

    # drop if contains stop phrases
    if _SKILL_STOP_RE.search(t):
        return None

    # often tokens are full sentences; drop long natural sentences
    if len(t) > 120 and t.count(' ') > 20:
        return None

    # remove stray leading bullets / punctuation
    t = _clean_whitespace_and_punct(t)

    # remove trailing explanatory clauses like "e.g." or "such as"
    t = re.sub(r'(?i)\b(e\.g\.|eg:|for example:|such as:|such as)\b.*$', '', t).strip()

    # split parenthesis content into separate tokens if present
    # but only return current token; the splitter will handle inner ones
    # remove stray parentheses wrappers
    if t.startswith('(') and t.endswith(')'):
        t = t[1:-1].strip()

    # remove leading phrases "experience in", "familiar with"
    t = re.sub(r'^(experience in|experience with|familiar with|knowledge of|proficient in|proficiency in)\s+', '', t, flags=re.I).strip()

    # filter out degree-like phrases
    if re.search(r'\b(degree|bachelor|masters|phd|university|graduat(e|ing))\b', t, flags=re.I):
        return None

    # if it still looks like a sentence (many words and verbs), drop it
    if len(t.split()) > 8:
        # allow medium-length hyphenated tech names, but drop long sentences
        return None

    # preserve exact known forms first (C++, C#, .NET)
    m = re.match(r'^(c\+\+|c#|\.net)$', t, flags=re.I)
    if m:
        normalized = m.group(1)
        # standard casing
        if normalized.lower() == 'c++':
            return 'C++'
        if normalized.lower() == 'c#':
            return 'C#'
        return '.NET'

    # normalize common js/node variants heuristically
    lower = t.lower()

    # if token is an acronym or short uppercase-like -> uppercase it
    if _is_acronym(t):
        return t.upper()

    # token contains dots or slashes (e.g., node.js) -> preserve punctuation and reasonable case
    if re.search(r'[._]', t):
        # normalize node.js / NodeJS variants
        if re.match(r'node(\.|js|js$)', lower):
            return 'Node.js'
        if re.match(r'^\.?net$', lower) or lower == '.net':
            return '.NET'
        # keep token but strip surrounding punctuation
        t2 = re.sub(r'^[\W_]+|[\W_]+$', '', t)
        return t2 if len(t2) <= 60 else None

    # short tokens that are likely languages/tech; capitalize appropriately
    if re.match(r'^[a-z0-9\+\#\-]+$', t, flags=re.I):
        # keep known language shapes like "python", "javascript", "java", "golang", "rust"
        if re.match(_PRESERVE_PATTERN, t):
            # special-case a few to conventional display names
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
            # default to title-case if not acronym-like
            return t.capitalize()

        # if looks like an acronym (aws, gcp, sql), uppercase
        if len(t) <= 4 and t.islower() and t in _ACRONYM_SET:
            return t.upper()

        # else capitalize (python -> Python)
        return t.capitalize()

    # multi-word token: title-case it but keep known all-caps pieces
    parts = re.split(r'[\s\-_]+', t)
    normalized_parts = []
    for p in parts:
        if _is_acronym(p):
            normalized_parts.append(p.upper())
        else:
            # preserve camelCase or PascalCase words if present
            if re.search(r'[A-Z][a-z]', p):
                normalized_parts.append(p)
            else:
                normalized_parts.append(p.capitalize())
    normalized = " ".join(normalized_parts).strip()

    # final sanity checks
    if len(normalized) == 0 or len(normalized) > 80:
        return None

    # remove trailing stray punctuation
    normalized = re.sub(r'[,:;]+$', '', normalized)
    return normalized

def auto_normalize_skills(raw_skill_candidates):
    """
    raw_skill_candidates: list of strings (each string may contain multiple skills or be a sentence)
    returns: list of normalized skill strings, deduplicated (preserving order)
    """
    if not raw_skill_candidates:
        return []

    seen = set()
    out = []

    for item in raw_skill_candidates:
        if not item:
            continue
        # first split on major delimiters
        pieces = re.split(_SPLIT_DELIM_RE, item)
        # also split out parentheses contents (e.g., "languages (Java, Python)") and include them
        parens = _PARENS_RE.findall(item)
        if parens:
            for p in parens:
                pieces.extend(re.split(_SPLIT_DELIM_RE, p))

        # further split on "and" if short lists like "Java and Python"
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
            # remove leading filler like "experience:" or bullet markers
            p = re.sub(r'^\s*[:\-\u2022\*]+\s*', '', p)
            # normalize token
            norm = _normalize_token(p)
            if not norm:
                continue
            nk = norm.lower()
            if nk not in seen:
                seen.add(nk)
                out.append(norm)

    return out