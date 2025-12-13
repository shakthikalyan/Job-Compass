import re
import os
import json
import time
import logging
import numpy as np
from functools import lru_cache
from django.db.models import Q
from typing import Dict,Any,List

from App import ai_client



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

_spacy_nlp = None
def _load_spacy_model(prefer_trf=True):
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
    except Exception:
        log.debug("spaCy not installed", exc_info=True)
        return None
    # Try transformer model first, then small model
    tried = []
    if prefer_trf:
        tried.append("en_core_web_trf")
    tried.append("en_core_web_sm")
    for mdl in tried:
        try:
            _spacy_nlp = spacy.load(mdl)
            log.info("Loaded spaCy model: %s", mdl)
            return _spacy_nlp
        except Exception:
            log.debug("spaCy failed to load model %s", mdl, exc_info=True)
            continue
    try:
        _spacy_nlp = spacy.blank("en")
        return _spacy_nlp
    except Exception:
        _spacy_nlp = None
        return None


# ---------------- Constants (preserved) ----------------
EMBEDDING_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.65))

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

LLM_SYSTEM_NO_THINK = (
    "You are an expert career coach and resume evaluator.\n"
    "Do NOT reveal chain-of-thought, reasoning steps, or analysis.\n"
    "Do NOT include <think> tags.\n"
    "Respond ONLY with valid JSON that matches the requested schema.\n"
    "If the output is not valid JSON, it is invalid."
)


SECTION_HEADINGS = [
    r'(?i)experience', r'(?i)work experience', r'(?i)professional experience',
    r'(?i)skills', r'(?i)technical skills', r'(?i)projects', r'(?i)education',
]

_SHORT_STOPWORDS = {"and", "or", "the", "with", "in", "on", "for", "of", "to", "experience"}
_PARENS_RE = re.compile(r'\((.*?)\)')
_SPLIT_DELIM_RE = re.compile(r'[,/;•\n\|]+')
_PRESERVE_PATTERN = re.compile(r'^(c\+\+|c#|\.net|node(\.|js)?|js|javascript|typescript|ts|java|python|go(lang)?|golang|rust|kotlin|swift)$', flags=re.I)
_ACRONYM_SET = {"aws", "gcp", "sql", "html", "css", "api", "ml", "nlp", "ci", "cd", "cli", "os", "ios", "android"}

_SKILL_STOP_PATTERNS = [
    r"currently has", r"in the process of obtaining", r"bachelor", r"master", r"degree",
    r"must obtain", r"work authorization", r"must have", r"minimum qualifications?",
    r"preferred qualifications?", r"responsibilities?", r"summary", r"experience coding",
]
_SKILL_STOP_RE = re.compile("|".join(_SKILL_STOP_PATTERNS), flags=re.I)

_GENERIC_SKILL_TOKENS = {
    'relevant', 'relevance', 'qualification', 'qualifications', 'practical', 'computer', 'science',
    'experience', 'skills', 'summary', 'responsibilities',
}

# ---------- file to text ---------- (unchanged)
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
                return _preprocess_extracted_text(text)
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
                return _preprocess_extracted_text(text)
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
    return _preprocess_extracted_text(text)

def _preprocess_extracted_text(text):
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F\n]', ' ', text)
    return text.strip()

# ---------- normalization helpers ---------- (unchanged)
def _clean_whitespace_and_punct(s):
    s = s or ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^[\s\-\–\—\•\*]+', '', s)
    s = re.sub(r'[\s\-\–\—\•\*]+$', '', s)
    return s

def _is_acronym(word):
    w = re.sub(r'[^a-zA-Z0-9]', '', (word or "")).lower()
    return w in _ACRONYM_SET or (word and word.isupper() and len(word) <= 5)

def _normalize_token(tok):
    if not tok:
        return None
    t = tok.strip()
    if _SKILL_STOP_RE.search(t):
        return None
    t = _clean_whitespace_and_punct(t)
    t = re.sub(r'(?i)\b(e\.g\.|eg:|for example:|such as:)\b.*$', '', t).strip()
    if t.startswith('(') and t.endswith(')'):
        t = t[1:-1].strip()
    t = re.sub(r'(?i)^(experience in|experience with|familiar with|knowledge of|proficient in)\s+', '', t).strip()
    if re.search(r'\b(degree|bachelor|masters|phd|university)\b', t, flags=re.I):
        return None
    if len(t.split()) > 8:
        return None
    lower = t.lower()
    if _is_acronym(t):
        return t.upper()
    if re.search(r'[._]', t):
        if re.match(r'node(\.|js|js$)', lower):
            return 'Node.js'
        t2 = re.sub(r'^[\W_]+|[\W_]+$', '', t)
        return t2 if len(t2) <= 60 else None
    if re.match(r'^[a-z0-9\+\#\-\.\_]+$', t, flags=re.I):
        low = t.lower()
        if re.match(_PRESERVE_PATTERN, t):
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
        return t.capitalize()
    parts = re.split(r'[\s\-_]+', t)
    out_parts = []
    for p in parts:
        if _is_acronym(p):
            out_parts.append(p.upper())
        else:
            out_parts.append(p.capitalize())
    normalized = " ".join(out_parts).strip()
    if not normalized or len(normalized) > 80:
        return None
    if normalized.lower() in _GENERIC_SKILL_TOKENS:
        return None
    return normalized

# ---------- preserve common multi-word phrases & improved splitting ----------
COMMON_MULTI_PHRASES = set([
    'industrial design', 'product design', 'colour theory', 'color theory',
    'user research', 'design systems', 'information architecture', 'human interface',
    'user flows', 'wire framing', 'wireframing', 'visual design'
])



def auto_normalize_skills(raw_skill_candidates):
    """
    Normalizer with reduced over-splitting:
    - preserves common multi-word phrases
    - avoids producing single-word fragments from complex sentences
    """
    if not raw_skill_candidates:
        return []
    seen = set()
    out = []
    for item in raw_skill_candidates:
        if not item:
            continue
        # If the candidate is long and looks like a grouped list "A, B or C", try to split more conservatively
        # (but keep multi-word nouns intact)
        # First break by main delimiters (commas/semicolon/newline) but then recombine adjacent tokens if they form a known phrase
        pieces = re.split(_SPLIT_DELIM_RE, item)
        parens = _PARENS_RE.findall(item)
        if parens:
            for p in parens:
                pieces.extend(re.split(_SPLIT_DELIM_RE, p))
        expanded = []
        for p in pieces:
            if p and re.search(r'\band\b', p, flags=re.I) and len(p.split(',')) == 1 and len(p.split()) <= 8:
                for sub in re.split(r'\band\b', p, flags=re.I):
                    expanded.append(sub)
            else:
                expanded.append(p)
        # Post-process each piece: attempt to keep two-word noun phrases intact if known
        for piece in expanded:
            p = (piece or "").strip()
            if not p:
                continue
            p = re.sub(r'^\s*[:\-\u2022\*]+\s*', '', p)
            # guard: if piece contains 'or' with 'or a related field' we keep the preceding phrase
            if re.search(r'\bor a related field\b', p, flags=re.I):
                # try to pick the noun phrase preceding 'or a related field'
                m = re.search(r'(?i)([A-Za-z \-]+?)\s*(?:,|\s)?\s*or a related field', p)
                if m:
                    cand = m.group(1).strip()
                    norm = _normalize_token(cand)
                    if norm:
                        nk = norm.lower()
                        if nk not in seen:
                            seen.add(nk); out.append(norm)
                        continue
            # if piece is a multi-word phrase and matches common list, preserve whole phrase
            cleaned = re.sub(r'\s+', ' ', p).strip()
            low = cleaned.lower()
            if low in COMMON_MULTI_PHRASES:
                norm = _normalize_token(cleaned)
                if norm:
                    nk = norm.lower()
                    if nk not in seen:
                        seen.add(nk); out.append(norm)
                    continue
            # otherwise normalize normally, but avoid single-word fragments from long sentences
            norm = _normalize_token(cleaned)
            if not norm:
                # fallback: if cleaned has >2 words, attempt to extract noun-noun bigrams
                words = [w for w in re.split(r'[\s\-_]+', cleaned) if w and len(w) > 1]
                if len(words) >= 2:
                    # try producing two-word candidates from adjacent words and keep ones matching common phrases
                    for i in range(len(words)-1):
                        cand = f"{words[i]} {words[i+1]}"
                        if cand.lower() in COMMON_MULTI_PHRASES:
                            nn = _normalize_token(cand)
                            if nn and nn.lower() not in seen:
                                seen.add(nn.lower()); out.append(nn)
                continue
            nk = norm.lower()
            if nk not in seen:
                seen.add(nk); out.append(norm)
    return out
def _dedupe_preserve_order(lst):
    seen = set(); out = []
    for x in lst:
        if not x: continue
        k = x.lower().strip()
        if k in seen: continue
        seen.add(k); out.append(x)
    return out

# ---------- ----------



def normalize_token(tok):
    return (tok or "").lower().strip()

def kb_lookup(token):
    if Skill is None:
        return None, None, []
    t = normalize_token(token)
    try:
        skill = Skill.objects.filter(Q(normalized=t) | Q(name__iexact=token)).first()
        if skill:
            syns = skill.synonyms or []
            matched_syns = [s for s in syns if normalize_token(s) == t or t in normalize_token(s) or normalize_token(s) in t]
            return skill, skill.name, matched_syns
    except Exception:
        pass
    try:
        for s in Skill.objects.all()[:10000]:
            for syn in s.synonyms or []:
                if normalize_token(syn) == t or t in normalize_token(syn) or normalize_token(syn) in t:
                    return s, s.name, [syn]
    except Exception:
        pass
    return None, None, []

EQUIVALENCES = {
    "docker": ["containerization", "containers", "containerized"],
    "react": ["react.js", "reactjs"],
    "python": ["py"],
    "flask": ["rest api", "rest"],
    "aws": ["amazon web services", "cloud"],
    "gcp": ["google cloud"],
    "postgresql": ["postgres", "postgresql"],
}

# ---------- tokenizers and small helpers (unchanged) ----------
VERB_KEYWORDS = [
    'design', 'designed', 'develop', 'developed', 'build', 'built', 'lead', 'led', 'managed',
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
        # protect multi-word degree phrases
        if strict and len(token.split()) > 6:
            continue
        if len(token) < 2 or len(token) > 200:
            continue
        tnorm = token.lower()
        if tnorm in seen:
            continue
        seen.add(tnorm)
        skills.append(token)
    return skills

# ---------- contact extraction (unchanged) ----------
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

# ---------- spaCy-backed NER (unchanged) ----------
def ner_extract_entities(text, max_orgs=6, max_persons=3):
    persons = []
    orgs = []
    dates = []
    titles = []

    nlp = _load_spacy_model()
    if nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                lab = ent.label_.upper()
                ent_text = ent.text.strip()
                if not ent_text:
                    continue
                if lab in ("PERSON",) and len(persons) < max_persons:
                    if ent_text not in persons: persons.append(ent_text)
                elif lab in ("ORG", "ORGANIZATION", "COMPANY", "FAC") and len(orgs) < max_orgs:
                    if ent_text not in orgs: orgs.append(ent_text)
                elif lab in ("DATE", "TIME"):
                    if ent_text not in dates: dates.append(ent_text)
            try:
                chunks = list(doc.noun_chunks)
                for i, chunk in enumerate(chunks):
                    ch_text = chunk.text.strip()
                    for o in orgs:
                        if o and o in ch_text:
                            if i > 0:
                                cand = chunks[i-1].text.strip()
                                if cand and cand not in titles:
                                    titles.append(cand)
            except Exception:
                pass
            if persons or orgs or dates or titles:
                return {"persons": persons[:max_persons], "orgs": orgs[:max_orgs], "dates": dates, "title_candidates": titles}
        except Exception:
            log.debug("spaCy NER failed, falling back to regex heuristics", exc_info=True)

    org_pattern = re.compile(
        r'([A-Z][A-Za-z0-9&\.\-]{2,}(?: (?:Ltd|LLP|Inc|Corp|Company|Solutions|Systems|Academy|College|Institute|Technologies|Tech|Lab|LLC|GmbH|Pvt|Private|Limited))?)'
    )
    for m in org_pattern.finditer(text):
        cand = m.group(1).strip()
        if cand and len(cand) < 120 and cand not in orgs:
            orgs.append(cand)
            if len(orgs) >= max_orgs: break

    for m in re.finditer(r'([A-Za-z0-9 \-\,&\.\']{2,60})\s+(?:at|@)\s+([A-Z][A-Za-z0-9&\.\- ]{2,80})', text):
        comp = m.group(2).strip()
        if comp and comp not in orgs:
            orgs.append(comp)
            if len(orgs) >= max_orgs: break

    email_match = re.search(r'([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})', text)
    if email_match:
        cand = email_match.group(1).replace('.', ' ').replace('_', ' ').title().strip()
        if cand and cand not in persons: persons.append(cand)

    header_line = None
    for line in text.splitlines()[:8]:
        line = line.strip()
        if not line: continue
        if re.search(r'\d{3,}', line) and '@' not in line:
            continue
        if 1 <= len(line.split()) <= 4 and all(re.match(r'^[A-Z][a-z\-\'\.]+$', w) for w in line.split()):
            header_line = line; break
    if header_line and header_line not in persons:
        persons.append(header_line.strip())

    for m in re.finditer(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*(?:\d{4}|\d{2})', text, flags=re.I):
        d = m.group(0).strip()
        if d not in dates: dates.append(d)
    for m in re.finditer(r'\b(19|20)\d{2}\b', text):
        d = m.group(0)
        if d not in dates: dates.append(d)

    for m in re.finditer(r'([A-Za-z/&\-\.\s]{2,60})\s+(?:at|@)\s+[A-Z]', text):
        cand = m.group(1).strip()
        if 2 <= len(cand.split()) <= 7 and cand not in titles:
            titles.append(cand)
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if re.search(r'\b(engineer|developer|manager|lead|intern|analyst|architect|scientist|consultant)\b', line, flags=re.I):
            if len(line.split()) <= 10 and line not in titles:
                titles.append(line)

    return {"persons": persons[:max_persons], "orgs": orgs[:max_orgs], "dates": dates, "title_candidates": titles[:10]}

# ---------- parsing pipeline (mostly preserved with enhancements) ----------
def build_resume_prompt(raw_text):
    return {"task": "Parse resume into structured JSON", "input_text": raw_text or ""}


def _split_sections_by_headings(text: str) -> Dict[str, str]:
    """
    Robustly split free-form job description text into sections where headings are lines
    that match words in SECTION_HEADINGS (case-insensitive) followed optionally by ':'.
    Returns dict mapping heading -> content. If no headings found, return {'body': text}.
    """
    if not text:
        return {"body": ""}

    s_text = text.replace('\r', '\n')
    # collapse repeated newlines to two to keep paragraphs
    s_text = re.sub(r'\n{2,}', '\n\n', s_text)
    lines = s_text.splitlines()
    indexes = []
    for i, line in enumerate(lines):
        clean = line.strip()
        # match heading tokens: e.g., "Responsibilities:" or "Required skills"
        for h in SECTION_HEADINGS:
            # allow small variations and trailing colon or whitespace
            if re.fullmatch(rf"{h}\b[:\s]*", clean, flags=re.I):
                indexes.append((i, clean))
                break
    if not indexes:
        # no explicit headings - return whole text in 'body'
        return {"body": s_text.strip()}

    sections = {}
    for idx, (line_idx, heading_line) in enumerate(indexes):
        start = line_idx + 1
        end = indexes[idx + 1][0] if idx + 1 < len(indexes) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        # normalize heading text (strip trailing colon/space)
        heading_norm = re.sub(r'[:\s]+$', '', heading_line).strip()
        sections[heading_norm] = content
    return sections

_JACCARD_HIGH = 0.5
_JACCARD_MED = 0.25

def _extract_keywords_from_phrase(phrase, max_terms=6):
    if not phrase:
        return []
    txt = re.sub(r'[^A-Za-z0-9\+\#\.\s\-]', ' ', phrase)
    parts = [p.strip().lower() for p in re.split(r'[\s\-]+', txt) if p.strip()]
    parts = [p for p in parts if p not in _SHORT_STOPWORDS and len(p) > 1]
    return parts[:max_terms]



# Preferred embedding model (sentence-transformers wrapper)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")  # preferred
EMBEDDING_FALLBACK = "all-MiniLM-L6-v2"
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.65))

# lazy holder
_embedding_model = None
_embedding_kind = None

@lru_cache(maxsize=1)
def get_embedding_model():
    global _embedding_model, _embedding_kind
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        log.exception("sentence_transformers not installed: %s", e)
        return None

    # try preferred BGE first
    for candidate in (EMBEDDING_MODEL, EMBEDDING_FALLBACK):
        try:
            m = SentenceTransformer(candidate)
            _embedding_model = m
            _embedding_kind = "st"
            log.info("Loaded embedding model: %s", candidate)
            return ("st", m)
        except Exception as exc:
            log.debug("Failed to load SentenceTransformer '%s': %s", candidate, exc, exc_info=True)

    log.warning("No sentence-transformers embedding model could be loaded.")
    return None

def embed_texts(texts):
    """
    Returns numpy.ndarray shape (N, D) or None on failure.
    Uses the loaded sentence-transformers (BGE or fallback).
    """
    if texts is None:
        return None
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = list(texts)

    backend = get_embedding_model()
    if not backend:
        log.debug("embed_texts: no embedding backend loaded")
        return None
    kind, model = backend
    try:
        # SentenceTransformer: model.encode([...], convert_to_numpy=True)
        try:
            emb = model.encode(texts_list, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            emb = model.encode(texts_list, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
       
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        emb = emb / norms
        return emb
    except Exception as e:
        log.exception("embed_texts failed: %s", e)
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
        if a.ndim == 1: a = a.reshape(1, -1)
        if b.ndim == 1: b = b.reshape(1, -1)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T)
    except Exception:
        log.exception("cosine_sim fallback failed")
        return None

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


import math
if 'semantic_match_score' not in globals():
    def semantic_match_score(resume_parsed, job_parsed):
        """
        Lightweight, faithful fallback of the original semantic_match_score.
        Returns dict {score (0..10), rating, breakdown, raw_points}
        """
        # try to call original if still present under different name
        if 'semantic_match_score' in globals() and globals()['semantic_match_score'] is not semantic_match_score:
            return globals()['semantic_match_score'](resume_parsed, job_parsed)

        # Defaults and constants
        CRIT_WEIGHT = 0.50
        BEN_WEIGHT = 0.10
        EXP_WEIGHT = 0.15
        PROJ_WEIGHT = 0.25
        SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", 0.65))

        # Basic safe-get helpers
        def _get_resume_skills(rp):
            # collect raw lists from parsed resume
            raw = []
            raw += (rp.get('technical_skills') or [])
            raw += (rp.get('soft_skills') or [])
            # include short text cues from summary
            summ = (rp.get('summary') or "")
            if summ:
                # extract obvious tool names from summary heuristically
                for m in re.findall(r'\b(Adobe|Photoshop|Illustrator|InDesign|Figma|Sketch|XD|user research|colour theory|color theory|prototype|prototyping)\b', summ, flags=re.I):
                    raw.append(m)
            return list(raw)

        def _normalize_list(xs):
            return [x for x in xs if x]

        # derive both raw and normalized forms — normalization is critical
        resume_raw_skills = _get_resume_skills(resume_parsed or {})
        # normalize & dedupe canonical tokens for matching
        try:
            resume_skills = _dedupe_preserve_order(auto_normalize_skills(resume_raw_skills))
        except Exception:
            # fallback: lowercase dedupe
            resume_skills = list({(s or "").lower().strip(): s for s in resume_raw_skills}.values())


        job_crit = _normalize_list(job_parsed.get('critical_skills') if isinstance(job_parsed, dict) else [])
        job_ben = _normalize_list(job_parsed.get('beneficial_skills') if isinstance(job_parsed, dict) else [])

        # short-circuit: if no job_crit, treat whole JD as a body of text and score with projects/exp
        num_crit = max(1, len(job_crit))

        # helper: compute credit for a single job skill using available functions if any
        def compute_skill_credit(skill_text):
            # 1) KB / exact match via is_equivalent if available
            if not skill_text:
                return 0.0, "0.00 (missing)"
            # Normalize the job skill candidate (prefer multiword preservation)
            try:
                job_norms = auto_normalize_skills([skill_text])
                job_skill = job_norms[0] if job_norms else skill_text.strip()
            except Exception:
                job_skill = skill_text.strip()

            # 1) Knowledge base or exact canonical matching using normalized tokens
            try:
                if 'is_equivalent' in globals():
                    matched, rule_score = is_equivalent(job_skill, resume_skills)
                    if matched and rule_score >= 0.9:
                        return 1.0, f"1.00 (exact rule {rule_score:.2f})"
                    if matched and rule_score >= 0.5:
                        return 0.8, f"0.8 (transferable {rule_score:.2f})"
            except Exception:
                pass


            # 2) simple normalized containment fallback
            low = (job_skill or "").lower().strip()
            for r in resume_skills:
                if not r: continue
                rnorm = (r or "").lower().strip()
                if low == rnorm:
                    return 1.0, "1.00 (exact token)"
                if low in rnorm or rnorm in low:
                    return 0.8, "0.8 (substr)"


            # 3) embedding semantic fallback if available
            try:
                model_tuple = None
                if 'get_embedding_model' in globals():
                    model_tuple = get_embedding_model()
                if model_tuple:
                    # build small embeddings and compare
                    s_emb = None
                    r_embs = None
                    try:
                        s_emb = embed_texts([skill_text])
                        r_embs = embed_texts(resume_skills) if resume_skills else None
                        if s_emb is not None and r_embs is not None:
                            sims = cosine_sim(s_emb, r_embs)
                            best = float(sims.max())
                            if best >= SIM_THRESHOLD:
                                return 1.0, f"1.00 (semantic {best:.2f})"
                            if best >= SIM_THRESHOLD - 0.15:
                                return 0.8, f"0.8 (partial sem {best:.2f})"
                            if best >= SIM_THRESHOLD - 0.30:
                                return 0.45, f"0.45 (weak sem {best:.2f})"
                    except Exception:
                        pass
            except Exception:
                pass

            # default missing
            return 0.0, "0.00 (missing)"

        weighted_sum = 0.0
        breakdown = {"critical": {}, "beneficial": {}}

        # critical loop
        for s in job_crit:
            credit, reason = compute_skill_credit(s)
            contrib = (credit * CRIT_WEIGHT) / num_crit
            weighted_sum += contrib
            breakdown["critical"][s] = f"{credit:.2f} ({reason})"

        # beneficial loop
        num_ben = max(1, len(job_ben))
        for b in job_ben:
            credit, reason = compute_skill_credit(b)
            contrib = (credit * BEN_WEIGHT) / num_ben
            weighted_sum += contrib
            breakdown["beneficial"][b] = f"{credit:.2f} ({reason})"

        # Experience credit (naive fallback)
        exp_credit = 0.0 
        try:
            years = None
            # 1) If resume_parsed provided an explicit 'years' field use it
            if isinstance(resume_parsed, dict):
                if isinstance(resume_parsed.get('years'), (int, float)):
                    years = int(resume_parsed.get('years'))
            # 2) if not, infer from experience blocks count and text
            exp_list = resume_parsed.get('experience') or []
            if exp_list and isinstance(exp_list, list):
                # if there are >=2 experience entries treat as mid/senior
                if len(exp_list) >= 3:
                    exp_credit = 1.0
                elif len(exp_list) >= 1:
                    # if any role contains 'design' or 'product' treat as relevant
                    found_relevant = any(re.search(r'\b(product|design|ux|ui|industrial)\b', json.dumps(e, default=str), flags=re.I) for e in exp_list)
                    exp_credit = 0.9 if found_relevant else 0.6
            # 3) years extracted from summary/body text (fall back)
            if years is None:
                text_blob = " ".join(filter(None, [resume_parsed.get('summary',''), " ".join(resume_raw_skills)]))
                m = re.search(r'(\d+)\+?\s+years?', text_blob)
                if m:
                    years = int(m.group(1))
                    if years >= 5:
                        exp_credit = max(exp_credit, 1.0)
                    elif years >= 3:
                        exp_credit = max(exp_credit, 0.9)
                    else:
                        exp_credit = max(exp_credit, 0.6)
            # 4) if job explicitly asked for years_required use that threshold
            req_years = job_parsed.get('years_required') if isinstance(job_parsed, dict) else None
            if req_years:
                if years and years >= req_years:
                    exp_credit = max(exp_credit, 1.0)
                elif years and years >= max(0, req_years - 1):
                    exp_credit = max(exp_credit, 0.7)
        except Exception:
            exp_credit = 0.2

        weighted_sum += exp_credit * EXP_WEIGHT

        # Project credit (simple averaging)
        proj_credit = 0.0
        try:
            projects = resume_parsed.get('projects') or []
            if projects:
                total_rel = 0.0
                for p in projects:
                    # project text
                    ptxt = (p.get('description') or "") + " " + (p.get('title') or "")
                    # token overlap with criticals
                    overlap = 0
                    for s in job_crit:
                        if (s or "").lower() in ptxt.lower() or (ptxt.lower() in (s or "").lower()):
                            overlap += 1
                    token_score = min(1.0, overlap / max(1.0, len(job_crit)))
                    # try semantic
                    sem = 0.0
                    try:
                        embp = embed_texts([ptxt])
                        embj = embed_texts(job_crit)
                        if embp is not None and embj is not None:
                            sims = cosine_sim(embj, embp)
                            sem = float(sims.max())
                    except Exception:
                        sem = 0.0
                    rel = max(token_score * 0.8, sem * 0.95)
                    # small boost if measurable indicated
                    if re.search(r'\b(reduce|increase|improve|%|percent|revenue|users|engagement)\b', ptxt, flags=re.I):
                        rel += 0.05
                    total_rel += min(1.0, rel)
                proj_credit = (total_rel / len(projects))
        except Exception:
            proj_credit = 0.0

        weighted_sum += proj_credit * PROJ_WEIGHT

        # final clamp and scale
        weighted_sum = max(0.0, min(1.0, weighted_sum))
        final_score = int(round(weighted_sum * 10.0))
        if final_score >= 8:
            rating = "Strong"
        elif final_score >= 6:
            rating = "Moderate"
        elif final_score >= 4:
            rating = "Fair"
        else:
            rating = "Weak"

        exp_label = "Unknown"
        if exp_credit >= 0.9:
            exp_label = "Experience Matched"
        elif exp_credit >= 0.6:
            exp_label = "Partial Experience"
        else:
            exp_label = "Experience Insufficient"

        # Projects details
        projects_info = {}
        try:
            proj_list = resume_parsed.get('projects') or []
            projects_info = { p.get('name','Project')[:80]: f"rel:{round( ( (float(sum(1 for s in job_crit if (s or '').lower() in ((p.get('description') or '').lower() or p.get('title','').lower())) ) / max(1,len(job_crit))) ), 2)}" for p in proj_list }
        except Exception:
            projects_info = {}

        return {
            "score": final_score,
            "rating": rating,
            "breakdown": breakdown,
            "raw_points": round(weighted_sum, 4),
            "experience": exp_label,
            "projects_count": len(resume_parsed.get('projects') or []),
            "projects": projects_info
        }

def llm_suggest_for_gaps(resume_snippet, job_snippet, gaps):
    """
    Ask the LLM to generate suggestions / example bullets for a list of gaps.
    We send one prompt covering all gaps (JSON output).
    Returns a dict mapping skill -> suggestion dict.
    """
    if not gaps:
        return {}

    # Limit number of gaps sent to avoid tokens explosion
    MAX_GAPS = 12
    gaps_to_send = gaps[:MAX_GAPS]

    # Build short context
    prompt = {
        "resume_skills": resume_snippet.get("skills", [])[:60],
        "resume_projects": resume_snippet.get("projects", [])[:6],
        "job_title": job_snippet.get("title", ""),
        "job_critical": job_snippet.get("critical_skills", [])[:60],
        "gaps": [{"skill": g["skill"], "type": g.get("type"), "importance": g.get("importance", 1.0)} for g in gaps_to_send]
    }

    # instruct LLM to return JSON
    system = "You are a concise career coach. Return a JSON object mapping each gap skill to an object with keys: suggestion (1-2 sentences), example_bullets (list up to 2). Respond with only valid JSON."
    user_text = "Context JSON:\n" + json.dumps(prompt, indent=2) + "\n\nProduce a JSON mapping each 'skill' -> {\"suggestion\":..., \"example_bullets\": [...]}."

    resp = ai_client.call_llm(prompt=user_text, system=system, max_tokens=512, temperature=0.2)
    text = resp.get("text", "")
    if not text:
        log.warning("LLM returned no text for gap suggestions, fallback to rule text.")
        return {}

    # Try to extract JSON from response robustly
    try:
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            js = json.loads(m.group(1))
            return js if isinstance(js, dict) else {}
        # fallback - try evaluate if it's pure JSON list or dict
        js = json.loads(text)
        return js if isinstance(js, dict) else {}
    except Exception as e:
        log.exception("Failed to parse LLM JSON - returning empty suggestions: %s", e)
        return {}

def gap_analysis(resume_parsed, job_parsed, top_n=8):
    resume_parsed = resume_parsed or {}
    job_parsed = job_parsed or {}

    # 1) collect resume and job skill sets
    resume_raw = []
    resume_raw += (resume_parsed.get('technical_skills') or [])
    resume_raw += (resume_parsed.get('soft_skills') or [])
    summary_text = (resume_parsed.get('summary') or "") or ""
    if summary_text:
        for m in re.findall(r'\b(Adobe|Photoshop|Illustrator|InDesign|Figma|Sketch|XD|user research|prototype|prototyping)\b', summary_text, flags=re.I):
            resume_raw.append(m)

    try:
        resume_norm = _dedupe_preserve_order(auto_normalize_skills(resume_raw))
    except Exception:
        resume_norm = _dedupe_preserve_order([ (s or "").strip() for s in resume_raw ])

    crit_raw = job_parsed.get('critical_skills') or []
    ben_raw = job_parsed.get('beneficial_skills') or []
    try:
        crit_norm = _dedupe_preserve_order(auto_normalize_skills(crit_raw))
    except Exception:
        crit_norm = _dedupe_preserve_order([ (s or "").strip() for s in crit_raw ])
    try:
        ben_norm = _dedupe_preserve_order(auto_normalize_skills(ben_raw))
    except Exception:
        ben_norm = _dedupe_preserve_order([ (s or "").strip() for s in ben_raw ])

    # lowercase resume tokens for cheap lookups
    resume_skills_lower = [ (s or "").lower().strip() for s in resume_norm ]
    resume_token_sets = [(r, set(re.findall(r'\w+', r))) for r in resume_skills_lower if r]

    SIM = SIM_THRESHOLD

    gaps = []
    transferable_candidates = []

    # Precompute embeddings once (resume and critical skills)
    emb_resume = None
    emb_crit = None
    try:
        if resume_norm:
            emb_resume = embed_texts(resume_norm)   # shape (R, D)
    except Exception:
        emb_resume = None

    try:
        if crit_norm:
            emb_crit = embed_texts(crit_norm)       # shape (C, D)
    except Exception:
        emb_crit = None

    # 2) Evaluate each critical skill
    for idx, s in enumerate(crit_norm):
        s_norm = (s or "").strip()
        if not s_norm:
            continue

        # a) exact match
        if s_norm.lower() in resume_skills_lower:
            continue

        # b) KB / equivalences (fast)
        try:
            if 'is_equivalent' in globals():
                matched, rule_score = is_equivalent(s_norm, resume_norm)
                if matched and rule_score >= 0.9:
                    continue
                if matched and rule_score >= 0.5:
                    gaps.append({
                        "skill": s,
                        "type": "transferable",
                        "importance": 1.0,
                        "suggestion": f"Related skill found via KB. Reframe a bullet to show {s}."
                    })
                    continue
        except Exception:
            pass

        # c) token overlap quick check
        tokens_s = set(re.findall(r'\w+', s_norm.lower()))
        best_ov = 0.0
        for r_text, tokens_r in resume_token_sets:
            ov = len(tokens_s & tokens_r) / max(1, len(tokens_s | tokens_r))
            if ov > best_ov:
                best_ov = ov
        if best_ov >= 0.5:
            gaps.append({
                "skill": s,
                "type": "transferable",
                "importance": 1.0,
                "suggestion": f"Found related words by token-overlap (score {best_ov:.2f}). Highlight transferability."
            })
            continue
        if best_ov >= 0.25:
            gaps.append({
                "skill": s,
                "type": "transferable",
                "importance": 0.8,
                "suggestion": f"Partial token overlap (score {best_ov:.2f}). Clarify relevance in bullets."
            })
            continue

        # d) embeddings semantic similarity (fast matrix op)
        sem_handled = False
        try:
            if emb_crit is not None and emb_resume is not None:
                sim_row = cosine_sim(emb_crit[idx:idx+1], emb_resume)   # shape (1, R)
                if sim_row is not None:
                    max_sim = float(sim_row.max())
                    if max_sim >= SIM:
                        sem_handled = True  # treat as matched
                    elif max_sim >= (SIM - 0.15):
                        gaps.append({
                            "skill": s,
                            "type": "transferable",
                            "importance": 0.9,
                            "suggestion": f"Related skill found semantically (sim {max_sim:.2f}). Emphasize similarity."
                        })
                        sem_handled = True
        except Exception:
            pass

        if sem_handled:
            continue

        # otherwise missing
        gaps.append({
            "skill": s,
            "type": "missing",
            "importance": 1.0,
            "suggestion": f"This role requires '{s}' but it isn't present in the resume."
        })

    # 3) Beneficial skills (lower importance)
    for b_raw in ben_norm:
        b_low = (b_raw or "").lower().strip()
        matched = False
        for r in resume_skills_lower:
            if not r: continue
            if b_low == r or b_low in r or r in b_low:
                matched = True
                break
        if not matched:
            gaps.append({
                "skill": b_raw,
                "type": "learnable",
                "importance": 0.4,
                "suggestion": f"Optional: learn or demonstrate '{b_raw}' with a short artifact or course."
            })

    # 4) Extras (resume-only strengths)
    extra_skills = []
    jd_tokens = set([x.lower().strip() for x in (crit_norm + ben_norm)])
    for r in resume_skills_lower:
        if not r:
            continue
        if r not in jd_tokens and re.search(r'\b(design|ux|ui|product|prototype|photoshop|illustrator|figma|indesign|usability|research|prototype)\b', r, flags=re.I):
            extra_skills.append({
                "skill": r,
                "type": "extra",
                "importance": 0.6,
                "suggestion": f"You have '{r}' on your resume; add a portfolio link or strong bullet to show relevance."
            })

    # combine & sort
    primary = [g for g in gaps if g['type'] in ('missing','transferable')]
    learnable = [g for g in gaps if g['type'] == 'learnable']
    combined = primary + extra_skills + learnable

    def sort_key(x):
        t = x.get('type')
        type_rank = 3
        if t == 'missing': type_rank = 0
        elif t == 'transferable': type_rank = 1
        elif t == 'extra': type_rank = 2
        elif t == 'learnable': type_rank = 4
        return (-x.get('importance', 0.0), type_rank, x.get('skill',''))

    combined_sorted = sorted(combined, key=sort_key)
    top_gaps = combined_sorted[:top_n]

    # 5) Use LLM to enrich suggestions for the top gaps (batch call)
    try:
        resume_snip = {"skills": resume_norm[:80], "projects": [p.get("name") for p in (resume_parsed.get("projects") or [])[:6]]}
        job_snip = {"title": job_parsed.get("title"), "critical_skills": crit_norm[:80]}
        llm_results = llm_suggest_for_gaps(resume_snip, job_snip, top_gaps)
        if llm_results:
            # update top_gaps suggestions with LLM results where present
            for g in top_gaps:
                sk = g["skill"]
                if sk in llm_results:
                    val = llm_results[sk]
                    if isinstance(val, dict):
                        g["suggestion"] = val.get("suggestion", g.get("suggestion"))
                        g["example_bullets"] = val.get("example_bullets", [])
    except Exception:
        log.exception("LLM enrichment for gap suggestions failed; returning baseline suggestions.")

    return top_gaps

def generate_personalized_recommendation(resume_parsed, job_parsed, gaps):
    """
    Prefer LLM-generated recommendations for richer output; fall back to prior rule-based generator.
    """
    try:
        # small structured prompt
        prompt = (
            "You are an expert career coach. Given:\n\n"
            f"Resume (skills): {json.dumps(resume_parsed.get('technical_skills', [])[:40])}\n\n"
            f"Job (title & critical): {job_parsed.get('title','')} -- {json.dumps(job_parsed.get('critical_skills',[])[:40])}\n\n"
            f"Gaps: {json.dumps(gaps[:8])}\n\n"
            "Produce a structured recommendation JSON with keys: kind (one of 'project','certification','course','summary'), "
            "text (concise recommendation), example_bullets (list of 1-3 example resume bullets). Keep json compact."
        )
        resp = ai_client.call_mixtral(prompt=prompt, system="You are a concise recommendation generator.", temperature=0.2, max_tokens=400)
        text = resp.get("text") or ""
        # attempt to parse JSON from response
        # safe extraction: find first { ... } block
        import re
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            import ast, json
            try:
                js = json.loads(m.group(1))
                return js
            except Exception:
                # if not valid JSON, return text blob
                return {"kind": "llm", "text": text}
        else:
            return {"kind": "llm", "text": text}
    except Exception:
        # fallback to original rule-based generator if LLM fails
        try:
            return globals().get('generate_personalized_recommendation', None)(resume_parsed, job_parsed, gaps)
        except Exception:
            # last fallback minimal
            return {"kind": "fallback", "text": "Consider reframing top projects and adding measurable outcomes."}

def handle_nl_query(parsed_resume: dict, parsed_job: dict, history: list, query: str) -> dict:
    """
    Use Mixtral to answer an arbitrary query about the resume+job.
    Returns dict: {answer: str, sources: [...]}
    """
    # Build compact context for prompt (keep it short - LLM context matters)
    resume_snippet = {
        "skills": parsed_resume.get("technical_skills", [])[:30],
        "soft_skills": parsed_resume.get("soft_skills", [])[:30],
        "projects": [p.get("name") for p in (parsed_resume.get("projects") or [])[:6]]
    }
    job_snippet = {
        "title": parsed_job.get("title") or "",
        "critical_skills": parsed_job.get("critical_skills", [])[:30],
        "beneficial_skills": parsed_job.get("beneficial_skills", [])[:30],
        "responsibilities": (parsed_job.get("responsibilities") or [])[:6]
    }

    system = (
        "You are an expert resume and recruiting assistant. Use the provided resume and job information to "
        "answer user questions concisely, give actionable suggestions, and supply example resume bullets where appropriate. "
        "If the user asks for rewrites, provide 2 variants: (1) concise, (2) impact + metrics."
    )

    prompt_obj = {
        "resume": resume_snippet,
        "job": job_snippet,
        "history": history[-6:] if history else [],
        "query": query
    }

    prompt = (
        "Context (JSON):\n" + json.dumps(prompt_obj, indent=2) + "\n\n"
        "Answer the query in clear sections. If you provide recommendations, label them 'Recommendation' and give 2-3 concrete steps. "
        "If the user asks 'rewrite' or 'improve', produce example bullets. Keep answers actionable and role-aware."
    )

    resp = ai_client.call_mixtral(prompt=prompt, system=system, temperature=0.1, max_tokens=512)
    text = resp.get("text") or resp.get("message") or ""
    return {"answer": text, "raw": resp}




def call_claude_parse(prompt_json):
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

    text = re.sub(r'\r\n', '\n', raw)
    text = re.sub(r'\n{3,}', '\n\n', text)

    contact = _extract_contact_info(text)
    try:
        ner = ner_extract_entities(text)
    except Exception:
        ner = {"persons": [], "orgs": [], "dates": [], "title_candidates": []}

    if not contact.get('name') and ner.get('persons'):
        contact['name'] = ner['persons'][0]

    parsed['contact'] = contact
    sections = _split_sections_by_headings(text)

    # technical skills extraction (unchanged flow) ...
    skills_candidates = []
    for k, v in sections.items():
        k_low = k.lower()
        if any(x in k_low for x in ('skill', 'tool', 'tech', 'key skill', 'technical')):
            skills_candidates.append((v, False))

    if not skills_candidates:
        header_snippet = text.split('\n\n', 1)[0][:800]
        header_tokens = _lines_to_skills(header_snippet, strict=True)
        if header_tokens:
            skills_candidates.append((", ".join(header_tokens), True))

    tech_skills = []
    for block, strict_flag in skills_candidates:
        tech_skills += _lines_to_skills(block, strict=strict_flag)

    for m in re.finditer(r'(?i)(?:tech(?: ?stack)?\s*[:\-])\s*(.+)', text):
        cap = m.group(1).strip()
        tech_skills += _lines_to_skills(cap, strict=False)

    # seen = set()
    # final_tech = []
    # for t in tech_skills:
    #     tn = t.lower().strip()
    #     if tn and tn not in seen:
    #         final_tech.append(t.strip()); seen.add(tn)

    tech_skills = _dedupe_preserve_order(tech_skills)

    # 2) apply the existing filtering to remove non-skill tokens
    filtered_raw = [t.strip() for t in tech_skills
                    if t and not re.search(r'\b(project|product|intern|manager|lead|experience|develop|worked)\b', t, flags=re.I)]

    normalized = auto_normalize_skills(filtered_raw)

    parsed['technical_skills'] = _dedupe_preserve_order(normalized)

    # soft skills
    soft_found = []
    for ss in COMMON_SOFT_SKILLS:
        if re.search(r'\b' + re.escape(ss) + r'\b', text, flags=re.I):
            soft_found.append(ss)
    if re.search(r'\blead(er|ing)?\b|\bpresident\b|\bmanage(d|r)?\b', text, flags=re.I):
        if 'leadership' not in soft_found:
            soft_found.append('leadership')
    parsed['soft_skills'] = soft_found

    # experience extraction (preserved) ...
    exps = []
    exp_block = None
    for h in sections:
        if re.search(r'(?i)experience', h):
            exp_block = sections[h]; break
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
                    start = mdate.group(1); end = mdate.group(2)
                lines_in_buf = block_text.split('. ')
                if len(lines_in_buf) > 0:
                    first = buffer_lines[0]
                    parts = re.split(r'\s{2,}|\s-\s|,', first)
                    if len(parts) >= 2:
                        company = parts[-1].strip(); role = parts[0].strip()
                    else:
                        if ' - ' in first:
                            role, company = [p.strip() for p in first.split(' - ', 1)]
                        elif '|' in first:
                            role, company = [p.strip() for p in first.split('|', 1)]
                        else:
                            role = first.strip()
                exps.append({"role": role or "", "company": company or "", "start": start, "end": end, "description": block_text})
                buffer_lines = []
            else:
                buffer_lines.append(line)
        if buffer_lines:
            block_text = " ".join(buffer_lines)
            exps.append({"role": buffer_lines[0], "company": "", "start": None, "end": None, "description": block_text})
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
                    if re.search(r'\d{4}', lines[j]): break
                    desc.append(lines[j]); j += 1
                cand.append({"role": ln, "company": "", "start": None, "end": None, "description": " ".join(desc)})
                i = j
            else:
                i += 1
        exps = cand

    parsed['experience'] = exps
    try:
        ner_glob = ner if 'ner' in locals() else ner_extract_entities(text)
        primary_org = ner_glob.get('orgs', [None])[0] if ner_glob else None
        for i, e in enumerate(exps):
            if isinstance(e, dict):
                if not e.get('company'):
                    found = None
                    desc = (e.get('description') or "")
                    for o in ner_glob.get('orgs', []) if ner_glob else []:
                        if o and o in desc:
                            found = o; break
                    if found:
                        e['company'] = found
                    elif primary_org:
                        if primary_org in text and len(exps) > 1:
                            e['company'] = primary_org
                if not e.get('role'):
                    for cand in ner_glob.get('title_candidates', []) if ner_glob else []:
                        if cand and cand in e.get('description',''):
                            e['role'] = cand; break
                    if not e.get('role'):
                        first_line = (e.get('description') or "").splitlines()[0].strip()
                        if first_line and len(first_line.split()) <= 8:
                            e['role'] = first_line
    except Exception:
        pass

    # projects extraction (preserved) ...
    def _extract_tech_list_from_text(txt):
        if not txt:
            return []
        m = re.search(r'(?i)(?:tech(?: ?stack)?|technologies|tools?)[:\s]*(.+)', txt)
        if m:
            cand = m.group(1)
        else:
            par = re.search(r'\(([^)]+\b(?:js|python|docker|react|node|aws|kubernetes|django|flask|sql)[^)]+)\)', txt, flags=re.I)
            cand = par.group(1) if par else ""
        items = [p.strip() for p in re.split(r'[,\|/;&]+', cand) if p.strip()]
        techs = []
        for it in items:
            techs += _lines_to_skills(it, strict=True)
        return techs

    def _is_project_sentence(sent):
        if not sent or len(sent.strip()) < 20:
            return False
        low = sent.lower()
        project_cues = ['project', 'prototype', 'launched', 'ship', 'developed', 'built', 'designed', 'implemented', 'led', 'created']
        if any(re.search(r'\b' + re.escape(c) + r'\b', low) for c in project_cues):
            return True
        if re.search(r'(?i)tech(?: ?stack)?[:\-\s]|\breact\b|\bnode\b|\bpython\b|\bdjango\b', low):
            return True
        if re.search(r'\b(%|improv|increase|reduce|users|customers|million|improved by|reduced)\b', low):
            return True
        return False

    def _expand_to_paragraph(sentence, full_text):
        if not sentence or not full_text:
            return sentence or ""
        normalized = full_text.replace('\r\n', '\n')
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', normalized) if p.strip()]
        for p in paragraphs:
            if sentence.strip() in p: return p
        sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', normalized) if s.strip()]
        idx = None
        for i, s in enumerate(sents):
            if sentence.strip() in s: idx = i; break
        if idx is None:
            start = " ".join(sentence.strip().split()[:6]).lower()
            for i, s in enumerate(sents):
                if start and s.lower().startswith(start): idx = i; break
        if idx is None: return sentence
        start_i = max(0, idx - 2); end_i = min(len(sents), idx + 3)
        expanded = " ".join(sents[start_i:end_i]); return expanded

    projects = []
    for h, block in sections.items():
        if re.search(r'(?i)project', h):
            for pr in re.split(r'\n{1,3}', block):
                pr = pr.strip()
                if not pr:
                    continue
                techs = _extract_tech_list_from_text(pr) if ' _extract_tech_list_from_text' in globals() else []
                projects.append({
                    "name": (pr.split('\n')[0] or "")[:200],
                    "description": pr,
                    "technologies": techs,
                    "source": h
                })

    for exp in parsed.get('experience', []):
        if not isinstance(exp, dict):
            continue
        desc_full = (exp.get('description') or "").strip()
        if not desc_full:
            continue
        cand_sents = [s.strip() for s in re.split(r'[.\n;•\-]{1,}\s*', desc_full) if s.strip()]
        for sent in cand_sents:
            if _is_project_sentence(sent):
                expanded = _expand_to_paragraph(sent, desc_full)
                techs = _extract_tech_list_from_text(expanded)
                heading_match = re.split(r'[:\-\|]\s*', expanded.strip(), 1)[0]
                if 6 <= len(heading_match.split()) <= 12:
                    name = heading_match.strip()[:200]
                else:
                    name = expanded.strip().split('\n', 1)[0][:200]
                dup = False
                for p in projects:
                    if expanded.strip()[:120].lower() in p.get('description','').lower() or p.get('description','').strip()[:120].lower() in expanded.strip().lower():
                        dup = True
                        p_techs = set([t.lower() for t in p.get('technologies', [])])
                        for t in techs:
                            if t and t.lower() not in p_techs:
                                p['technologies'].append(t)
                        break
                if not dup:
                    projects.append({
                        "name": name,
                        "description": expanded,
                        "technologies": techs,
                        "source": exp.get('company') or exp.get('role') or 'experience'
                    })

    tech_stack_candidates = []
    for m in re.finditer(r'(?i)(?:tech(?: ?stack)?|technologies|tools)[:\s]*(.+)', text):
        cap = m.group(1).strip()
        techs = _extract_tech_list_from_text(cap) if ' _extract_tech_list_from_text' in globals() else []
        if techs:
            tech_stack_candidates.extend([t for t in techs if t and t.strip()])

    if tech_stack_candidates:
        normalized_from_stack = auto_normalize_skills(list(dict.fromkeys(tech_stack_candidates)))
        existing_techs_lower = {t.lower() for t in parsed.get('technical_skills', [])}
        new_techs = [t for t in normalized_from_stack if t and t.lower() not in existing_techs_lower]
        parsed['technical_skills'] = (parsed.get('technical_skills') or []) + new_techs

    projects = [p for p in projects if (p.get('name') or "").strip().lower() not in ("tech stack", "tech-stack", "techstack")]

    seen_map = {}
    final_projects = []
    for p in projects:
        key = (p.get('name','').strip().lower() + '|' + (p.get('description','')[:120] or '').strip().lower())
        if key in seen_map:
            idx = seen_map[key]
            existing = final_projects[idx]
            if len(p.get('description','') or "") > len(existing.get('description','') or ""):
                final_projects[idx] = p
        else:
            seen_map[key] = len(final_projects)
            techs = []
            tseen = set()
            for t in (p.get('technologies') or []):
                tn = t.strip()
                if not tn or len(tn.split()) > 6: continue
                tl = tn.lower()
                if tl in tseen: continue
                tseen.add(tl); techs.append(tn)
            p['technologies'] = techs
            final_projects.append(p)

        def _looks_like_real_project(p):
            if not p or not isinstance(p, dict):
                return False
            name = (p.get('name') or "").strip()
            desc = (p.get('description') or "").strip()
            techs = p.get('technologies') or []
            if name.lower() in ("tech stack", "tech-stack", "techstack"):
                return False
            # Accept projects if:
            #  - description is reasonably long, OR
            #  - technologies present, OR
            #  - contains project cues (developed/implemented/designed) even if short
            if len(desc) >= 50:
                return True
            if techs and len(techs) >= 1:
                return True
            if re.search(r'\b(project|prototype|launched|developed|built|designed|implemented|created|led)\b', desc, flags=re.I):
                return True
            # small-name multiword projects accepted
            if name and len(name.split()) >= 2 and len(desc) >= 30:
                return True
            return False


    filtered_projects = [p for p in projects if _looks_like_real_project(p)]
    if not filtered_projects:
        filtered_projects = [p for p in projects if (p.get('name') or "").strip().lower() not in ("tech stack","tech-stack","techstack")][:3]
    else:
        filtered_projects = filtered_projects[:4]

        # --- fallback: if no projects found, scan experience and summary for candidate project sentences ---
    if not filtered_projects:
        fallback_projects = []
        # 1) from experience descriptions
        for e in parsed.get('experience', []):
            try:
                desc = (e.get('description') or "")
                if not desc: continue
                # find short sentences containing key cues
                for sent in re.split(r'(?<=[\.\n])\s+', desc):
                    if len(sent.strip()) < 10: continue
                    if re.search(r'\b(designed|developed|implemented|launched|prototype|project|built|created)\b', sent, flags=re.I):
                        name = sent.strip().split('.')[0][:120]
                        fallback_projects.append({
                            "name": name,
                            "description": _expand_to_paragraph(sent, desc),
                            "technologies": _extract_tech_list_from_text(desc),
                            "source": e.get('company') or e.get('role') or 'experience'
                        })
            except Exception:
                continue
        # 2) from summary
        summ = parsed.get('summary','') or ""
        for sent in re.split(r'(?<=[\.\n])\s+', summ):
            if len(sent.strip()) < 12: continue
            if re.search(r'\b(project|prototype|designed|developed|launched)\b', sent, flags=re.I):
                fallback_projects.append({"name": sent.strip()[:120], "description": sent.strip(), "technologies": _extract_tech_list_from_text(summ), "source": "summary"})
        # attach up to 3 fallback projects
        if fallback_projects:
            # dedupe and keep the longest descriptions
            seenp = set(); fp = []
            for p in fallback_projects:
                key = (p.get('name','') or '').strip().lower()
                if key in seenp: continue
                seenp.add(key); fp.append(p)
                if len(fp) >= 3: break
            filtered_projects = fp

    parsed['projects'] = filtered_projects

    edu = []
    for h, block in sections.items():
        if re.search(r'(?i)education', h):
            for line in block.splitlines():
                line = line.strip()
                if not line: continue
                edu.append(line)
    parsed['education'] = edu

    certs = []
    for h, block in sections.items():
        if re.search(r'(?i)certif', h):
            certs += _lines_to_skills(block, strict=False)
    parsed['certifications'] = certs

    if 'summary' in sections:
        parsed['summary'] = sections['summary'][:800]
    else:
        first_para = ""
        for p in text.split('\n\n'):
            if p.strip():
                first_para = p.strip(); break
        parsed['summary'] = first_para[:800]

    parsed['technical_skills'] = parsed.get('technical_skills') or []
    parsed['soft_skills'] = parsed.get('soft_skills') or []
    parsed['projects'] = parsed.get('projects') or []
    parsed['experience'] = parsed.get('experience') or []
    parsed['education'] = parsed.get('education') or []
    parsed['certifications'] = parsed.get('certifications') or []

    return parsed

# parse_and_store_resume / parse_and_store_job remain functionally same as before
def parse_and_store_resume(resume_obj):
    text = ""
    try:
        if getattr(resume_obj, "file", None):
            text = simple_text_from_file(resume_obj.file) or ""
    except Exception:
        text = resume_obj.raw_text or ""
        log.debug("file extraction error", exc_info=True)

    text = text or (resume_obj.raw_text or "")
    text = re.sub(r'\r\n', '\n', text or "")
    text = text.strip()
    resume_obj.raw_text = text

    if not text:
        parsed = {"contact": {}, "technical_skills": [], "soft_skills": [], "experience": [], "projects": [], "education": [], "certifications": [], "summary": ""}
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
    text = (job_obj.raw_text or "").strip()
    try:
        if getattr(job_obj, "file", None) and not text:
            text = simple_text_from_file(job_obj.file) or ""
    except Exception:
        text = job_obj.raw_text or ""
    text = re.sub(r'\r\n', '\n', text or "").strip()
    job_obj.raw_text = text

    if not text:
        parsed = {"title": job_obj.title or "Job", "critical_skills": [], "beneficial_skills": [], "experience_level": "Unknown", "years_required": None, "responsibilities": [], "domain_keywords": []}
        job_obj.parsed = parsed
        job_obj.save()
        return parsed

    parsed = parse_job_description_text(text) or {}
    parsed_title = parsed.get("title") or (text.splitlines()[0].strip() if text.splitlines() else job_obj.title or "Job")
    job_obj.parsed = parsed; job_obj.title = parsed_title; job_obj.save()
    return parsed

# --- Helpers for improved JD parsing: place these above parse_job_description_text in utils.py ---

def _split_or_group_degree_phrases(text_block):
    """
    Convert "A, B, or C" and "A or B" lists into balanced multi-word candidates.
    Preserve known multi-word noun phrases such as 'industrial design' or 'product design'
    and return a list of candidate phrase strings.
    """
    out = []
    if not text_block:
        return out

    # Split by semicolons or newlines into sentence-sized chunks
    parts = re.split(r'[\r\n;]+', text_block)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If this part looks like a list with commas and an 'or'
        if ',' in part and re.search(r'\bor\b', part, flags=re.I):
            items = [p.strip() for p in re.split(r',\s*', part)]
            # Handle trailing "or X" in the last item like "A, B, or C"
            last = items[-1]
            if re.search(r'\bor\b', last, flags=re.I):
                last_parts = re.split(r'\bor\b', last, flags=re.I)
                items[-1] = last_parts[0].strip()
                items.append(last_parts[-1].strip())
            for it in items:
                if it:
                    out.append(it)
        # If it contains 'or' without commas (e.g., "X or Y")
        elif re.search(r'\b or \b', part, flags=re.I) and ',' not in part:
            for it in re.split(r'\bor\b', part, flags=re.I):
                it = it.strip()
                if it:
                    out.append(it)
        else:
            out.append(part)

    # Post-process: protect obvious short connector fragments
    cleaned = []
    for cand in out:
        # remove leading/trailing 'or a related field' style fragments
        cand = re.sub(r'(?i)^\s*(or\s+)?a\s+related\s+field[,:]?\s*$', '', cand).strip()
        if not cand:
            continue
        cleaned.append(cand)
    return cleaned


def _extract_candidates_from_block(block_text):
    """
    Produce a list of candidate requirement strings from a JD block.
    Uses bullets, grouped lists, and parenthetical expansion while preferring multi-word phrases.
    """
    if not block_text:
        return []

    cands = []

    # 1) bullet lines first
    for ln in re.split(r'[\r\n]+', block_text):
        l = ln.strip()
        if not l:
            continue
        # remove bullet markers
        l = re.sub(r'^\s*[-•\*\u2022]\s*', '', l).strip()
        if l:
            cands.append(l)

    # 2) group comma/or lists and extract items
    grouped = []
    for part in cands[:]:
        if ',' in part or re.search(r'\bor\b', part, flags=re.I):
            grouped.extend(_split_or_group_degree_phrases(part))
        else:
            grouped.append(part)

    # If no bullets found, try splitting the block itself
    if not grouped:
        # split by sentences but keep parenthetical groups
        for p in re.split(r'(?<=[.!?])\s+', block_text):
            p = p.strip()
            if p:
                if ',' in p or re.search(r'\bor\b', p, flags=re.I):
                    grouped.extend(_split_or_group_degree_phrases(p))
                else:
                    grouped.append(p)

    # 3) Expand parenthetical fragments into their contents as separate candidates (but keep the main phrase)
    final = []
    for g in grouped:
        final.append(g)
        for par in re.findall(r'\(([^)]+)\)', g):
            par = par.strip()
            if par and par not in final:
                final.append(par)

    # 4) Deduplicate preserving order
    seen = set(); out = []
    for f in final:
        key = f.lower().strip()
        if key in seen: continue
        seen.add(key)
        out.append(f.strip())
    return out


# --- Replacement parse_job_description_text function ---

# ------------------ helpers for improved JD parsing ------------------

def _split_or_group_degree_phrases(text_block):
    """
    Convert 'A, B, or C' and 'A or B' lists into balanced multi-word candidates.
    Preserve known multi-word noun phrases such as 'industrial design' or 'product design'
    and return a list of candidate phrase strings.
    """
    out = []
    if not text_block:
        return out

    # Split by semicolons or newlines into sentence-sized chunks
    parts = re.split(r'[\r\n;]+', text_block)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # If this part looks like a list with commas and an 'or'
        if ',' in part and re.search(r'\bor\b', part, flags=re.I):
            items = [p.strip() for p in re.split(r',\s*', part)]
            # Handle trailing "or X" in the last item like "A, B, or C"
            last = items[-1]
            if re.search(r'\bor\b', last, flags=re.I):
                # split last into its components
                tail = re.split(r'\bor\b', last, flags=re.I)
                # replace last with separated items
                items = items[:-1] + [t.strip() for t in tail if t.strip()]
            for it in items:
                if it:
                    out.append(it)
        else:
            # Not an explicit list; try to preserve multiword noun phrases by returning the part
            out.append(part)
    # final cleaning
    cleaned = []
    for o in out:
        o = re.sub(r'^\s*[-•\*\u2022]\s*', '', o).strip()
        if o and o not in cleaned:
            cleaned.append(o)
    return cleaned


def _extract_candidates_from_block(block_text):
    """
    Try to extract multi-word candidate phrases from a 'requirements' block.
    Strategy: find comma separated items, parenthetical lists, and phrases linked by 'or'.
    This returns a list of candidate string phrases (keeps multiword tokens).
    """
    if not block_text:
        return []

    candidates = []
    # Look for obvious parenthetical enumerations: (A, B or C)
    for par in re.findall(r'\(([^)]+)\)', block_text):
        candidates += _split_or_group_degree_phrases(par)

    # Break large block into lines/bullets; each line is a candidate
    lines = [ln.strip() for ln in re.split(r'[\r\n]+', block_text) if ln.strip()]
    for ln in lines:
        # strip bullets and leading markers
        ln_clean = re.sub(r'^\s*[-•\*\u2022]\s*', '', ln).strip()
        # if line looks like "A, B or C" group them
        if ',' in ln_clean and re.search(r'\bor\b', ln_clean, flags=re.I):
            candidates += _split_or_group_degree_phrases(ln_clean)
        else:
            # preserve as-is (multiword)
            candidates.append(ln_clean)

    # split commas but prefer to keep multiword tokens together if they contain known words
    out = []
    for c in candidates:
        # if the chunk contains common separators but also pattern 'X Y' keep it
        # split only if splitting would create single-word tokens
        parts = [p.strip() for p in re.split(r',\s*', c) if p.strip()]
        if len(parts) > 1 and all(len(p.split()) == 1 for p in parts):
            out += parts
        else:
            out.append(c)
    # dedupe and return
    seen = set(); final = []
    for s in out:
        key = s.lower().strip()
        if key and key not in seen:
            final.append(s.strip()); seen.add(key)
    return final


def _bullets_from_block(block_text):
    """Return cleaned bullet lines from a block of text."""
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


def parse_job_description_text(text: str) -> Dict[str, Any]:
    """
    Parse job description text into a normalized dict structure expected by matching/gap code.
    Always returns a dict with keys: title, critical_skills (list), beneficial_skills (list),
    responsibilities (list), experience (str), raw (original).
    This is intentionally conservative and won't throw on unexpected text.
    """
    parsed = {"title": "", "critical_skills": [], "beneficial_skills": [], "responsibilities": [], "experience": "", "raw": text}
    text = (text or "").strip()
    if not text:
        return parsed

    # split into sections
    sections = _split_sections_by_headings(text)

    # title: try first line or section 'title'
    first_line = text.splitlines()[0].strip()
    parsed['title'] = sections.get('title', first_line) or first_line or "Job"

    # responsibilities: if there's a Responsibilities section, split lines
    resp_candidates = []
    for k in sections:
        if re.search(r"responsib", k, flags=re.I):
            resp_candidates.append(sections[k])
    if resp_candidates:
        resp_text = "\n\n".join(resp_candidates)
        parsed['responsibilities'] = [r.strip(" •-") for r in re.split(r'[\n\r]+', resp_text) if r.strip()]
    else:
        # as fallback, search for bullet-like lines under the body
        body = sections.get('body', text)
        bullets = [l.strip(" •-") for l in re.split(r'[\n\r]+', body) if len(l.strip()) > 20 and (l.strip().startswith("-") or l.strip().startswith("•") or ':' in l)]
        parsed['responsibilities'] = bullets[:10]

    # find explicit skill lines using simple heuristics
    crit = []
    ben = []
    # look for lines that contain "required", "must have", "required skills"
    for k, v in sections.items():
        if re.search(r"required|must have|must-have|essential", k, flags=re.I) or re.search(r"required|must have|must-have|essential", v, flags=re.I):
            crit += re.findall(r'([A-Za-z0-9\-\+\.& ]{2,80})', v)
        if re.search(r"preferred|nice to have|beneficial|optional", k, flags=re.I) or re.search(r"preferred|nice to have|beneficial|optional", v, flags=re.I):
            ben += re.findall(r'([A-Za-z0-9\-\+\.& ]{2,80})', v)

    # fallback: look for skill words in whole text near 'skills' or 'responsibilities'
    if not crit:
        m = re.search(r'(skills[:\s]*)(.+?)(\n\n|$)', text, flags=re.I | re.S)
        if m:
            crit += re.split(r'[,\n;•\u2022]+', m.group(2))
    if not ben:
        m = re.search(r'(preferred[:\s]*)(.+?)(\n\n|$)', text, flags=re.I | re.S)
        if m:
            ben += re.split(r'[,\n;•\u2022]+', m.group(2))

    def clean_skill_list(lst):
        out = []
        for s in lst:
            if not s:
                continue
            s2 = s.strip().strip('•-–—. ')
            # drop very long garbage
            if len(s2) > 120:
                continue
            # drop purely sentence-like lines
            if len(s2.split()) > 10:
                continue
            out.append(re.sub(r'\s{2,}', ' ', s2))
        # dedupe preserving order
        seen = set(); final = []
        for it in out:
            k = it.lower().strip()
            if k and k not in seen:
                final.append(it.strip()); seen.add(k)
        return final

    parsed['critical_skills'] = clean_skill_list(crit)[:50]
    parsed['beneficial_skills'] = clean_skill_list(ben)[:50]

    # experience: try find lines like "X years" or an experience/requirements section
    exp = ""
    exp_match = re.search(r'([0-9]+)\+?\s+years', text, flags=re.I)
    if exp_match:
        exp = exp_match.group(0)
    else:
        # look for 'experience' section content short summary
        if 'experience' in sections:
            exp = sections['experience'].strip().splitlines()[0][:200]
    parsed['experience'] = exp

    return parsed




def _strip_think(text: str) -> str:
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def build_gap_analysis_prompt(resume_parsed: dict, job_parsed: dict) -> str:
    """
    Builds a strict prompt for LLM-only skill gap analysis.
    """

    job_title = job_parsed.get("title", "the job role")
    jd_skills = job_parsed.get("critical_skills", []) + job_parsed.get("beneficial_skills", [])
    resume_skills = (
        (resume_parsed.get("technical_skills") or []) +
        (resume_parsed.get("soft_skills") or [])
    )

    resume_projects = resume_parsed.get("projects") or []
    resume_summary = resume_parsed.get("summary") or ""

    return f"""
You are evaluating a candidate for the role: "{job_title}"

JOB REQUIREMENTS (skills mentioned in the job description):
{jd_skills}

CANDIDATE RESUME SKILLS:
{resume_skills}

CANDIDATE SUMMARY:
{resume_summary}

CANDIDATE PROJECTS:
{resume_projects}

TASK:
1. Identify skill gaps between the resume and the job.
2. Decide the gap_type for each skill:
   - "missing" (not present at all)
   - "weak" (mentioned but not demonstrated clearly)
3. For EACH skill gap, provide 1–2 concise, actionable recommendations
   that help strengthen the job match.
4. Recommendations must be resume-focused (what to add, clarify, or highlight).

OUTPUT FORMAT (STRICT JSON ONLY):
[
  {{
    "skill": "<skill name>",
    "gap_type": "missing | weak",
    "recommendations": [
      "<recommendation 1>",
      "<recommendation 2>"
    ]
  }}
]

IMPORTANT RULES:
- Do NOT invent unrelated skills.
- Do NOT include explanations outside JSON.
- Do NOT include <think> tags.
- Keep recommendations concise and job-relevant.
"""

def llm_gap_analysis(resume_parsed: dict, job_parsed: dict, max_items: int = 8):
    """
    LLM-only gap analysis. No KB. No rules. Single batched call.
    """

    prompt = build_gap_analysis_prompt(resume_parsed, job_parsed)

    resp = ai_client.call_mixtral(
        prompt=prompt,
        system=LLM_SYSTEM_NO_THINK,
        temperature=0.2,
        max_tokens=700
    )

    raw_text = _strip_think(resp.get("text", ""))

    try:
        gaps = json.loads(raw_text)
        if not isinstance(gaps, list):
            return []

        # defensive cleanup
        cleaned = []
        for g in gaps:
            if not isinstance(g, dict):
                continue
            skill = g.get("skill")
            gap_type = g.get("gap_type")
            recs = g.get("recommendations")

            if skill and gap_type in ("missing", "weak") and isinstance(recs, list):
                cleaned.append({
                    "skill": skill,
                    "gap_type": gap_type,
                    "recommendations": recs[:2]
                })

        return cleaned[:max_items]

    except Exception:
        # If model violates JSON contract, fail gracefully
        return []
    
def build_query_prompt(question: str, resume_parsed: dict, job_parsed: dict) -> str:
    return f"""
You are answering a user's question about their job match.

JOB TITLE:
{job_parsed.get("title")}

JOB REQUIREMENTS:
{job_parsed.get("critical_skills")}

RESUME SUMMARY:
{resume_parsed.get("summary")}

RESUME SKILLS:
{resume_parsed.get("technical_skills")}

QUESTION:
{question}

RULES:
- Be concise and clear.
- Do NOT reveal reasoning steps.
- Do NOT include <think> tags.
"""


def llm_answer_query(question: str, resume_parsed: dict, job_parsed: dict) -> str:
    resp = ai_client.call_mixtral(
        prompt=build_query_prompt(question, resume_parsed, job_parsed),
        system=LLM_SYSTEM_NO_THINK,
        temperature=0.3,
        max_tokens=300
    )

    return _strip_think(resp.get("text", "")).strip()