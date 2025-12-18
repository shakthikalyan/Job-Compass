import re, logging,pdfplumber,PyPDF2
from typing import Dict

log = logging.getLogger(__name__)
_PARENS_RE = re.compile(r'\((.*?)\)')
_SPLIT_DELIM_RE = re.compile(r'[,/;•\n\|]+')
_SKILL_STOP_PATTERNS = [
    r"currently has", r"in the process of obtaining", r"bachelor", r"master", r"degree",
    r"must obtain", r"work authorization", r"must have", r"minimum qualifications?",
    r"preferred qualifications?", r"responsibilities?", r"summary", r"experience coding",
]
_SKILL_STOP_RE = re.compile("|".join(_SKILL_STOP_PATTERNS), flags=re.I)
_ACRONYM_SET = {"aws", "gcp", "sql", "html", "css", "api", "ml", "nlp", "ci", "cd", "cli", "os", "ios", "android"}
_PRESERVE_PATTERN = re.compile(r'^(c\+\+|c#|\.net|node(\.|js)?|js|javascript|typescript|ts|java|python|go(lang)?|golang|rust|kotlin|swift)$', flags=re.I)
_GENERIC_SKILL_TOKENS = {
    'relevant', 'relevance', 'qualification', 'qualifications', 'practical', 'computer', 'science',
    'experience', 'skills', 'summary', 'responsibilities',
}

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
    r'(?i)skills', r'(?i)technical skills', r'(?i)projects', r'(?i)education',
]

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

def build_resume_prompt(raw_text):
    return {"task": "Parse resume into structured JSON", "input_text": raw_text or ""}

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
        parsed = call_parse(prompt) or {}
    except Exception as e:
        parsed = {"error": f"parser error: {str(e)}"}
        log.exception("Parser failed")

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

def _dedupe_preserve_order(lst):
    seen = set(); out = []
    for x in lst:
        if not x: continue
        k = x.lower().strip()
        if k in seen: continue
        seen.add(k); out.append(x)
    return out

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
            
            # otherwise normalize normally, but avoid single-word fragments from long sentences
            norm = _normalize_token(cleaned)
            if not norm:
                # fallback: if cleaned has >2 words, attempt to extract noun-noun bigrams
                words = [w for w in re.split(r'[\s\-_]+', cleaned) if w and len(w) > 1]
                if len(words) >= 2:
                    # try producing two-word candidates from adjacent words and keep ones matching common phrases
                    for i in range(len(words)-1):
                        cand = f"{words[i]} {words[i+1]}"
                        if cand.lower():
                            nn = _normalize_token(cand)
                            if nn and nn.lower() not in seen:
                                seen.add(nn.lower()); out.append(nn)
                continue
            nk = norm.lower()
            if nk not in seen:
                seen.add(nk); out.append(norm)
    return out

def call_parse(prompt_json):
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

def _split_sections_by_headings(text: str) -> Dict[str, str]:
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
