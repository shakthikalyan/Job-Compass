# App/management/commands/seed_esco.py
import csv
import os
import sys
from django.core.management.base import BaseCommand
from django.db import transaction

from App.models import Skill, Occupation, OccupationSkillRelation
# Use your normalize_token util if you have it; fallback to a simple lowercase strip
try:
    from App.utils import normalize_token
except Exception:
    def normalize_token(s: str) -> str:
        return s.lower().strip()

BATCH_SIZE = 1000

def _open_csv(path):
    return open(path, encoding="utf-8-sig", newline='')

def _get_first_existing(keys, row):
    """Return the first non-empty value from row for keys in order."""
    for k in keys:
        if k in row and row[k] not in (None, ''):
            return row[k]
    return None

class Command(BaseCommand):
    help = "Seed ESCO CSVs: skills, occupations, and occupation-skill relations."

    def add_arguments(self, parser):
        parser.add_argument('--skills', type=str, help='Path to skills_en.csv')
        parser.add_argument('--occupations', type=str, help='Path to occupations_en.csv')
        parser.add_argument('--relations', type=str, help='Path to occupationSkillRelations_en.csv')
        parser.add_argument('--limit', type=int, default=0, help='Optional: limit number of rows per file (0 = no limit)')
        parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size for bulk inserts')

    def handle(self, *args, **options):
        skills_path = options.get('skills')
        occ_path = options.get('occupations')
        rel_path = options.get('relations')
        limit = options.get('limit') or 0
        batch = options.get('batch') or BATCH_SIZE

        if skills_path:
            if not os.path.isfile(skills_path):
                self.stderr.write(self.style.ERROR(f"Skills file not found: {skills_path}"))
            else:
                self.import_skills(skills_path, limit, batch)

        if occ_path:
            if not os.path.isfile(occ_path):
                self.stderr.write(self.style.ERROR(f"Occupations file not found: {occ_path}"))
            else:
                self.import_occupations(occ_path, limit, batch)

        if rel_path:
            if not os.path.isfile(rel_path):
                self.stderr.write(self.style.ERROR(f"Relations file not found: {rel_path}"))
            else:
                self.import_relations(rel_path, limit, batch)

    def import_skills(self, path, limit, batch):
        self.stdout.write(f"Importing skills from {path} ...")
        created = 0
        updated = 0
        rows = 0
        objs_to_create = []

        with _open_csv(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                name = _get_first_existing(['preferredLabel', 'preferred_label', 'name'], row)
                esco_id = _get_first_existing(['uri', 'esco_uri', 'id'], row)

                if not name or not esco_id:
                    continue

                normalized = normalize_token(name)
                skill_qs = Skill.objects.filter(esco_id=esco_id)

                if skill_qs.exists():
                    skill = skill_qs.first()
                    skill.name = name
                    skill.normalized = normalized
                    skill.save()
                    updated += 1
                else:
                    objs_to_create.append(Skill(
                        name=name,
                        normalized=normalized,
                        esco_id=esco_id,
                        source='esco'
                    ))

                    if len(objs_to_create) >= batch:
                        Skill.objects.bulk_create(objs_to_create)
                        created += len(objs_to_create)
                        objs_to_create = []

                if limit and rows >= limit:
                    break

            if objs_to_create:
                Skill.objects.bulk_create(objs_to_create)
                created += len(objs_to_create)

        self.stdout.write(self.style.SUCCESS(f"Skills import done. Created: {created}, Updated: {updated}"))

    def import_occupations(self, path, limit, batch):
        self.stdout.write(f"Importing occupations from {path} ...")
        created = 0
        updated = 0
        rows = 0
        objs_to_create = []

        with _open_csv(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows += 1
                name = _get_first_existing(['preferredLabel', 'preferred_label', 'name'], row)
                esco_id = _get_first_existing(['uri', 'esco_uri', 'id'], row)

                if not name or not esco_id:
                    continue

                occupation_qs = Occupation.objects.filter(esco_id=esco_id)

                if occupation_qs.exists():
                    occupation = occupation_qs.first()
                    occupation.name = name
                    occupation.save()
                    updated += 1
                else:
                    objs_to_create.append(Occupation(
                        name=name,
                        esco_id=esco_id,
                        source='esco'
                    ))

                    if len(objs_to_create) >= batch:
                        Occupation.objects.bulk_create(objs_to_create)
                        created += len(objs_to_create)
                        objs_to_create = []

                if limit and rows >= limit:
                    break

            if objs_to_create:
                Occupation.objects.bulk_create(objs_to_create)
                created += len(objs_to_create)

        self.stdout.write(self.style.SUCCESS(f"Occupations import done. Created: {created}, Updated: {updated}"))

    def import_relations(self, path, limit, batch):
        """
        Robust relations importer:
        - Uses tail (UUID) mapping from ESCO CSVs to canonical label + full URI
        - Tries many DB-match strategies (esco_id exact, esco_id tail, name, normalized, synonyms)
        - If a Skill/Occupation is still not found, creates it (with esco_id and preferred label)
        - Bulk-creates relations with ignore_conflicts (Postgres)
        """
        import csv, os, re
        from django.db import transaction
        from App.models import Occupation, Skill, OccupationSkillRelation

        def tail_of(s):
            if not s:
                return None
            return s.rstrip('/').split('/')[-1].split('#')[-1].split(':')[-1].strip()

        def norm(s):
            try:
                from App.utils import normalize_token as _n
                return _n(s)
            except Exception:
                return s.lower().strip() if isinstance(s, str) else s

        def load_tail_map(csv_path, uri_keys=None, label_keys=None, max_rows=0):
            m = {}
            if not csv_path or not os.path.isfile(csv_path):
                return m
            with open(csv_path, encoding='utf-8-sig', newline='') as fh:
                r = csv.DictReader(fh)
                for i,row in enumerate(r):
                    # find uri
                    uri = None
                    for k in (uri_keys or ['uri','esco_uri','id','@id']):
                        if k in row and row[k]:
                            uri = row[k]; break
                    # find label
                    label = None
                    for k in (label_keys or ['preferredLabel','preferred_label','title','label','name']):
                        if k in row and row[k]:
                            label = row[k]; break
                    if isinstance(label, dict):
                        label = label.get('en') or list(label.values())[0]
                    if not uri or not label:
                        continue
                    t = tail_of(uri)
                    if t:
                        m[t] = {'uri': uri.strip(), 'label': label.strip()}
                    if max_rows and (i+1) >= max_rows:
                        break
            return m

        # Paths - adjust if needed
        skills_csv = os.path.join('dataset','skills_en.csv')
        occ_csv = os.path.join('dataset','occupations_en.csv')

        self.stdout.write("Building tail->label maps from ESCO CSVs...")
        tail2skill = load_tail_map(skills_csv)
        tail2occ = load_tail_map(occ_csv)
        self.stdout.write(f"Loaded skill tails: {len(tail2skill)}, occupation tails: {len(tail2occ)}")

        # Build DB quick lookups
        skill_by_esco = {}
        skill_by_tail = {}
        skill_by_name = {}
        skill_by_norm = {}
        skill_by_syn = {}

        for s in Skill.objects.all():
            if s.esco_id:
                skill_by_esco[s.esco_id.strip()] = s
                t = tail_of(s.esco_id)
                if t:
                    skill_by_tail[t] = s
            if s.name:
                skill_by_name[s.name.strip()] = s
                skill_by_norm[norm(s.name)] = s
            try:
                for syn in s.synonyms or []:
                    if isinstance(syn, str):
                        skill_by_syn[syn.strip()] = s
                        skill_by_norm[norm(syn)] = s
            except Exception:
                pass

        occ_by_esco = {}
        occ_by_tail = {}
        occ_by_name = {}
        occ_by_norm = {}
        for o in Occupation.objects.all():
            if o.esco_id:
                occ_by_esco[o.esco_id.strip()] = o
                t = tail_of(o.esco_id)
                if t:
                    occ_by_tail[t] = o
            if o.name:
                occ_by_name[o.name.strip()] = o
                occ_by_norm[norm(o.name)] = o

        # iterate relations CSV
        rows = 0
        created = 0
        skipped = 0
        to_create = []
        miss_logs = 0
        max_miss_logs = 50

        with open(path, encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            occ_col = next((c for c in cols if 'occupation' in c.lower() and ('uri' in c.lower() or 'id' in c.lower())), cols[0] if cols else None)
            skill_col = next((c for c in cols if 'skill' in c.lower() and ('uri' in c.lower() or 'id' in c.lower())), cols[1] if len(cols) > 1 else None)
            rel_col = next((c for c in cols if any(x in c.lower() for x in ['relationtype','relation','type'])), None)

            self.stdout.write(f"Detected columns -> occ: {occ_col}, skill: {skill_col}, rel: {rel_col}")

            for row in reader:
                rows += 1
                occ_raw = (row.get(occ_col) or '').strip()
                skill_raw = (row.get(skill_col) or '').strip()
                rel_raw = (row.get(rel_col) or '').strip() if rel_col else ''

                if not occ_raw or not skill_raw:
                    skipped += 1
                    continue

                occ_tail = tail_of(occ_raw)
                skill_tail = tail_of(skill_raw)

                # get canonical labels from CSV maps
                occ_info = tail2occ.get(occ_tail) if occ_tail else None
                skill_info = tail2skill.get(skill_tail) if skill_tail else None

                occ_label = occ_info['label'] if occ_info else occ_raw
                occ_full_uri = occ_info['uri'] if occ_info else occ_raw
                skill_label = skill_info['label'] if skill_info else skill_raw
                skill_full_uri = skill_info['uri'] if skill_info else skill_raw

                # try DB lookups in order (esco exact, tail, name, normalized, synonyms)
                occ_obj = occ_by_esco.get(occ_full_uri) or occ_by_tail.get(occ_tail) or occ_by_name.get(occ_label) or occ_by_norm.get(norm(occ_label))
                skill_obj = skill_by_esco.get(skill_full_uri) or skill_by_tail.get(skill_tail) or skill_by_name.get(skill_label) or skill_by_norm.get(norm(skill_label)) or skill_by_syn.get(skill_label) or skill_by_syn.get(norm(skill_label))

                # If missing, create the record with esco_id and name (safe because CSV is canonical)
                if not occ_obj:
                    occ_obj = Occupation.objects.create(name=occ_label, esco_id=occ_full_uri, source='esco')
                    # update caches
                    occ_by_esco[occ_full_uri] = occ_obj
                    if occ_tail:
                        occ_by_tail[occ_tail] = occ_obj
                    occ_by_name[occ_label] = occ_obj
                    occ_by_norm[norm(occ_label)] = occ_obj

                if not skill_obj:
                    skill_obj = Skill.objects.create(name=skill_label, esco_id=skill_full_uri, normalized=norm(skill_label), source='esco')
                    skill_by_esco[skill_full_uri] = skill_obj
                    if skill_tail:
                        skill_by_tail[skill_tail] = skill_obj
                    skill_by_name[skill_label] = skill_obj
                    skill_by_norm[norm(skill_label)] = skill_obj

                # determine relation_type
                rv = rel_raw.lower()
                relation_type = 'other'
                if 'essential' in rv:
                    relation_type = 'essential'
                elif 'optional' in rv or 'nice' in rv:
                    relation_type = 'optional'

                to_create.append(OccupationSkillRelation(occupation=occ_obj, skill=skill_obj, relation_type=relation_type))

                if len(to_create) >= batch:
                    with transaction.atomic():
                        OccupationSkillRelation.objects.bulk_create(to_create, ignore_conflicts=True)
                    created += len(to_create)
                    to_create = []

                if limit and rows >= limit:
                    break

            if to_create:
                with transaction.atomic():
                    OccupationSkillRelation.objects.bulk_create(to_create, ignore_conflicts=True)
                created += len(to_create)

        self.stdout.write(self.style.SUCCESS(f"Relations import done. Processed: {rows}, Created: {created}, Skipped: {skipped}"))

