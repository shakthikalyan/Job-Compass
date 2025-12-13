# # App/management/commands/seed_data.py
# import os
# import pandas as pd
# from django.core.management.base import BaseCommand
# from django.db import transaction
# from App.models import Domain, Occupation, Skill, OccupationSkillRelation, Technology, OccupationTechnology

# def safe_normalize(s):
#     if s is None:
#         return ""
#     return str(s).strip().lower()

# def normalise_soc(soc):
#     if soc is None:
#         return None
#     # handle NaN from pandas
#     try:
#         if pd.isna(soc):
#             return None
#     except:
#         pass

#     s = str(soc).strip()

#     # remove trailing .0 when Excel converts codes into floats
#     if s.endswith(".0"):
#         s = s[:-2]

#     # remove internal spaces if any exist accidentally
#     s = s.replace(" ", "")

#     return s

# class Command(BaseCommand):
#     help = "Seed DB from O*NET excel/CSV exports. --occupation --skills --knowledge --abilities --tech"

#     def add_arguments(self, parser):
#         parser.add_argument('--occupation', required=True, help='Path to Occupation Data xlsx/csv')
#         parser.add_argument('--skills', required=True, help='Path to Skills xlsx/csv (Element Type: Skill)')
#         parser.add_argument('--knowledge', required=False, help='Path to Knowledge xlsx/csv (Element Type: Knowledge)')
#         parser.add_argument('--abilities', required=False, help='Path to Abilities xlsx/csv (Element Type: Ability)')
#         parser.add_argument('--tech', required=False, help='Path to Technology Skills xlsx/csv')

#     @transaction.atomic
#     def handle(self, *args, **options):
#         if Occupation.objects.exists() or Skill.objects.exists():
#             self.stdout.write(self.style.WARNING(
#                 "Database already seeded. Aborting to prevent duplicate processing."
#             ))
#             return
    
#         occ_path = options['occupation']
#         skills_path = options['skills']
#         knowledge_path = options.get('knowledge')
#         abilities_path = options.get('abilities')
#         tech_path = options.get('tech')

#         unmatched_socs = set()

#         # --- Occupations ---
#         self.stdout.write("Loading occupations...")
#         occ_df = pd.read_excel(occ_path) if occ_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(occ_path)
#         occ_count = 0
#         for _, row in occ_df.iterrows():
#             soc_raw = row.get('O*NET-SOC Code') or row.get('ONET-SOC Code') or row.get('SOC')
#             soc = normalise_soc(soc_raw)
#             title = row.get('Title') or row.get('Occupation Title') or row.get('Occupation')
#             desc = row.get('Description') or row.get('Occupation Description') or ''
#             domain_name = row.get('Job Zone') or row.get('Domain') or None

#             if not title:
#                 continue

#             domain = None
#             if domain_name and str(domain_name).strip():
#                 domain, _ = Domain.objects.get_or_create(name=str(domain_name).strip(), defaults={'description': ''})

#             occ, created = Occupation.objects.get_or_create(
#                 name=str(title).strip(),
#                 defaults={
#                     'external_id': soc if soc else None,
#                     'description': str(desc).strip() if desc else None,
#                     'domain': domain,
#                     'source': 'onet'
#                 }
#             )
#             if soc and not occ.external_id:
#                 occ.external_id = soc
#                 occ.save()
#             occ_count += 1
#         self.stdout.write(f"Imported/updated {occ_count} occupations.")

#         # --- Element processing (Skills/Knowledge/Abilities) ---
#         def process_element_file(path, default_element_type=None):
#             if not path:
#                 return 0, 0
#             df = pd.read_excel(path) if path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(path)
#             created_skills = 0
#             created_relations = 0
#             for _, row in df.iterrows():
#                 soc_raw = row.get('O*NET-SOC Code') or row.get('ONET-SOC Code') or row.get('SOC')
#                 soc = normalise_soc(soc_raw)
#                 elem_name = row.get('Element Name') or row.get('Element') or row.get('Data Element') or row.get('Data Element Name')
#                 elem_type = row.get('Element Type') or row.get('Element Group') or default_element_type or 'Skill'
#                 data_value = row.get('Data Value') or row.get('Importance') or 0.0
#                 level = row.get('Level') or row.get('Data Value (Level)') or 0.0
#                 element_id = row.get('Element ID') or None

#                 if not elem_name:
#                     continue

#                 # robust occupation lookup
#                 occupation = None
#                 if soc:
#                     occupation = Occupation.objects.filter(external_id__iexact=soc).first()
#                     if not occupation:
#                         # try without dashes/spaces
#                         soc_alt = soc.replace('-', '').replace(' ', '')
#                         occupation = Occupation.objects.filter(external_id__iregex=soc_alt).first()
#                 if not occupation:
#                     # fallback: sometimes the SOC column contains the occupation title in some exports
#                     if soc_raw and isinstance(soc_raw, str) and soc_raw.strip():
#                         occupation = Occupation.objects.filter(name__iexact=soc_raw.strip()).first()
#                 if not occupation:
#                     # if still missing, skip but record it for inspection
#                     unmatched_socs.add(str(soc_raw))
#                     continue

#                 skill_name = str(elem_name).strip()
#                 normalized = safe_normalize(skill_name)
#                 # deduplicate using normalized name to avoid near-duplicates
#                 skill = Skill.objects.filter(normalized=normalized).first()
#                 if not skill:
#                     skill, skill_created = Skill.objects.get_or_create(
#                         name=skill_name,
#                         defaults={'normalized': normalized, 'element_type': elem_type, 'external_id': element_id, 'source': 'onet'}
#                     )
#                     if skill_created:
#                         created_skills += 1

#                 try:
#                     importance = float(data_value)
#                 except Exception:
#                     importance = 0.0
#                 try:
#                     lvl = float(level)
#                 except Exception:
#                     lvl = 0.0

#                 # relation classification thresholds (tweakable)
#                 if importance >= 75:
#                     rel = 'essential'
#                 elif importance >= 40:
#                     rel = 'important'
#                 elif importance > 0:
#                     rel = 'optional'
#                 else:
#                     rel = 'other'

#                 # SINGLE authoritative relation per (occupation, skill)
#                 rel_obj, created_flag = OccupationSkillRelation.objects.get_or_create(
#                     occupation=occupation,
#                     skill=skill,
#                     defaults={
#                         'element_type': elem_type,
#                         'relation_type': rel,
#                         'importance': importance,
#                         'level': lvl,
#                         'provenance': 'onet'
#                     }
#                 )
#                 if not created_flag:
#                     updated = False
#                     # prefer higher importance and higher level when aggregating
#                     if importance > (rel_obj.importance or 0.0):
#                         rel_obj.importance = importance
#                         rel_obj.relation_type = rel
#                         updated = True
#                     if lvl > (rel_obj.level or 0.0):
#                         rel_obj.level = lvl
#                         updated = True
#                     if elem_type and rel_obj.element_type != elem_type:
#                         rel_obj.element_type = elem_type
#                         updated = True
#                     if updated:
#                         rel_obj.provenance = 'onet'
#                         rel_obj.save()
#                 else:
#                     created_relations += 1

#             return created_skills, created_relations

#         self.stdout.write("Processing Skills...")
#         s_count, sr_count = process_element_file(skills_path, default_element_type='Skill')
#         self.stdout.write(f"Skills: created {s_count} skills, {sr_count} occupation-skill relations.")

#         if knowledge_path:
#             self.stdout.write("Processing Knowledge...")
#             k_count, kr_count = process_element_file(knowledge_path, default_element_type='Knowledge')
#             self.stdout.write(f"Knowledge: created {k_count} skills, {kr_count} relations.")

#         if abilities_path:
#             self.stdout.write("Processing Abilities...")
#             a_count, ar_count = process_element_file(abilities_path, default_element_type='Ability')
#             self.stdout.write(f"Abilities: created {a_count} skills, {ar_count} relations.")

#         # --- Technology Skills ---
#         if tech_path:
#             self.stdout.write("Processing Technology Skills...")
#             df = pd.read_excel(tech_path) if tech_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(tech_path)
#             tcreated = 0
#             trel = 0
#             for _, row in df.iterrows():
#                 soc_raw = row.get('O*NET-SOC Code') or row.get('ONET-SOC Code') or row.get('SOC')
#                 soc = normalise_soc(soc_raw)
#                 tech_name = row.get('Technology Name') or row.get('Element Name') or row.get('Technology')
#                 data_value = row.get('Data Value') or row.get('Importance') or 0.0
#                 if not tech_name:
#                     continue

#                 occupation = None
#                 if soc:
#                     occupation = Occupation.objects.filter(external_id__iexact=soc).first()
#                     if not occupation:
#                         occupation = Occupation.objects.filter(external_id__iregex=soc.replace('-', '')).first()
#                 if not occupation:
#                     unmatched_socs.add(str(soc_raw))
#                     continue

#                 tech, created = Technology.objects.get_or_create(
#                     name=str(tech_name).strip(),
#                     defaults={'normalized': safe_normalize(tech_name)}
#                 )
#                 if created:
#                     tcreated += 1

#                 try:
#                     importance = float(data_value)
#                 except Exception:
#                     importance = 0.0

#                 # single relation per occupation-technology
#                 rel_obj, rel_created = OccupationTechnology.objects.update_or_create(
#                     occupation=occupation,
#                     technology=tech,
#                     defaults={'importance': importance, 'provenance': 'onet'}
#                 )
#                 trel += 1

#             self.stdout.write(f"Imported {tcreated} technologies and {trel} occupation-technology relations.")

#         # Save unmatched SOCs to file for manual review
#         if unmatched_socs:
#             path = os.path.join(os.getcwd(), "unmatched_soc_codes.txt")
#             with open(path, "w", encoding="utf-8") as fh:
#                 for s in sorted(unmatched_socs):
#                     fh.write(f"{s}\n")
#             self.stdout.write(self.style.WARNING(f"Found {len(unmatched_socs)} unmatched SOC values. Written to {path}"))

#         self.stdout.write(self.style.SUCCESS("Seeding finished."))


# App/management/commands/seed_data.py
import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.db import transaction
from App.models import Domain, Occupation, Skill, OccupationSkillRelation, Technology, OccupationTechnology

def safe_normalize(s):
    if s is None:
        return ""
    return str(s).strip().lower()

def normalise_soc(soc):
    if soc is None:
        return None
    # handle NaN from pandas
    try:
        if pd.isna(soc):
            return None
    except:
        pass

    s = str(soc).strip()

    # remove trailing .0 when Excel converts codes into floats
    if s.endswith(".0"):
        s = s[:-2]

    # remove internal spaces if any exist accidentally
    s = s.replace(" ", "")

    return s

class Command(BaseCommand):
    help = "Seed DB from O*NET excel/CSV exports. --occupation --skills --knowledge --abilities --tech"

    def add_arguments(self, parser):
        parser.add_argument('--occupation', required=True, help='Path to Occupation Data xlsx/csv')
        parser.add_argument('--skills', required=True, help='Path to Skills xlsx/csv (Element Type: Skill)')
        parser.add_argument('--knowledge', required=False, help='Path to Knowledge xlsx/csv (Element Type: Knowledge)')
        parser.add_argument('--abilities', required=False, help='Path to Abilities xlsx/csv (Element Type: Ability)')
        parser.add_argument('--tech', required=False, help='Path to Technology Skills xlsx/csv')
        parser.add_argument('--force', action='store_true', help='Force seeding even if DB already has Occupation/Skill rows')

    @transaction.atomic
    def handle(self, *args, **options):
        # if DB already seeded, block unless --force provided
        if (Occupation.objects.exists() or Skill.objects.exists()) and not options.get('force', False):
            self.stdout.write(self.style.WARNING(
                "Database already seeded. Aborting to prevent duplicate processing. "
                "Use --force to override and run again."
            ))
            return

        occ_path = options['occupation']
        skills_path = options['skills']
        knowledge_path = options.get('knowledge')
        abilities_path = options.get('abilities')
        tech_path = options.get('tech')

        unmatched_socs = set()

        # --- Occupations ---
        self.stdout.write("Loading occupations...")
        occ_df = pd.read_excel(occ_path) if occ_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(occ_path)

        # debug: show candidate columns for domain detection
        occ_cols = occ_df.columns.tolist()
        self.stdout.write(self.style.NOTICE(f"Occupation file columns detected: {occ_cols}"))

        occ_count = 0
        # candidate domain columns - broadened to handle common variants
        possible_domain_cols = ['Job Zone', 'JobZone', 'Job zone', 'Domain', 'Category', 'Job_Zone', 'Job Zone (O*NET)']

        for _, row in occ_df.iterrows():
            soc_raw = row.get('O*NET-SOC Code') or row.get('ONET-SOC Code') or row.get('SOC')
            soc = normalise_soc(soc_raw)
            title = row.get('Title') or row.get('Occupation Title') or row.get('Occupation')
            desc = row.get('Description') or row.get('Occupation Description') or ''

            # robust domain lookup - check multiple column names
            domain_name = None
            for col in possible_domain_cols:
                if col in row and row.get(col) and str(row.get(col)).strip():
                    domain_name = row.get(col)
                    break

            if not title:
                continue

            domain = None
            if domain_name and str(domain_name).strip():
                # coerce numeric Job Zone to a readable domain label if it's purely numeric
                dn = str(domain_name).strip()
                if dn.isdigit():
                    dn = f"Job Zone {dn}"
                domain, _ = Domain.objects.get_or_create(name=dn, defaults={'description': ''})

            occ, created = Occupation.objects.get_or_create(
                name=str(title).strip(),
                defaults={
                    'external_id': soc if soc else None,
                    'description': str(desc).strip() if desc else None,
                    'domain': domain,
                    'source': 'onet'
                }
            )
            if soc and not occ.external_id:
                occ.external_id = soc
                occ.save()
            occ_count += 1
        self.stdout.write(f"Imported/updated {occ_count} occupations.")

        # --- Element processing (Skills/Knowledge/Abilities) ---
        def process_element_file(path, default_element_type=None):
            if not path:
                return 0, 0
            df = pd.read_excel(path) if path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(path)
            # debug columns for this elements file
            self.stdout.write(self.style.NOTICE(f"Processing elements file {path}. Columns: {df.columns.tolist()}"))

            created_skills = 0
            created_relations = 0
            for _, row in df.iterrows():
                soc_raw = row.get('O*NET-SOC Code') or row.get('ONET-SOC Code') or row.get('SOC')
                soc = normalise_soc(soc_raw)
                # broaden element name detection
                elem_name = (row.get('Element Name') or row.get('Element') or row.get('Data Element') or
                             row.get('Data Element Name') or row.get('Title') or row.get('Element Name...') or None)
                elem_type = row.get('Element Type') or row.get('Element Group') or default_element_type or 'Skill'
                data_value = row.get('Data Value') or row.get('Importance') or 0.0
                level = row.get('Level') or row.get('Data Value (Level)') or 0.0
                element_id = row.get('Element ID') or None

                if not elem_name:
                    continue

                # robust occupation lookup
                occupation = None
                if soc:
                    occupation = Occupation.objects.filter(external_id__iexact=soc).first()
                    if not occupation:
                        # try without dashes/spaces
                        soc_alt = soc.replace('-', '').replace(' ', '')
                        occupation = Occupation.objects.filter(external_id__iregex=soc_alt).first()
                if not occupation:
                    # fallback: sometimes the SOC column contains the occupation title in some exports
                    if soc_raw and isinstance(soc_raw, str) and soc_raw.strip():
                        occupation = Occupation.objects.filter(name__iexact=soc_raw.strip()).first()
                if not occupation:
                    # if still missing, skip but record it for inspection
                    unmatched_socs.add(str(soc_raw))
                    continue

                skill_name = str(elem_name).strip()
                normalized = safe_normalize(skill_name)
                # deduplicate using normalized name to avoid near-duplicates
                skill = Skill.objects.filter(normalized=normalized).first()
                if not skill:
                    skill, skill_created = Skill.objects.get_or_create(
                        name=skill_name,
                        defaults={'normalized': normalized, 'element_type': elem_type, 'external_id': element_id, 'source': 'onet'}
                    )
                    if skill_created:
                        created_skills += 1

                try:
                    importance = float(data_value)
                except Exception:
                    importance = 0.0
                try:
                    lvl = float(level)
                except Exception:
                    lvl = 0.0

                # relation classification thresholds (tweakable)
                if importance >= 75:
                    rel = 'essential'
                elif importance >= 40:
                    rel = 'important'
                elif importance > 0:
                    rel = 'optional'
                else:
                    rel = 'other'

                # SINGLE authoritative relation per (occupation, skill)
                rel_obj, created_flag = OccupationSkillRelation.objects.get_or_create(
                    occupation=occupation,
                    skill=skill,
                    defaults={
                        'element_type': elem_type,
                        'relation_type': rel,
                        'importance': importance,
                        'level': lvl,
                        'provenance': 'onet'
                    }
                )
                if not created_flag:
                    updated = False
                    # prefer higher importance and higher level when aggregating
                    if importance > (rel_obj.importance or 0.0):
                        rel_obj.importance = importance
                        rel_obj.relation_type = rel
                        updated = True
                    if lvl > (rel_obj.level or 0.0):
                        rel_obj.level = lvl
                        updated = True
                    if elem_type and rel_obj.element_type != elem_type:
                        rel_obj.element_type = elem_type
                        updated = True
                    if updated:
                        rel_obj.provenance = 'onet'
                        rel_obj.save()
                else:
                    created_relations += 1

            return created_skills, created_relations

        self.stdout.write("Processing Skills...")
        s_count, sr_count = process_element_file(skills_path, default_element_type='Skill')
        self.stdout.write(f"Skills: created {s_count} skills, {sr_count} occupation-skill relations.")

        if knowledge_path:
            self.stdout.write("Processing Knowledge...")
            k_count, kr_count = process_element_file(knowledge_path, default_element_type='Knowledge')
            self.stdout.write(f"Knowledge: created {k_count} skills, {kr_count} relations.")

        if abilities_path:
            self.stdout.write("Processing Abilities...")
            a_count, ar_count = process_element_file(abilities_path, default_element_type='Ability')
            self.stdout.write(f"Abilities: created {a_count} skills, {ar_count} relations.")

        # --- Technology Skills ---
        if tech_path:
            self.stdout.write("Processing Technology Skills...")
            df = pd.read_excel(tech_path) if tech_path.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(tech_path)
            self.stdout.write(self.style.NOTICE(f"Technology file columns detected: {df.columns.tolist()}"))

            tcreated = 0
            trel = 0
            possible_tech_cols = ['Technology Name', 'Element Name', 'Technology', 'Example', 'Commodity Title', 'Title', 'Tool', 'Tech']

            for _, row in df.iterrows():
                soc_raw = row.get('O*NET-SOC Code') or row.get('ONET-SOC Code') or row.get('SOC')
                soc = normalise_soc(soc_raw)

                tech_name = None
                for col in possible_tech_cols:
                    if col in row and row.get(col) and str(row.get(col)).strip():
                        tech_name = row.get(col)
                        break

                # Some files put the specific vendor/tool in 'Example' column (observed)
                # If still None, attempt to parse 'Commodity Title' as fallback
                if not tech_name:
                    # additional check for common fallback columns
                    if 'Commodity Title' in row and row.get('Commodity Title'):
                        tech_name = row.get('Commodity Title')

                data_value = row.get('Data Value') or row.get('Importance') or row.get('Hot Technology') or 0.0
                if not tech_name:
                    continue

                occupation = None
                if soc:
                    occupation = Occupation.objects.filter(external_id__iexact=soc).first()
                    if not occupation:
                        occupation = Occupation.objects.filter(external_id__iregex=soc.replace('-', '')).first()
                if not occupation:
                    unmatched_socs.add(str(soc_raw))
                    continue

                tech, created = Technology.objects.get_or_create(
                    name=str(tech_name).strip(),
                    defaults={'normalized': safe_normalize(tech_name)}
                )
                if created:
                    tcreated += 1

                try:
                    importance = float(data_value) if (isinstance(data_value, (int, float)) or str(data_value).replace('.','',1).isdigit()) else 0.0
                except Exception:
                    importance = 0.0

                # single relation per occupation-technology
                rel_obj, rel_created = OccupationTechnology.objects.update_or_create(
                    occupation=occupation,
                    technology=tech,
                    defaults={'importance': importance, 'provenance': 'onet'}
                )
                trel += 1

            self.stdout.write(f"Imported {tcreated} technologies and {trel} occupation-technology relations.")

        # Save unmatched SOCs to file for manual review
        if unmatched_socs:
            path = os.path.join(os.getcwd(), "unmatched_soc_codes.txt")
            with open(path, "w", encoding="utf-8") as fh:
                for s in sorted(unmatched_socs):
                    fh.write(f"{s}\n")
            self.stdout.write(self.style.WARNING(f"Found {len(unmatched_socs)} unmatched SOC values. Written to {path}"))

        self.stdout.write(self.style.SUCCESS("Seeding finished."))
