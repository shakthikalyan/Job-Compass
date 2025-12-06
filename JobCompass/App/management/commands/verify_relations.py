# App/management/commands/verify_esco.py
import csv
import os
from collections import Counter, defaultdict
from django.core.management.base import BaseCommand
from App.models import Skill, Occupation, OccupationSkillRelation
from django.db import connection

def tail_of(s):
    if not s:
        return None
    return s.rstrip('/').split('/')[-1].split('#')[-1].split(':')[-1].strip()

class Command(BaseCommand):
    help = "Verify ESCO import consistency and relations correctness."

    def add_arguments(self, parser):
        parser.add_argument('--relations-csv', type=str, default='dataset/occupationSkillRelations_en.csv', help='Path to relations CSV (for cross-checking tails)')
        parser.add_argument('--sample-size', type=int, default=50, help='How many misses to save as sample')

    def handle(self, *args, **options):
        relations_csv = options['relations_csv']
        sample_size = options['sample_size']

        total_skills = Skill.objects.count()
        skills_with_esco = Skill.objects.exclude(esco_id__in=[None,'']).count()
        total_occ = Occupation.objects.count()
        occ_with_esco = Occupation.objects.exclude(esco_id__in=[None,'']).count()
        total_rel = OccupationSkillRelation.objects.count()

        self.stdout.write("SUMMARY")
        self.stdout.write(f"Skills total: {total_skills}, with esco_id: {skills_with_esco}")
        self.stdout.write(f"Occupations total: {total_occ}, with esco_id: {occ_with_esco}")
        self.stdout.write(f"Relations total: {total_rel}")

        # Referential integrity quick check
        bad_occ = OccupationSkillRelation.objects.filter(occupation__isnull=True).count()
        bad_skill = OccupationSkillRelation.objects.filter(skill__isnull=True).count()
        self.stdout.write(f"Relations with missing occupation: {bad_occ}, missing skill: {bad_skill}")

        # If the relations CSV exists, do tail cross-check: count tails that exist in DB mapping
        if os.path.isfile(relations_csv):
            self.stdout.write("Cross-checking relation CSV tails against DB esco_id tails...")
            # Build sets of tails from DB
            skill_esco_tails = set()
            for s in Skill.objects.exclude(esco_id__in=[None,'']).values_list('esco_id', flat=True):
                tail = tail_of(s)
                if tail:
                    skill_esco_tails.add(tail)

            occ_esco_tails = set()
            for o in Occupation.objects.exclude(esco_id__in=[None,'']).values_list('esco_id', flat=True):
                tail = tail_of(o)
                if tail:
                    occ_esco_tails.add(tail)

            missed = []
            tail_counter = Counter()
            with open(relations_csv, encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                occ_col = next((c for c in cols if 'occupation' in c.lower() and ('uri' in c.lower() or 'id' in c.lower())), cols[0] if cols else None)
                skill_col = next((c for c in cols if 'skill' in c.lower() and ('uri' in c.lower() or 'id' in c.lower())), cols[1] if len(cols)>1 else None)
                rows = 0
                for row in reader:
                    rows += 1
                    occ_tail = tail_of(row.get(occ_col) or '')
                    skill_tail = tail_of(row.get(skill_col) or '')
                    occ_ok = (occ_tail in occ_esco_tails)
                    skill_ok = (skill_tail in skill_esco_tails)
                    if not (occ_ok and skill_ok):
                        missed.append({
                            'row': rows,
                            'occ_tail': occ_tail,
                            'skill_tail': skill_tail,
                            'occ_ok': occ_ok,
                            'skill_ok': skill_ok
                        })
                        tail_counter.update([occ_tail, skill_tail])
                    if len(missed) >= sample_size:
                        break

            self.stdout.write(f"Checked {rows} rows from CSV. Sample misses: {len(missed)}")
            # write sample CSV
            sample_path = 'missed_relations_sample.csv'
            with open(sample_path, 'w', newline='', encoding='utf-8') as outf:
                writer = csv.DictWriter(outf, fieldnames=['row','occ_tail','skill_tail','occ_ok','skill_ok'])
                writer.writeheader()
                for m in missed:
                    writer.writerow(m)
            self.stdout.write(f"Wrote sample misses to {sample_path}")

            # top missing tails
            top_missing = tail_counter.most_common(50)
            top_path = 'top_missing_tails.csv'
            with open(top_path, 'w', newline='', encoding='utf-8') as outf:
                writer = csv.writer(outf)
                writer.writerow(['tail','count'])
                for t,c in top_missing:
                    writer.writerow([t or '', c])
            self.stdout.write(f"Wrote top missing tails to {top_path}")

        # Additional DB-level sample: show 20 relations to inspect
        self.stdout.write("Sample relations (20):")
        for r in OccupationSkillRelation.objects.select_related('occupation','skill')[:20]:
            self.stdout.write(f"{r.id}: {r.occupation.name} ({r.occupation.esco_id}) -> {r.skill.name} ({r.skill.esco_id}) [{r.relation_type}]")

        self.stdout.write("Verification complete.")
