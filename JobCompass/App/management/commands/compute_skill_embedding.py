# App/management/commands/compute_skill_embeddings.py
import os
import numpy as np
from django.core.management.base import BaseCommand
from App.models import Skill

# Optional: configure model via env var
MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

class Command(BaseCommand):
    help = "Compute and store embeddings for Skill.name using sentence-transformers."

    def add_arguments(self, parser):
        parser.add_argument('--batch', type=int, default=128, help='Batch size for encoding')
        parser.add_argument('--limit', type=int, default=0, help='Limit how many skills to process (0 = all)')
        parser.add_argument('--force', action='store_true', help='Recompute embeddings even if they exist')

    def handle(self, *args, **options):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            self.stderr.write(self.style.ERROR("sentence-transformers not installed. Run: pip install sentence-transformers torch"))
            return

        model = SentenceTransformer(MODEL_NAME)
        batch_size = int(options.get('batch', 128))
        limit = int(options.get('limit', 0))
        force = options.get('force', False)

        qs = Skill.objects.all().order_by('id')
        total = qs.count()
        if limit and limit < total:
            qs = qs[:limit]
            total = limit

        names = []
        skills = []
        for s in qs:
            # skip if embedding exists and not forcing
            if s.embedding and not force:
                continue
            names.append(s.name)
            skills.append(s)

        if not names:
            self.stdout.write(self.style.SUCCESS("No skills to embed (all done)"))
            return

        self.stdout.write(self.style.NOTICE(f"Computing embeddings for {len(names)} skills using model {MODEL_NAME}"))

        # encode in batches
        for i in range(0, len(names), batch_size):
            batch_names = names[i:i+batch_size]
            batch_skills = skills[i:i+batch_size]
            emb = model.encode(batch_names, convert_to_numpy=True, show_progress_bar=False)
            for s_obj, vec in zip(batch_skills, emb):
                # store as bytes (float32)
                s_obj.embedding = vec.astype('float32').tobytes()
                s_obj.save(update_fields=['embedding'])
            self.stdout.write(self.style.SUCCESS(f"Saved embeddings for batch {i}..{i+len(batch_names)-1}"))

        self.stdout.write(self.style.SUCCESS(f"Embedding computation complete. Processed {len(names)} skills."))
