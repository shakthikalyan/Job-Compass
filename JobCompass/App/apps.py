from django.apps import AppConfig
import sys

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'App'
    def ready(self):
        # Do NOT load heavy ML models during manage.py commands
        blocked_cmds = {"makemigrations", "migrate", "collectstatic", "test"}
        if set(sys.argv) & blocked_cmds:
            return

        # lazy import / load model only when running the server or the actual service
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")