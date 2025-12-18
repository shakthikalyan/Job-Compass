# App/apps.py
from django.apps import AppConfig
import sys
import logging

log = logging.getLogger(__name__)

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'App'

    # def ready(self):
    #     # Do NOT load heavy ML models during manage.py commands
    #     blocked_cmds = {"makemigrations", "migrate", "collectstatic", "test"}
    #     if set(sys.argv) & blocked_cmds:
    #         log.info("App ready during management command - skipping heavy init.")
    #         return

    #     # Try to warm-start the embedding model. If it fails, we continue but
    #     # the utils module will behave in rule-only mode (and log the failure).
    #     try:
    #         # local import to avoid circular imports at migrate time
    #         from . import utils
    #         log.info("App ready. Attempting to preload embedding model (if available).")
    #         model = utils.get_embedding_model()  # lru_cache wrapped; safe to call
    #         if model is not None:
    #             log.info("Embedding model preloaded successfully in App.ready().")
    #         else:
    #             log.warning("Embedding model not available at startup. System will use rule-based fallbacks.")
    #     except Exception:
    #         log.exception("Preloading embedding model failed in App.ready(). Continuing start without model.")
