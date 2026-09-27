from functools import lru_cache

from django.conf import settings

from newsclf.predictor import ArtifactError, TaskPredictor

__all__ = ["ArtifactError", "get_predictor"]


@lru_cache(maxsize=None)
def get_predictor(task_key: str) -> TaskPredictor:
    """Loaded once per process: the saved vectoriser/tokenizer and label order are reused for every request."""
    return TaskPredictor(settings.NEWS_ARTIFACTS_DIR / task_key)
