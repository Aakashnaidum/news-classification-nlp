from django.conf import settings
from django.db import models

from newsclf.config import TASKS


class Classification(models.Model):
    """A classification a signed-in user requested; visible only to that user."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="classifications")
    task = models.CharField(max_length=16, choices=[(k, t.title) for k, t in TASKS.items()])
    model_name = models.CharField(max_length=32)
    text = models.TextField()
    label = models.CharField(max_length=32)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
