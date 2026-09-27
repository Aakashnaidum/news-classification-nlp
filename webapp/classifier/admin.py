from django.contrib import admin

from .models import Classification


@admin.register(Classification)
class ClassificationAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "task", "model_name", "label", "confidence")
    list_filter = ("task", "model_name")
