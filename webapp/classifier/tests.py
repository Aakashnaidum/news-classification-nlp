import tempfile
from pathlib import Path

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from django.urls import reverse

from newsclf.synthetic import write_synthetic_data

from . import services
from .models import Classification

TMP = Path(tempfile.mkdtemp())
ARTIFACTS = TMP / "artifacts"


def setUpModule():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts import train_app_models

    write_synthetic_data(TMP / "data")
    train_app_models.main(["--data-dir", str(TMP / "data"), "--out-dir", str(ARTIFACTS)])


@override_settings(NEWS_ARTIFACTS_DIR=ARTIFACTS)
class ClassifierViewTests(TestCase):
    def setUp(self):
        services.get_predictor.cache_clear()

    def test_home_lists_tasks_with_test_scores(self):
        r = self.client.get(reverse("home"))
        self.assertContains(r, "News category (7 classes)")
        self.assertContains(r, "TF-IDF + logistic regression")

    def test_classify_uses_saved_artifacts(self):
        r = self.client.post(reverse("classify", args=["category"]),
                             {"text": "minister party election vote in parliament", "model": "tfidf_logreg"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context["result"].label, "politics")
        self.assertEqual(Classification.objects.count(), 0)  # anonymous: not stored

    def test_fake_news_page_shows_genre_caveat(self):
        r = self.client.get(reverse("classify", args=["fake_news"]))
        self.assertContains(r, "It is not a fact-checker")

    def test_empty_and_unknown_model_rejected(self):
        r = self.client.post(reverse("classify", args=["category"]), {"text": "   "})
        self.assertIsNone(r.context["result"])
        r = self.client.post(reverse("classify", args=["category"]), {"text": "hello", "model": "evil"})
        self.assertIsNone(r.context["result"])

    def test_unknown_task_404_and_missing_artifacts_503(self):
        self.assertEqual(self.client.get(reverse("classify", args=["nope"])).status_code, 404)
        with self.settings(NEWS_ARTIFACTS_DIR=TMP / "missing"):
            services.get_predictor.cache_clear()
            self.assertEqual(self.client.get(reverse("classify", args=["category"])).status_code, 503)

    def test_history_is_private_per_user(self):
        self.assertEqual(self.client.get(reverse("history")).status_code, 302)
        alice = User.objects.create_user("alice", password="s3cure-pass-123")
        User.objects.create_user("bob", password="s3cure-pass-123")
        self.client.force_login(alice)
        self.client.post(reverse("classify", args=["category"]), {"text": "coach team goal league"})
        self.assertEqual(Classification.objects.filter(user=alice).count(), 1)
        self.client.logout()
        self.client.login(username="bob", password="s3cure-pass-123")
        self.assertContains(self.client.get(reverse("history")), "Nothing yet")
