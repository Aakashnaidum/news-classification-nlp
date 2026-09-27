from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from newsclf.predictor import MAX_INPUT_CHARS


class ClassifyForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={"rows": 8}), max_length=MAX_INPUT_CHARS, strip=True)
    model = forms.ChoiceField(required=False)

    def __init__(self, *args, model_choices=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model"].choices = list(model_choices)


class RegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ["username", "password1", "password2"]
