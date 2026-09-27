from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import redirect, render

from newsclf.config import TASKS

from .forms import ClassifyForm, RegisterForm
from .models import Classification
from .services import ArtifactError, get_predictor


def home(request):
    cards = []
    for key, task in TASKS.items():
        try:
            p = get_predictor(key)
            cards.append({"key": key, "task": task, "manifest": p.manifest, "available": p.available_models})
        except ArtifactError as exc:
            cards.append({"key": key, "task": task, "error": str(exc)})
    return render(request, "classifier/home.html", {"cards": cards})


def classify(request, task_key):
    if task_key not in TASKS:
        raise Http404("Unknown task")
    task = TASKS[task_key]
    try:
        predictor = get_predictor(task_key)
    except ArtifactError as exc:
        return render(request, "classifier/classify.html", {"task": task, "error": str(exc)}, status=503)

    manifest = predictor.manifest
    choices = [(n, manifest["models"][n]["title"]) for n in predictor.available_models]
    result = None
    if request.method == "POST":
        form = ClassifyForm(request.POST, model_choices=choices)
        if form.is_valid():
            model_name = form.cleaned_data["model"] or manifest["default_model"]
            try:
                result = predictor.predict(form.cleaned_data["text"], model_name)
            except (ValueError, ArtifactError) as exc:
                form.add_error("text", str(exc))
            else:
                if request.user.is_authenticated:
                    Classification.objects.create(
                        user=request.user, task=task_key, model_name=model_name, text=form.cleaned_data["text"],
                        label=result.label, confidence=result.probabilities[result.label])
    else:
        form = ClassifyForm(model_choices=choices, initial={"model": manifest["default_model"]})

    probs = sorted(result.probabilities.items(), key=lambda kv: -kv[1]) if result else []
    return render(request, "classifier/classify.html", {
        "task": task, "task_key": task_key, "form": form, "result": result, "probs": probs,
        "models": manifest["models"], "labels": manifest["labels"],
        "result_title": manifest["models"][result.model]["title"] if result else None,
    })


@login_required
def history(request):
    return render(request, "classifier/history.html",
                  {"items": Classification.objects.filter(user=request.user)[:200]})


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            login(request, form.save())
            messages.success(request, "Account created.")
            return redirect("home")
    else:
        form = RegisterForm()
    return render(request, "registration/register.html", {"form": form})
