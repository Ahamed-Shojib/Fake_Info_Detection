from django.shortcuts import render
from django.http import JsonResponse
import json
from .utils import clean_text, truncate_text, classify_news_article

def classify_form(request):
    result = None
    if request.method == "POST":
        input_text = request.POST.get("article", "")
        if input_text:
            result = classify_news_article(truncate_text(clean_text(input_text)))
    return render(request, "classify_form.html", {"result": result})