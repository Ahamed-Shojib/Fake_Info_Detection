from django.urls import path
from . import views

urlpatterns = [
    path("", views.classify_form, name="classify_form"),
]