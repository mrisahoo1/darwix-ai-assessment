from django.urls import path
from . import views

urlpatterns = [
    path('transcribe/', views.TranscriptionView.as_view(), name='transcribe'),
]
