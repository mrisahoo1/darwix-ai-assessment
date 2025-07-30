from django.urls import path
from . import views

urlpatterns = [
    path('suggest-titles/', views.TitleSuggestionView.as_view(), name='suggest-titles'),
]
