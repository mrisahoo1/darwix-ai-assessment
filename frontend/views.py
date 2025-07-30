from django.shortcuts import render


def index(request):
    """Main frontend page with both features"""
    return render(request, 'frontend/index.html')
