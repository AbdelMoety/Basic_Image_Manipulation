# vision_tools/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.image_processor_view, name='image_processor'),
]