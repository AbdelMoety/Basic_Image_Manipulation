# vision_tools/urls.py
from django.urls import path

from . import api_views, views

urlpatterns = [
    path("", views.image_processor_view, name="image_processor"),
    path("api/health/", api_views.health, name="api_health"),
    path("api/operations/", api_views.operations_list, name="api_operations"),
    path("api/process/", api_views.process_image_api, name="api_process"),
]