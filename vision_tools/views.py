from django.shortcuts import render

from .processing import process_image_upload


def image_processor_view(request):
    context = {
        "selected_operation": request.POST.get("filter_type", "sobel"),
    }

    if request.method == "POST" and request.FILES.get("image"):
        result = process_image_upload(request.FILES, request.POST)
        context.update(result)

    return render(request, "vision_tools/index.html", context)
