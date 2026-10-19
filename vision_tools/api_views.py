from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .processing import process_image_upload


OPERATIONS = [
    {
        "id": "rgb2gray",
        "label": "RGB to Grayscale (Manual)",
        "group": "Color, Histograms & Contrast",
    },
    {
        "id": "plot_histograms",
        "label": "Plot Histogram & CDF (Manual)",
        "group": "Color, Histograms & Contrast",
    },
    {
        "id": "equalize",
        "label": "Histogram Equalization (Manual)",
        "group": "Color, Histograms & Contrast",
    },
    {
        "id": "normalize",
        "label": "Min-Max Normalization (Manual)",
        "group": "Color, Histograms & Contrast",
    },
    {
        "id": "threshold",
        "label": "Global Manual Threshold",
        "group": "Thresholding",
    },
    {
        "id": "iter_threshold",
        "label": "Iterative Optimal Threshold",
        "group": "Thresholding",
    },
    {
        "id": "local_threshold",
        "label": "Local Adaptive Threshold (Gaussian)",
        "group": "Thresholding",
    },
    {
        "id": "awgn",
        "label": "Add White Gaussian Noise (AWGN)",
        "group": "Noise & Spatial Smoothing",
    },
    {
        "id": "salt_pepper",
        "label": "Add Salt & Pepper Noise",
        "group": "Noise & Spatial Smoothing",
    },
    {
        "id": "uniform_noise",
        "label": "Add Uniform Noise",
        "group": "Noise & Spatial Smoothing",
    },
    {"id": "mean", "label": "Mean (Box) Filter", "group": "Noise & Spatial Smoothing"},
    {"id": "gaussian", "label": "Gaussian Filter", "group": "Noise & Spatial Smoothing"},
    {"id": "median", "label": "Median Filter", "group": "Noise & Spatial Smoothing"},
    {"id": "sobel", "label": "Sobel Edge Detection", "group": "Edge Detection"},
    {"id": "prewitt", "label": "Prewitt Edge Detection", "group": "Edge Detection"},
    {"id": "roberts", "label": "Roberts Cross Edge Detection", "group": "Edge Detection"},
    {"id": "auto_canny", "label": "Canny Edge Detection", "group": "Edge Detection"},
    {
        "id": "lowpass",
        "label": "Ideal Low-Pass Filter (FFT)",
        "group": "Frequency Domain & Hybrid",
    },
    {
        "id": "highpass",
        "label": "Ideal High-Pass Filter (FFT)",
        "group": "Frequency Domain & Hybrid",
    },
    {
        "id": "hybrid",
        "label": "Create Hybrid Image",
        "group": "Frequency Domain & Hybrid",
    },
]


@require_http_methods(["GET"])
def operations_list(request):
    return JsonResponse({"operations": OPERATIONS})


@csrf_exempt
@require_http_methods(["POST"])
def process_image_api(request):
    result = process_image_upload(request.FILES, request.POST)
    status = 400 if result.get("error") else 200
    return JsonResponse(result, status=status)


@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "ok", "service": "vision-tools-api"})