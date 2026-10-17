import os
import uuid

import cv2
from django.core.files.storage import FileSystemStorage

from . import utils


def _save_uploaded_file(fs: FileSystemStorage, uploaded_file):
    unique_name = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    filename = fs.save(unique_name, uploaded_file)
    return fs.path(filename), fs.url(filename)


def process_image_upload(request_files, request_post):
    """
    Run the selected CV operation on uploaded image(s).
    request_files / request_post behave like Django request.FILES / request.POST.
    Returns a dict with optional keys: error, info, uploaded_image, processed_image, uploaded_image_2.
    """
    context = {}

    if not request_files.get("image"):
        context["error"] = "Primary image is required."
        return context

    fs = FileSystemStorage()
    operation = request_post.get("filter_type", "sobel")
    apply_gray = request_post.get("convert_gray") in ("on", "true", "1", True)

    img1_path, img1_url = _save_uploaded_file(fs, request_files["image"])
    img1 = cv2.imread(img1_path)

    if img1 is None:
        context["error"] = "Could not decode the uploaded image."
        return context

    if apply_gray or operation == "rgb2gray":
        img1_work = utils.ensure_gray_from_bgr(img1)
    else:
        img1_work = img1

    kernel_size = int(request_post.get("kernel_size") or 3)
    sigma = float(request_post.get("sigma") or 1.0)
    noise_mean = float(request_post.get("noise_mean") or 0.0)
    noise_sigma = float(request_post.get("noise_sigma") or 25.0)
    cutoff_low = float(request_post.get("cutoff_low") or 25.0)
    cutoff_high = float(request_post.get("cutoff_high") or 25.0)
    alpha = float(request_post.get("alpha") or 0.5)
    beta = float(request_post.get("beta") or 0.5)
    thresh_val = float(request_post.get("thresh_val") or 128.0)
    iter_diff = float(request_post.get("iter_diff") or 0.001)
    local_win = int(request_post.get("local_win") or 15)
    local_c = float(request_post.get("local_c") or 5.0)
    canny_sigma = float(request_post.get("canny_sigma") or 0.33)

    if kernel_size < 3:
        kernel_size = 3
    elif kernel_size % 2 == 0:
        kernel_size += 1

    if local_win < 3:
        local_win = 3
    elif local_win % 2 == 0:
        local_win += 1

    try:
        result_img = None

        if operation == "rgb2gray":
            result_img = img1_work

        elif operation == "awgn":
            result_img = utils.add_awgn(img1_work, mean=noise_mean, sigma=noise_sigma)

        elif operation == "mean":
            result_img = utils.mean_filter(img1_work, kernel_size=kernel_size)

        elif operation == "gaussian":
            result_img = utils.gaussian_filter(img1_work, kernel_size=kernel_size, sigma=sigma)

        elif operation == "median":
            result_img = utils.median_filter(img1_work, kernel_size=kernel_size)

        elif operation == "sobel":
            result_img = utils.sobel_edge_detection(img1_work)

        elif operation == "prewitt":
            result_img = utils.prewitt_edge_detection(img1_work)

        elif operation == "roberts":
            result_img = utils.roberts_edge_detection(img1_work)

        elif operation == "auto_canny":
            result_img, (low, high) = utils.auto_canny(
                img1_work, sigma=canny_sigma, blur_ksize=kernel_size
            )
            context["info"] = (
                f"Auto-Canny Thresholds — Low: {low}, High: {high} "
                f"(Blur Kernel: {kernel_size}×{kernel_size})"
            )

        elif operation == "lowpass":
            result_img = utils.frequency_filter_pipeline(
                img1_work, cutoff=cutoff_low, filter_type="lowpass"
            )

        elif operation == "highpass":
            result_img = utils.frequency_filter_pipeline(
                img1_work, cutoff=cutoff_high, filter_type="highpass"
            )

        elif operation == "hybrid":
            if not request_files.get("image_2"):
                context["error"] = "Please upload a second image to generate a Hybrid Image."
                return context

            img2_path, img2_url = _save_uploaded_file(fs, request_files["image_2"])
            img2 = cv2.imread(img2_path)
            if img2 is None:
                context["error"] = "Could not decode the second uploaded image."
                return context

            img2_work = utils.ensure_gray_from_bgr(img2) if apply_gray else img2
            result_img = utils.create_hybrid_image(
                img1_work,
                img2_work,
                cutoff_low=cutoff_low,
                cutoff_high=cutoff_high,
                alpha=alpha,
                beta=beta,
            )
            context["uploaded_image_2"] = img2_url

        elif operation == "threshold":
            result_img = utils.threshold_img(img1_work, threshold=thresh_val)
            context["info"] = f"Applied Manual Global Threshold T = {thresh_val}"

        elif operation == "iter_threshold":
            opt_t = utils.iter_threshold(img1_work, diff=iter_diff)
            result_img = utils.threshold_img(img1_work, threshold=opt_t)
            context["info"] = f"Converged Iterative Threshold T = {opt_t:.2f}"

        elif operation == "local_threshold":
            result_img = utils.local_thresholding(img1_work, window_size=local_win, C=local_c)
            context["info"] = (
                f"Local Gaussian Thresholding (Window: {local_win}×{local_win}, C: {local_c})"
            )

        elif operation == "equalize":
            result_img, _, _ = utils.manual_equalize(img1_work)
            context["info"] = "Applied Manual Histogram Equalization (Grayscale)"

        elif operation == "normalize":
            _, result_img, mn, mx = utils.manual_normalize(img1_work)
            context["info"] = (
                f"Manual Min-Max Normalization (Original Min: {mn}, Original Max: {mx} → [0, 255])"
            )

        elif operation == "plot_histograms":
            plot_filename = f"hist_{os.path.splitext(os.path.basename(img1_path))[0]}.png"
            plot_path = fs.path(plot_filename)
            utils.plot_histograms(img1_work, plot_path)
            context["uploaded_image"] = img1_url
            context["processed_image"] = fs.url(plot_filename)
            context["info"] = "Computed Manual Histogram & CDF per Channel"
            return context

        else:
            context["error"] = f"Unknown operation: {operation}"
            return context

        out_filename = f"processed_{operation}_{os.path.basename(img1_path)}"
        out_path = fs.path(out_filename)
        cv2.imwrite(out_path, result_img)

        context["uploaded_image"] = img1_url
        context["processed_image"] = fs.url(out_filename)

    except Exception as e:
        context["error"] = f"Error processing image: {str(e)}"

    return context
