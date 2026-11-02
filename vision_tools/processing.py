import os
import json
import uuid
import cv2
from django.core.files.storage import FileSystemStorage
import numpy as np
from . import utils
from . import image_store

VALID_DETECTIONS = ("lines", "circles", "ellipses")


def _get_shifted_fft(fs: FileSystemStorage, content_hash: str, image: np.ndarray):
    """Return (fft_shifted, cache_hit). Key = source file hash + final array shape + gray/color."""
    h, w = image.shape[:2]
    tag = "gray" if image.ndim == 2 else "color"
    cache_dir = fs.path("fft_cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = f"{content_hash}_{h}x{w}_{tag}"
    cache_path = os.path.join(cache_dir, f"{base}.npy")

    if os.path.exists(cache_path):
        try:
            return np.load(cache_path), True
        except (OSError, ValueError):
            pass

    fft_shifted = utils.shifted_fft(image).astype(np.complex64)
    tmp_path = os.path.join(cache_dir, f"{base}.{uuid.uuid4().hex[:6]}.tmp.npy")
    np.save(tmp_path, fft_shifted)
    os.replace(tmp_path, cache_path)
    return fft_shifted, False


def process_image_upload(request_files, request_post):
    """
    Run the selected CV operation on uploaded image(s).
    request_files / request_post behave like Django request.FILES / request.POST.
    Returns a dict with optional keys: error, info, uploaded_image, processed_image, uploaded_image_2.
    """
    context = {}

    fs = FileSystemStorage()
    operation = request_post.get("filter_type", "sobel")
    apply_gray = request_post.get("convert_gray") in ("on", "true", "1", True)

    # Re-use an already-uploaded image (image_id) or save a new upload once (de-duplicated by content).
    src = image_store.resolve(fs, request_files, request_post, "image", "image_id")
    if src is None:
        context["error"] = "Primary image is required (or its earlier upload expired - select it again)."
        return context
    img1_id, img1_path, img1_url = src

    img1 = image_store.load_bgr(img1_id, img1_path)     # decoded once, then served from memory
    if img1 is None:
        context["error"] = "Could not decode the uploaded image."
        return context

    if apply_gray or operation == "rgb2gray":
        img1_work = image_store.load_gray(img1_id, img1)  # grayscale computed once per image
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
    sp_amount = float(request_post.get("sp_amount") or 0.05)
    sp_salt_ratio = float(request_post.get("sp_salt_ratio") or 0.5)
    uniform_low = float(request_post.get("uniform_low") or -25.0)
    uniform_high = float(request_post.get("uniform_high") or 25.0)
    thresh_val = float(request_post.get("thresh_val") or 128.0)
    iter_diff = float(request_post.get("iter_diff") or 0.001)
    local_win = int(request_post.get("local_win") or 15)
    local_c = float(request_post.get("local_c") or 5.0)
    canny_low = float(request_post.get("canny_low") or 50)
    canny_high = float(request_post.get("canny_high") or 150)
    hough_thresh = int(float(request_post.get("hough_thresh") or 100))
    draw_on_image = request_post.get("draw_on_image") in ("on", "true", "1", True)
    circle_r_min = int(float(request_post.get("circle_r_min") or 20))
    circle_r_max = int(float(request_post.get("circle_r_max") or 100))
    circle_ratio = float(request_post.get("circle_ratio") or 0.4)
    circle_min_dist = float(request_post.get("circle_min_dist") or 15)
    circle_max = int(float(request_post.get("circle_max") or 20))
    ellipse_a_min = float(request_post.get("ellipse_a_min") or 15)
    ellipse_a_max = float(request_post.get("ellipse_a_max") or 100)
    ellipse_b_min = float(request_post.get("ellipse_b_min") or 8)
    ellipse_votes = float(request_post.get("ellipse_votes") or 15)
    ellipse_support = float(request_post.get("ellipse_support") or 0.75)
    ellipse_max = int(float(request_post.get("ellipse_max") or 10))
    snake_points = int(float(request_post.get("snake_points") or 60))
    snake_alpha = float(request_post.get("snake_alpha") or 1.0)
    snake_beta = float(request_post.get("snake_beta") or 1.0)
    snake_gamma = float(request_post.get("snake_gamma") or 1.2)
    snake_iters = int(float(request_post.get("snake_iters") or 300))
    snake_radius = int(float(request_post.get("snake_radius") or 1))
    snake_sync = request_post.get("snake_sync") in ("on", "true", "1", True)

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

        elif operation == "salt_pepper":
            result_img = utils.add_salt_and_pepper_noise(
                img1_work, amount=sp_amount, salt_ratio=sp_salt_ratio
            )
            context["info"] = (
                f"Salt & Pepper Noise — Amount: {sp_amount * 100:.1f}% of pixels, "
                f"Salt ratio: {sp_salt_ratio * 100:.0f}%"
            )

        elif operation == "uniform_noise":
            result_img = utils.add_uniform_noise(img1_work, low=uniform_low, high=uniform_high)
            context["info"] = f"Uniform Noise U({uniform_low:g}, {uniform_high:g})"

        elif operation == "mean":
            result_img = utils.mean_filter(img1_work, kernel_size=kernel_size)

        elif operation == "gaussian":
            result_img = utils.gaussian_filter(img1_work, kernel_size=kernel_size, sigma=sigma)

        elif operation == "median":
            result_img = utils.median_filter(img1_work, kernel_size=kernel_size)

        elif operation == "sobel":
            result_img , _ , _ , _ = utils.sobel_edge_detection(img1_work)

        elif operation == "prewitt":
            result_img , _ , _ , _ = utils.prewitt_edge_detection(img1_work)

        elif operation == "roberts":
            result_img , _ , _ , _ = utils.roberts_edge_detection(img1_work)

        elif operation == "auto_canny":
            result_img, (low, high) = utils.auto_canny(
                img1_work,
                low_threshold=canny_low,
                high_threshold=canny_high,
                blur_ksize=kernel_size,
            )
            context["info"] = (
                f"Canny Thresholds — Low: {low}, High: {high} "
                f"(Blur Kernel: {kernel_size}×{kernel_size})"
            )

        elif operation == "manual_canny":
            # Canny runs ONCE; every selected detector reuses the same edge map.
            raw_det = request_post.get("detections") or ""
            detections = [d for d in (s.strip() for s in raw_det.split(",")) if d]
            unknown = [d for d in detections if d not in VALID_DETECTIONS]
            if unknown:
                raise ValueError(f"Unknown detection type(s): {', '.join(unknown)}")

            edges = image_store.get_edges(
                img1_id, img1_work, canny_low, canny_high, sigma, kernel_size
            )
            theta = None
            if "circles" in detections or "ellipses" in detections:
                theta = image_store.get_theta(img1_id, img1_work, sigma, kernel_size)
            info_parts = [
                f"Manual Canny — Low: {canny_low:g}, High: {canny_high:g} "
                f"(Gaussian: {kernel_size}×{kernel_size}, σ = {sigma:g}), "
                f"{int((edges > 0).sum())} edge pixels"
            ]

            # Draw on the original (colour) upload, or on the edge map when unchecked
            canvas = img1 if (detections and draw_on_image) else edges
            result_img = canvas

            if "lines" in detections:
                rhos, thetas, votes = utils.hough_transform(edges, threshold=hough_thresh)
                result_img = utils.draw_hough_lines(result_img, rhos, thetas)
                info_parts.append(f"{len(rhos)} line(s) (vote threshold {hough_thresh})")

            if "circles" in detections:
                if circle_r_min < 1 or circle_r_max <= circle_r_min:
                    raise ValueError("Circle radii: need 1 <= min radius < max radius.")
                h_img, w_img = img1_work.shape[:2]
                r_step = utils.circle_radius_step(h_img, w_img, circle_r_min, circle_r_max)
                circles, _ = utils.hough_circles(
                    img1_work,
                    r_min=circle_r_min,
                    r_max=circle_r_max,
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                    sigma=sigma,
                    gaussian_kernel_size=kernel_size,
                    r_step=r_step,
                    threshold_ratio=circle_ratio,
                    min_dist=circle_min_dist,
                    max_circles=circle_max,
                    edges=edges,
                    theta=theta,
                )
                result_img = utils.draw_circles(result_img, circles)
                info_parts.append(
                    f"{len(circles)} circle(s) (radius {circle_r_min}–{circle_r_max}px, step {r_step}, "
                    f"min support {circle_ratio * 100:.0f}%)"
                    + (" — radius step raised automatically to fit memory" if r_step > 1 else "")
                )

            if "ellipses" in detections:
                if ellipse_b_min < 1 or ellipse_a_min < ellipse_b_min or ellipse_a_max <= ellipse_a_min:
                    raise ValueError(
                        "Ellipse axes: need 1 <= min minor axis <= min major axis < max major axis."
                    )
                ellipses, _ = utils.hough_ellipses(
                    img1_work,
                    a_min=ellipse_a_min,
                    a_max=ellipse_a_max,
                    b_min=ellipse_b_min,
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                    sigma=sigma,
                    gaussian_kernel_size=kernel_size,
                    min_votes=ellipse_votes,
                    min_support=ellipse_support,
                    max_ellipses=ellipse_max,
                    edges=edges,
                    theta=theta,
                )
                result_img = utils.draw_ellipses(result_img, ellipses)
                info_parts.append(
                    f"{len(ellipses)} ellipse(s) (semi-major {ellipse_a_min:g}–{ellipse_a_max:g}px, "
                    f"min semi-minor {ellipse_b_min:g}px, min outline support {ellipse_support * 100:.0f}%)"
                )

            if detections:
                info_parts.append(
                    f"drawn on {'the original image' if draw_on_image else 'the edge map'}"
                )
            context["info"] = " | ".join(info_parts)

        elif operation == "snake":
            raw = request_post.get("snake_contour")
            if not raw:
                raise ValueError("Draw the initial contour first (Draw contour button).")
            try:
                drawn = json.loads(raw)
            except json.JSONDecodeError:
                raise ValueError("Initial contour is not valid JSON.")

            res = utils.snake_segmentation(
                img1, drawn, n_points=snake_points,
                alpha=snake_alpha, beta=snake_beta, gamma=snake_gamma,
                max_iterations=snake_iters, search_radius=max(1, snake_radius),
                sigma=sigma, synchronous=snake_sync,
            )
            canvas = img1 if draw_on_image else res["smoothed"]
            result_img = utils.draw_contour(canvas, res["initial"], color=(255, 128, 0), thickness=1, show_points=False)
            result_img = utils.draw_contour(result_img, res["final"], color=(0, 255, 255), thickness=2)

            context["chain_code"] = "".join(map(str, res["chain"]))
            context["stats"] = {
                "perimeter": round(float(res["perimeter"]), 2),
                "area": round(float(res["area"]), 2),
                "iterations": int(res["iterations"]),
                "points": int(len(res["final"])),
            }
            context["info"] = (
                f"Snake — {len(res['final'])} points, {res['iterations']} iterations, "
                f"start pixel (row, col) = {res['start']}, chain length {len(res['chain'])}, "
                f"perimeter = {res['perimeter']:.2f}px, area = {res['area']:.2f}px² "
                f"(orange = initial, yellow = final)"
            )

        elif operation == "contour_measure":
            raw = request_post.get("snake_contour")
            if not raw:
                raise ValueError("Draw the contour first (Draw contour button).")
            try:
                pts = np.asarray(json.loads(raw), dtype=float)
            except json.JSONDecodeError:
                raise ValueError("Contour is not valid JSON.")
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                raise ValueError("Draw at least 3 points for the contour.")

            h_img, w_img = img1.shape[:2]
            contour = np.rint(pts[:, ::-1]).astype(np.int64)          # (x, y) -> (row, col)
            contour[:, 0] = np.clip(contour[:, 0], 0, h_img - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, w_img - 1)

            start, chain, _ = utils.generate_chain_code(contour)
            perimeter = utils.calculate_perimeter(chain)
            area = utils.calculate_area(contour)

            canvas = img1 if draw_on_image else img1_work
            result_img = utils.draw_contour(canvas, contour, color=(0, 255, 255), thickness=2)

            context["chain_code"] = "".join(map(str, chain))
            context["stats"] = {
                "perimeter": round(float(perimeter), 2),
                "area": round(float(area), 2),
                "points": int(len(contour)),
            }
            context["info"] = (
                f"Contour measurements — {len(contour)} points, start pixel (row, col) = {start}, "
                f"chain length {len(chain)}, perimeter = {perimeter:.2f}px, area = {area:.2f}px²"
            )

        elif operation in ("lowpass", "highpass"):
            cutoff = cutoff_low if operation == "lowpass" else cutoff_high
            content_hash = img1_id
            fft_shifted, cache_hit = _get_shifted_fft(fs, content_hash, img1_work)
            result_img = utils.frequency_filter_pipeline(
                img1_work, cutoff=cutoff, filter_type=operation, fft_shifted=fft_shifted
            )
            context["info"] = (
                f"{'Low' if operation == 'lowpass' else 'High'}-pass, cutoff = {cutoff} "
                f"(FFT {'loaded from cache' if cache_hit else 'computed and cached'})"
            )

        elif operation == "hybrid":
            src2 = image_store.resolve(fs, request_files, request_post, "image_2", "image_2_id")
            if src2 is None:
                context["error"] = "Please upload a second image to generate a Hybrid Image."
                return context
            img2_id, img2_path, img2_url = src2
            img2 = image_store.load_bgr(img2_id, img2_path)
            if img2 is None:
                context["error"] = "Could not decode the second uploaded image."
                return context

            img2_work = image_store.load_gray(img2_id, img2) if apply_gray else img2

            hashes = {1: img1_id, 2: img2_id}
            hits = {}

            def fft_getter(img, idx):
                fft, hit = _get_shifted_fft(fs, hashes[idx], img)
                hits[idx] = hit
                return fft

            result_img = utils.create_hybrid_image(
                img1_work,
                img2_work,
                cutoff_low=cutoff_low,
                cutoff_high=cutoff_high,
                alpha=alpha,
                beta=beta,
                fft_getter=fft_getter,
            )
            context["uploaded_image_2"] = img2_url
            context["info"] = (
                "Hybrid image — FFT cache: "
                f"image 1 {'hit' if hits.get(1) else 'miss'}, "
                f"image 2 {'hit' if hits.get(2) else 'miss'}"
            )

        
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
            plot_filename = f"hist_{uuid.uuid4().hex[:8]}.png"
            plot_path = fs.path(plot_filename)
            utils.plot_histograms(img1_work, plot_path)
            context["uploaded_image"] = img1_url
            context["processed_image"] = fs.url(plot_filename)
            context["info"] = "Computed Manual Histogram & CDF per Channel"
            return context

        else:
            context["error"] = f"Unknown operation: {operation}"
            return context

        out_filename = f"processed_{operation}_{uuid.uuid4().hex[:8]}.png"
        out_path = fs.path(out_filename)
        cv2.imwrite(out_path, result_img)

        context["uploaded_image"] = img1_url
        context["processed_image"] = fs.url(out_filename)

    except Exception as e:
        context["error"] = f"Error processing image: {str(e)}"

    return context