from collections import deque

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. Core Utilities & Color Conversion
# ==========================================

def _process_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rounds, clips to [0, 255], and converts to uint8 for image saving/display."""
    return np.clip(np.round(matrix), 0, 255).astype(np.uint8)


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """Converts a 3-channel RGB image to a 2D uint8 grayscale image manually."""
    if img.ndim == 2:
        return img
    # Vectorized matrix-vector dot product using float32 (2x faster than float64)
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    gray = np.dot(img[..., :3].astype(np.float32), weights)
    return _process_matrix(gray)


def ensure_gray_from_bgr(img: np.ndarray) -> np.ndarray:
    """Converts an OpenCV BGR array to 2D grayscale using manual rgb2gray."""
    if img.ndim == 2:
        return img
    # Directly apply BGR weights [0.1140, 0.5870, 0.2989] to avoid extra array copy
    weights_bgr = np.array([0.1140, 0.5870, 0.2989], dtype=np.float32)
    gray = np.dot(img[..., :3].astype(np.float32), weights_bgr)
    return _process_matrix(gray)


def resize_images_to_match(image_1: np.ndarray, image_2: np.ndarray):
    """Resizes two images to the minimum height and width between them."""
    h_min = min(image_1.shape[0], image_2.shape[0])
    w_min = min(image_1.shape[1], image_2.shape[1])
    resize_shape = (w_min, h_min)

    img1_resized = cv2.resize(image_1, resize_shape, interpolation=cv2.INTER_LINEAR)
    img2_resized = cv2.resize(image_2, resize_shape, interpolation=cv2.INTER_LINEAR)
    return img1_resized, img2_resized


# ==========================================
# 2. Frequency Domain & Hybrid Images
# ==========================================

def create_low_pass_filter(shape: tuple, cutoff_frequency: float) -> np.ndarray:
    """Creates an ideal circular 2D low-pass filter mask of shape (H, W)."""
    H, W = shape[:2]
    center_y, center_x = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    # Compare squared distances to avoid computing sqrt over the entire grid
    dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
    return (dist_sq <= (cutoff_frequency ** 2)).astype(np.float32)


def create_high_pass_filter(shape: tuple, cutoff_frequency: float) -> np.ndarray:
    """Creates an ideal circular 2D high-pass filter mask of shape (H, W)."""
    return 1.0 - create_low_pass_filter(shape, cutoff_frequency)


def shifted_fft(image: np.ndarray) -> np.ndarray:
    """Computes 2D FFT and shifts zero-frequency component to center (supports 2D & 3D)."""
    return np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))


def apply_frequency_filter(freq_mask: np.ndarray, fft_shifted: np.ndarray, process: bool = True) -> np.ndarray:
    """Applies a 2D frequency mask to a 2D or 3D shifted FFT array and inverts to spatial domain."""
    if fft_shifted.ndim == 3 and freq_mask.ndim == 2:
        freq_mask = freq_mask[:, :, np.newaxis]

    filtered_shift = np.fft.ifftshift(fft_shifted * freq_mask, axes=(0, 1))
    spatial_image = np.real(np.fft.ifft2(filtered_shift, axes=(0, 1)))

    return _process_matrix(spatial_image) if process else spatial_image


def frequency_filter_pipeline(
    image: np.ndarray,
    cutoff: float,
    filter_type: str = "lowpass",
    process: bool = True,
    fft_shifted: np.ndarray | None = None,
) -> np.ndarray:
    """End-to-end helper to apply low-pass or high-pass filtering to an image.
    If fft_shifted is given, the forward FFT is skipped."""
    if fft_shifted is None:
        fft_shifted = shifted_fft(image)
    if filter_type == "lowpass":
        mask = create_low_pass_filter(image.shape, cutoff)
    elif filter_type == "highpass":
        mask = create_high_pass_filter(image.shape, cutoff)
    else:
        raise ValueError("filter_type must be 'lowpass' or 'highpass'")

    return apply_frequency_filter(mask, fft_shifted, process=process)


def create_hybrid_image(
    image_1: np.ndarray,
    image_2: np.ndarray,
    cutoff_low: float = 20.0,
    cutoff_high: float = 20.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    process: bool = True,
    fft_getter=None,  # callable(image, source_index) -> shifted FFT, or None
) -> np.ndarray:
    """Combines low-frequency components of image_1 with high-frequency components of image_2."""
    img1, img2 = resize_images_to_match(image_1, image_2)
    if img1.ndim != img2.ndim:
        img1 = ensure_gray_from_bgr(img1)
        img2 = ensure_gray_from_bgr(img2)

    fft_1 = fft_getter(img1, 1) if fft_getter else None
    fft_2 = fft_getter(img2, 2) if fft_getter else None

    low_passed = frequency_filter_pipeline(
        img1, cutoff_low, filter_type="lowpass", process=False, fft_shifted=fft_1
    )
    high_passed = frequency_filter_pipeline(
        img2, cutoff_high, filter_type="highpass", process=False, fft_shifted=fft_2
    )

    hybrid = (alpha * low_passed) + (beta * high_passed)
    return _process_matrix(hybrid) if process else hybrid


# ==========================================
# 3. Noise & Vectorized Spatial Convolution
# ==========================================

def add_awgn(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """Adds Additive White Gaussian Noise (AWGN) to a 2D or 3D image array."""
    rng = np.random.default_rng()
    noise = rng.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return _process_matrix(noisy_image)


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05, salt_ratio: float = 0.5) -> np.ndarray:
    """
    Adds salt-and-pepper (impulse) noise to a 2D or 3D image array.

    amount:      fraction of pixels corrupted, in [0, 1].
    salt_ratio:  share of the corrupted pixels that become salt (white, 255);
                 the rest become pepper (black, 0). 0.5 gives an even mix.

    The mask is drawn per pixel *location*, so on color images a corrupted pixel is
    pure white/black in every channel instead of a random color speckle.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError("Salt & pepper amount must be between 0 and 1.")
    if not 0.0 <= salt_ratio <= 1.0:
        raise ValueError("Salt ratio must be between 0 and 1.")

    rng = np.random.default_rng()
    noisy_image = image.copy()

    # One uniform draw per pixel: [0, salt_end) -> salt, [salt_end, amount) -> pepper.
    draw = rng.random(image.shape[:2])
    salt_end = amount * salt_ratio
    noisy_image[draw < salt_end] = 255
    noisy_image[(draw >= salt_end) & (draw < amount)] = 0
    return noisy_image


def add_uniform_noise(image: np.ndarray, low: float = -25.0, high: float = 25.0) -> np.ndarray:
    """
    Adds uniformly distributed noise U(low, high) to a 2D or 3D image array.
    Each value is drawn independently per pixel and per channel (same as add_awgn).
    """
    if low > high:
        raise ValueError("Uniform noise: the lower bound must not exceed the upper bound.")

    rng = np.random.default_rng()
    noise = rng.uniform(low, high, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return _process_matrix(noisy_image)


def convolve(image: np.ndarray, kernel: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Fast vectorized 2D convolution (from scratch).
    Instead of iterating over (H * W) pixels in Python, iterates over (kh * kw)
    kernel taps and accumulates shifted array views across 2D or 3D images at once.
    """
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")

    kh, kw = kernel.shape
    pad_h = ((kh - 1) // 2, kh // 2)
    pad_w = ((kw - 1) // 2, kw // 2)

    if image.ndim == 2:
        pad_width = (pad_h, pad_w)
    elif image.ndim == 3:
        pad_width = (pad_h, pad_w, (0, 0))
    else:
        raise ValueError("Unsupported image shape. Expected 2D or 3D array.")

    padded = np.pad(image.astype(np.float32), pad_width, mode="reflect")
    h, w = image.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)

    # Loop over kernel elements (e.g., 9 iterations for 3x3) instead of millions of pixels
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight != 0.0:
                output += weight * padded[i : i + h, j : j + w]

    return _process_matrix(output) if clip else output


def convolve_separable(image: np.ndarray, k_vert: np.ndarray, k_horiz: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Applies a separable 2D filter in two 1D passes: O(k) instead of O(k^2).
    Used for Mean, Gaussian, and Local Thresholding filters.
    """
    k_v = np.asarray(k_vert, dtype=np.float32).reshape(-1, 1)
    k_h = np.asarray(k_horiz, dtype=np.float32).reshape(1, -1)
    temp = convolve(image, k_v, clip=False)
    return convolve(temp, k_h, clip=clip)


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies a uniform box/mean smoothing filter using separable 1D passes."""
    k_1d = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return convolve_separable(image, k_1d, k_1d, clip=True)


def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Applies a 2D Gaussian filter using separable 1D Gaussian vectors."""
    if sigma <= 0:
        radius = kernel_size // 2
        sigma = 0.3 * (radius - 1) + 0.8
    ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size, dtype=np.float32)
    k_1d = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    k_1d /= np.sum(k_1d)
    return convolve_separable(image, k_1d, k_1d, clip=True)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Vectorized non-linear median filter using zero-copy sliding_window_view
    instead of Python pixel loops.
    """
    pad = kernel_size // 2
    if image.ndim == 2:
        padded = np.pad(image, pad, mode="reflect")
        windows = sliding_window_view(padded, window_shape=(kernel_size, kernel_size))
        output = np.median(windows, axis=(-2, -1))
    elif image.ndim == 3:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        windows = sliding_window_view(padded, window_shape=(kernel_size, kernel_size), axis=(0, 1))
        output = np.median(windows, axis=(-2, -1))
    else:
        raise ValueError("Unsupported image shape.")

    return _process_matrix(output)


# ==========================================
# 4. Edge Detection Operators
# ==========================================

def _edge_magnitude(image: np.ndarray, kx: np.ndarray, ky: np.ndarray , process = True) -> np.ndarray:
    """Convolves with horizontal and vertical gradient kernels: sqrt(Gx^2 + Gy^2)."""
    gx = convolve(image, kx, clip=False)
    gy = convolve(image, ky, clip=False)
    magnitude = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    if process:
      magnitude = _process_matrix(magnitude)
    return magnitude , gx , gy , theta


def sobel_edge_detection(image: np.ndarray, process: bool = True):
    """Returns (magnitude, gx, gy, theta). With process=False the magnitude stays float (unclipped)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return _edge_magnitude(image, kx, ky, process=process)


def prewitt_edge_detection(image: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    return _edge_magnitude(image, kx, ky)


def roberts_edge_detection(image: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    return _edge_magnitude(image, kx, ky)


def auto_canny(image: np.ndarray, low_threshold: float = 50, high_threshold: float = 150, blur_ksize: int = 5):
    """Canny with manual separable Gaussian pre-blur and user-supplied hysteresis thresholds."""
    low, high = int(low_threshold), int(high_threshold)
    if low < 0 or high < 0:
        raise ValueError("Canny thresholds must not be negative.")
    if low >= high:
        raise ValueError("Canny: the lower threshold must be smaller than the upper threshold.")
    gray = ensure_gray_from_bgr(image)
    blurred = gaussian_filter(gray, kernel_size=blur_ksize, sigma=0.0)
    return cv2.Canny(blurred, low, high), (low, high)


# ---- Manual Canny (from scratch) ----

def mapping(values: np.ndarray, ranges, index_mapping) -> np.ndarray:
    """Maps every value to index_mapping[k], where k is the first range [lo, hi) containing it."""
    values = np.expand_dims(values, axis=-1)
    ranges_np = np.array(ranges, dtype=np.float64)
    inside = (values >= ranges_np[:, 0]) & (values < ranges_np[:, 1])
    first_true = inside.argmax(axis=-1)
    return np.array(index_mapping)[first_true]


def non_maximum_suppression(magnitude: np.ndarray, theta_mapping: np.ndarray) -> np.ndarray:
    """Keeps a pixel only if it is >= both neighbours along its (quantised) gradient direction."""
    padded = np.pad(magnitude, 1, mode="constant", constant_values=-1)

    c = padded[1:-1, 1:-1]
    l, r = padded[1:-1, :-2], padded[1:-1, 2:]
    u, d = padded[:-2, 1:-1], padded[2:, 1:-1]
    ul, dr = padded[:-2, :-2], padded[2:, 2:]
    ur, dl = padded[:-2, 2:], padded[2:, :-2]

    conds = [theta_mapping == 0, theta_mapping == 45, theta_mapping == 90, theta_mapping == 135]
    prev = np.select(conds, [l, ul, u, ur], default=np.inf)
    nxt = np.select(conds, [r, dr, d, dl], default=np.inf)

    keep = (c >= prev) & (c >= nxt)
    return np.where(keep, magnitude, 0)


def hysteresis(matrix: np.ndarray, weak: int, strong: int) -> np.ndarray:
    """
    Promotes weak pixels that are 8-connected (directly or through other weak
    pixels) to a strong pixel. Returns a uint8 image with values 0 / 255.
    """
    work = np.pad(matrix, 1, mode="constant", constant_values=-1)
    queue = deque(map(tuple, np.argwhere(work == strong)))  # seeds: only strong pixels
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        i, j = queue.popleft()
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if work[ni, nj] == weak:
                work[ni, nj] = strong
                queue.append((ni, nj))

    return np.where(work[1:-1, 1:-1] == strong, 255, 0).astype(np.uint8)


def manual_canny(
    image: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    sigma: float = 1.0,
    gaussian_kernel_size: int = 3,
) -> np.ndarray:
    """Canny edge detector built from scratch: blur -> Sobel -> NMS -> double threshold -> hysteresis."""
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Canny thresholds must not be negative.")
    if low_threshold >= high_threshold:
        raise ValueError("Canny: the lower threshold must be smaller than the upper threshold.")

    gray = ensure_gray_from_bgr(image)
    blurred = gaussian_filter(gray, gaussian_kernel_size, sigma)
    magnitude, _, _, theta = sobel_edge_detection(blurred, process=False)

    # Gradient angle folded into [0, 180) and quantised to 0 / 45 / 90 / 135 degrees
    theta_deg = (np.degrees(theta) + 180) % 180
    angle_ranges = [(0, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5), (157.5, 180)]
    theta_mapping = mapping(theta_deg, angle_ranges, [0, 45, 90, 135, 0])

    thin = non_maximum_suppression(magnitude, theta_mapping)

    # Double threshold: below low -> 0, between -> weak (75), at/above high -> strong (255)
    levels = mapping(
        thin,
        [(-np.inf, low_threshold), (low_threshold, high_threshold), (high_threshold, np.inf)],
        [0, 75, 255],
    )
    return hysteresis(levels, weak=75, strong=255)


# ==========================================
# 4b. Hough Transform (Line Detection)
# ==========================================

def hough_transform(edges: np.ndarray, threshold: int, peak_window: int = 9, max_lines: int = 200):
    """
    Line Hough transform on a binary edge map, using rho = y*cos(t) - x*sin(t).
    Returns (rhos, thetas_in_radians, votes), strongest lines first. Only local peaks of the
    accumulator above `threshold` are kept, so one real line yields one detected line.
    """
    ys, xs = np.nonzero(edges)
    thetas = np.deg2rad(np.arange(0, 180, 1))
    height, width = edges.shape[:2]
    rho_max = int(np.ceil(np.hypot(height, width)))
    n_rho = 2 * rho_max + 1

    accumulator = np.zeros((n_rho, len(thetas)), dtype=np.uint32)
    if ys.size == 0:
        return np.array([]), np.array([]), np.array([])

    ys = ys.astype(np.float32)
    xs = xs.astype(np.float32)
    # One bincount per angle keeps memory at O(N) instead of O(N * 180).
    for t, theta in enumerate(thetas):
        rho = ys * np.cos(theta) - xs * np.sin(theta)
        idx = np.round(rho).astype(np.int64) + rho_max
        accumulator[:, t] = np.bincount(idx, minlength=n_rho)

    # Non-maximum suppression in (rho, theta) space
    half = peak_window // 2
    # theta wraps at 180 deg with rho -> -rho, so pad the theta axis with the flipped opposite edge
    wrapped = np.hstack([accumulator[::-1, -half:], accumulator, accumulator[::-1, :half]])
    padded = np.pad(wrapped, ((half, half), (0, 0)), mode="constant")
    local_max = sliding_window_view(padded, (peak_window, peak_window)).max(axis=(-2, -1))
    peaks = (accumulator > threshold) & (accumulator == local_max)

    r_idx, t_idx = np.nonzero(peaks)
    votes = accumulator[r_idx, t_idx]
    order = np.argsort(-votes)[:max_lines]
    r_idx, t_idx, votes = r_idx[order], t_idx[order], votes[order]
    return (r_idx - rho_max).astype(np.float64), thetas[t_idx], votes


def draw_hough_lines(image: np.ndarray, rhos, thetas, color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draws infinite (image-clipped) lines given in (rho, theta) form on a BGR copy of `image`."""
    result = np.array(image, dtype=np.uint8, order="C", copy=True)
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    height, width = result.shape[:2]
    length = int(np.hypot(height, width))

    for rho, theta in zip(rhos, thetas):
        c, s = np.cos(theta), np.sin(theta)
        x0, y0 = -rho * s, rho * c
        p1 = (int(x0 + length * c), int(y0 + length * s))
        p2 = (int(x0 - length * c), int(y0 - length * s))
        cv2.line(result, p1, p2, color, thickness)

    return result


# ==========================================
# 4c. Hough Transform (Circle Detection)
# ==========================================

# Max accumulator size (radii x H x W voxels, float32). Keeps peak memory around 400 MB.
MAX_CIRCLE_VOXELS = 25_000_000


def circle_radius_step(height: int, width: int, r_min: int, r_max: int) -> int:
    """Smallest radius step that keeps the 3D accumulator under MAX_CIRCLE_VOXELS."""
    max_radii = max(1, MAX_CIRCLE_VOXELS // (height * width))
    return max(1, int(np.ceil((r_max - r_min + 1) / max_radii)))


def _local_maxima_3d(volume: np.ndarray) -> np.ndarray:
    """True where a voxel is >= all of its 26 neighbours (shift-and-compare, no pixel loops)."""
    padded = np.pad(volume, 1, mode="constant", constant_values=-np.inf)
    nr, h, w = volume.shape
    is_max = np.ones(volume.shape, dtype=bool)
    for dr in range(3):
        for dy in range(3):
            for dx in range(3):
                if (dr, dy, dx) == (1, 1, 1):
                    continue
                is_max &= volume >= padded[dr:dr + nr, dy:dy + h, dx:dx + w]
    return is_max


def hough_circle_transform(edges, theta, r_min, r_max, r_step=1, vote_window=3):
    """
    Gradient-based Hough Circle Transform.
    Each edge pixel votes ONLY along its gradient direction (both ways, so bright-on-dark
    and dark-on-bright circles both work) instead of voting on a full circle -> 1 vote
    per radius per direction instead of ~2*pi*r votes.

    edges : binary edge map (output of manual_canny)
    theta : gradient angle map in radians (output of sobel_edge_detection)
    Returns accumulator of shape (num_radii, H, W) and the radii array.
    """
    h, w = edges.shape
    ys, xs = np.where(edges > 0)
    cos_t = np.cos(theta[ys, xs])
    sin_t = np.sin(theta[ys, xs])

    radii = np.arange(r_min, r_max + 1, r_step)
    box = np.ones(vote_window, dtype=np.float32)          # sums votes in a small window -> tolerates +-1px jitter
    accumulator = np.zeros((len(radii), h, w), dtype=np.float32)

    for k, r in enumerate(radii):
        votes = np.zeros(h * w, dtype=np.float32)
        for sign in (1, -1):
            cx = np.rint(xs + sign * r * cos_t).astype(int)
            cy = np.rint(ys + sign * r * sin_t).astype(int)
            inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            votes += np.bincount(cy[inside] * w + cx[inside], minlength=h * w)
        # separable convolution (clip=False keeps real vote counts)
        accumulator[k] = convolve_separable(votes.reshape(h, w), box, box, clip=False)

    return accumulator, radii


def find_circles(accumulator, radii, threshold_ratio=0.4, min_dist=15, max_circles=20, max_candidates=5000):
    """
    Picks peaks from the 3D accumulator.
    threshold_ratio : minimum fraction of the circle's circumference (2*pi*r) that must be supported by edge pixels.
    min_dist        : circles closer than this in (cx, cy, r) space are treated as duplicates.
    Returns an array of rows (cx, cy, r, score), best first.
    """
    score = accumulator / (2 * np.pi * radii)[:, None, None]
    candidates = (score >= threshold_ratio) & _local_maxima_3d(score)
    r_idx, ys, xs = np.nonzero(candidates)

    order = np.argsort(-score[r_idx, ys, xs])[:max_candidates]
    r_idx, ys, xs = r_idx[order], ys[order], xs[order]

    accepted = np.empty((0, 3), dtype=np.float32)
    circles = []
    for ri, y, x in zip(r_idx, ys, xs):
        point = np.array([x, y, radii[ri]], dtype=np.float32)
        if len(accepted) and np.min(np.linalg.norm(accepted - point, axis=1)) < min_dist:
            continue
        accepted = np.vstack([accepted, point])
        circles.append((x, y, radii[ri], score[ri, y, x]))
        if len(circles) >= max_circles:
            break
    return np.array(circles, dtype=np.float32).reshape(-1, 4)


def hough_circles(image, r_min, r_max, low_threshold=50, high_threshold=150, sigma=1.4,
                  gaussian_kernel_size=5, r_step=1, threshold_ratio=0.4, min_dist=15, max_circles=20):
    """Full pipeline built only from this module's own functions. `image` is BGR or grayscale."""
    gray = ensure_gray_from_bgr(image)
    edges = manual_canny(image, low_threshold, high_threshold, sigma, gaussian_kernel_size)

    # manual_canny doesn't return the gradient angles, so recompute them (cheap, vectorized)
    smoothed = gaussian_filter(gray, gaussian_kernel_size, sigma)
    _, _, _, theta = sobel_edge_detection(smoothed, process=False)

    accumulator, radii = hough_circle_transform(edges, theta, r_min, r_max, r_step)
    circles = find_circles(accumulator, radii, threshold_ratio, min_dist, max_circles)
    return circles, edges


def draw_circles(image, circles, color=(0, 0, 255), thickness=2, center_color=(0, 255, 0)):
    """Draws circles with pure NumPy (no cv2.circle). Colours are BGR; returns a 3-channel image."""
    out = np.array(image, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    h, w = out.shape[:2]

    for cx, cy, r, _ in circles:
        x0, x1 = max(int(cx - r - thickness - 1), 0), min(int(cx + r + thickness + 2), w)
        y0, y1 = max(int(cy - r - thickness - 1), 0), min(int(cy + r + thickness + 2), h)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)
        region = out[y0:y1, x0:x1]                       # view into `out`
        region[np.abs(dist - r) <= thickness / 2] = color
        region[dist <= 2] = center_color
    return out


# ==========================================
# 4d. Hough Transform (Ellipse Detection)
# ==========================================

def _symmetric_pairs(xs, ys, theta, d_min, d_max, angle_tol_deg=12):
    """
    Two edge points that are symmetric about an ellipse centre have PARALLEL tangents, i.e.
    gradients pointing in OPPOSITE directions. Edge points are bucketed by gradient angle, so only
    points in opposite buckets are compared (instead of all N^2 pairs).
    Returns index arrays (I, J) of the matching pairs.
    """
    n_bins = 2 * int(np.ceil(180 / angle_tol_deg))            # bucket width <= tolerance, always even
    ang = np.mod(theta, 2 * np.pi)
    bins = np.minimum((ang / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    members = [np.nonzero(bins == b)[0] for b in range(n_bins)]
    tol = np.deg2rad(angle_tol_deg)

    I, J = [], []
    for b in range(n_bins):
        P = members[b]
        partner = np.concatenate([members[(b + n_bins // 2 + s) % n_bins] for s in (-1, 0, 1)])
        if len(P) == 0 or len(partner) == 0:
            continue
        pi, qi = np.meshgrid(P, partner, indexing="ij")
        pi, qi = pi.ravel(), qi.ravel()
        keep = pi < qi                                         # every pair is seen from both sides; keep it once
        pi, qi = pi[keep], qi[keep]
        diff = np.abs(np.angle(np.exp(1j * (ang[pi] - ang[qi] - np.pi))))
        dist = np.hypot(xs[pi] - xs[qi], ys[pi] - ys[qi])
        ok = (diff <= tol) & (dist >= d_min) & (dist <= d_max)
        I.append(pi[ok]); J.append(qi[ok])

    if not I:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(I), np.concatenate(J)


def _conic_to_axes(A, B, C):
    """Centered conic A x^2 + B xy + C y^2 = 1  ->  (semi-major a, semi-minor b, major-axis angle phi)."""
    vals, vecs = np.linalg.eigh(np.array([[A, B / 2], [B / 2, C]]))      # ascending eigenvalues
    a, b = 1 / np.sqrt(vals[0]), 1 / np.sqrt(vals[1])
    phi = np.arctan2(vecs[1, 0], vecs[0, 0])
    return a, b, phi


def _is_ellipse(p):
    return (p[0] > 0) and (p[2] > 0) and (4 * p[0] * p[2] - p[1] ** 2 > 0)


def _fit_centered_ellipse(qx, qy, n_iter=500, inlier_tol=1.5, rng=None):
    """
    RANSAC fit of A x^2 + B xy + C y^2 = 1 to points given RELATIVE to the ellipse centre.
    With the centre known only 3 unknowns remain, so 3 points define a candidate (batched, no Python loop).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    m = len(qx)
    if m < 6:
        return None
    X = np.stack([qx ** 2, qx * qy, qy ** 2], axis=1)                    # (m, 3)

    def sampson(p):                                                     # approx. distance of every point to the conic
        F = X @ p
        gx = 2 * p[0] * qx + p[1] * qy
        gy = p[1] * qx + 2 * p[2] * qy
        return np.abs(F - 1) / (np.hypot(gx, gy) + 1e-9)

    M = X[rng.integers(0, m, size=(n_iter, 3))]                          # (n_iter, 3, 3)
    M = M[np.abs(np.linalg.det(M)) > 1e-6]
    if len(M) == 0:
        return None
    P = np.linalg.solve(M, np.ones((len(M), 3, 1)))[..., 0]              # (k, 3)
    P = P[(P[:, 0] > 0) & (P[:, 2] > 0) & (4 * P[:, 0] * P[:, 2] - P[:, 1] ** 2 > 0)]
    if len(P) == 0:
        return None

    F = P @ X.T                                                          # (k, m)
    gx = 2 * P[:, [0]] * qx + P[:, [1]] * qy
    gy = P[:, [1]] * qx + 2 * P[:, [2]] * qy
    inliers = (np.abs(F - 1) / (np.hypot(gx, gy) + 1e-9) < inlier_tol).sum(axis=1)
    best = P[np.argmax(inliers)]

    for _ in range(3):                                                   # refit on inliers (least squares)
        inl = sampson(best) < inlier_tol
        if inl.sum() < 6:
            break
        refit = np.linalg.lstsq(X[inl], np.ones(inl.sum()), rcond=None)[0]
        if not _is_ellipse(refit):
            break
        best = refit
    return best if _is_ellipse(best) else None


def _ellipse_support(edges_dilated, cx, cy, a, b, phi):
    """Fraction of the ellipse outline that lies on (dilated) edge pixels - the final 'is this real?' score."""
    h, w = edges_dilated.shape
    perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))   # Ramanujan
    t = np.linspace(0, 2 * np.pi, max(int(perimeter), 16), endpoint=False)
    x = np.rint(cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)).astype(int)
    y = np.rint(cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hit = np.zeros(len(t), dtype=bool)
    hit[inside] = edges_dilated[y[inside], x[inside]] > 0
    return hit.mean()


def hough_ellipse_transform(edges, theta, a_max, b_min=5, angle_tol_deg=12, vote_window=5, max_edge_points=2500, seed=0):
    """
    Step 1 of ellipse detection: find CENTRE candidates.
    Every pair of 'symmetric' edge points votes for its midpoint in a 2D accumulator.
    Returns (centre accumulator, xs, ys, pair indices I, J, pair midpoints mx, my).
    """
    h, w = edges.shape
    ys, xs = np.where(edges > 0)
    if len(xs) > max_edge_points:                                       # keep the pair count manageable
        keep = np.random.default_rng(seed).choice(len(xs), max_edge_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    th = theta[ys, xs]

    I, J = _symmetric_pairs(xs, ys, th, d_min=2 * b_min, d_max=2 * a_max, angle_tol_deg=angle_tol_deg)
    mx, my = (xs[I] + xs[J]) / 2.0, (ys[I] + ys[J]) / 2.0

    votes = np.bincount(np.rint(my).astype(int) * w + np.rint(mx).astype(int), minlength=h * w).astype(np.float32)
    box = np.ones(vote_window, dtype=np.float32)
    acc = convolve_separable(votes.reshape(h, w), box, box, clip=False)
    return acc, xs, ys, I, J, mx, my


def find_ellipses(edges, acc, xs, ys, I, J, mx, my, min_votes=15, min_center_dist=10, max_candidates=30,
                  center_tol=3.0, inlier_tol=1.5, n_iter=500, a_min=8, a_max=100, b_min=5,
                  min_support=0.6, max_ellipses=10):
    """
    Step 2: for each centre candidate, collect the points that voted for it, RANSAC-fit (a, b, phi),
    then keep the ellipse only if enough of its outline is really covered by edge pixels.
    Returns rows (cx, cy, a, b, phi_radians, support), best first.
    """
    # --- centre peaks (greedy non-maximum suppression) ---
    peaks = (acc >= min_votes) & _local_maxima_3d(acc[None])[0]
    py, px = np.nonzero(peaks)
    order = np.argsort(-acc[py, px])
    centers = []
    for y, x in zip(py[order], px[order]):
        if all(np.hypot(x - cx, y - cy) >= min_center_dist for cx, cy in centers):
            centers.append((x, y))
        if len(centers) >= max_candidates:
            break

    box3 = np.ones(3, dtype=np.float32)
    edges_dilated = convolve_separable(edges.astype(np.float32), box3, box3, clip=False)
    rng = np.random.default_rng(0)

    results = []
    for cx0, cy0 in centers:
        sel = (np.abs(mx - cx0) <= center_tol) & (np.abs(my - cy0) <= center_tol)
        if sel.sum() < 3:
            continue
        cx, cy = np.median(mx[sel]), np.median(my[sel])                  # refine the centre with sub-pixel midpoints
        pts = np.unique(np.concatenate([I[sel], J[sel]]))
        p = _fit_centered_ellipse(xs[pts] - cx, ys[pts] - cy, n_iter, inlier_tol, rng)
        if p is None:
            continue
        a, b, phi = _conic_to_axes(*p)
        if not (a_min <= a <= a_max and b >= b_min):
            continue
        support = _ellipse_support(edges_dilated, cx, cy, a, b, phi)
        if support >= min_support:
            results.append((cx, cy, a, b, phi, support))

    results.sort(key=lambda r: -r[5])
    return np.array(results[:max_ellipses], dtype=np.float32).reshape(-1, 6)


def hough_ellipses(image, a_min, a_max, b_min=5, low_threshold=50, high_threshold=150, sigma=1.4,
                   gaussian_kernel_size=5, angle_tol_deg=12, min_votes=15, min_support=0.6,
                   min_center_dist=10, max_ellipses=10):
    """Full pipeline built only from this module's own functions. `image` is BGR or grayscale."""
    gray = ensure_gray_from_bgr(image)
    edges = manual_canny(image, low_threshold, high_threshold, sigma, gaussian_kernel_size)
    smoothed = gaussian_filter(gray, gaussian_kernel_size, sigma)
    _, _, _, theta = sobel_edge_detection(smoothed, process=False)

    acc, xs, ys, I, J, mx, my = hough_ellipse_transform(edges, theta, a_max, b_min, angle_tol_deg)
    ellipses = find_ellipses(edges, acc, xs, ys, I, J, mx, my, min_votes=min_votes, min_center_dist=min_center_dist,
                             a_min=a_min, a_max=a_max, b_min=b_min, min_support=min_support, max_ellipses=max_ellipses)
    return ellipses, edges


def draw_ellipses(image, ellipses, color=(0, 0, 255), thickness=2, center_color=(0, 255, 0)):
    """Draws ellipses with pure NumPy (no cv2.ellipse). Colours are BGR; returns a 3-channel image."""
    out = np.array(image, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    h, w = out.shape[:2]

    for cx, cy, a, b, phi, _ in ellipses:
        r = a + thickness + 2
        x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 1, w)
        y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 1, h)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        u = (xx - cx) * np.cos(phi) + (yy - cy) * np.sin(phi)              # coordinates in the ellipse's own frame
        v = -(xx - cx) * np.sin(phi) + (yy - cy) * np.cos(phi)
        g2 = (u / a) ** 2 + (v / b) ** 2
        dist = np.abs(g2 - 1) / (2 * np.hypot(u / a ** 2, v / b ** 2) + 1e-9)   # ~ pixel distance to the outline
        region = out[y0:y1, x0:x1]
        region[dist <= thickness / 2] = color
        region[np.hypot(xx - cx, yy - cy) <= 2] = center_color
    return out


# ==========================================
# 5. Thresholding (Global, Iterative, Local)
# ==========================================

def threshold_img(img: np.ndarray, threshold: float) -> np.ndarray:
    """Manual binary thresholding: returns uint8 image with values 0 or 255."""
    gray = ensure_gray_from_bgr(img)
    return np.where(gray > threshold, 255, 0).astype(np.uint8)


def iter_threshold(img: np.ndarray, diff: float = 1e-3) -> float:
    """
    Computes optimal global threshold iteratively using cluster means.
    Optimized to operate on the 256-bin manual histogram in O(256) per iteration
    instead of scanning millions of pixels every loop iteration.
    """
    gray = ensure_gray_from_bgr(img)
    hist = manual_histogram(gray).astype(np.float64)
    intensities = np.arange(256, dtype=np.float64)

    chosen_threshold = 0.0
    current_threshold = (float(gray.min()) + float(gray.max())) / 2.0

    while np.abs(current_threshold - chosen_threshold) > diff:
        split = int(np.floor(current_threshold)) + 1

        small_weights = hist[:split]
        large_weights = hist[split:]

        small_sum = small_weights.sum()
        large_sum = large_weights.sum()

        if small_sum == 0 or large_sum == 0:
            break

        smaller_mean = np.dot(intensities[:split], small_weights) / small_sum
        larger_mean = np.dot(intensities[split:], large_weights) / large_sum

        chosen_threshold = current_threshold
        current_threshold = (larger_mean + smaller_mean) / 2.0

    return float(chosen_threshold)


def local_thresholding(image: np.ndarray, window_size: int = 15, C: float = 5.0) -> np.ndarray:
    """
    Adaptive local Gaussian thresholding using separable 1D manual convolution.
    Reduces 15x15 window from 225 operations per pixel to 30 vectorized slice additions.
    """
    gray = ensure_gray_from_bgr(image).astype(np.float32)

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer >= 3.")

    radius = window_size // 2
    sigma = 0.3 * (radius - 1) + 0.8

    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    k_1d = np.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    k_1d /= k_1d.sum()

    threshold_map = convolve_separable(gray, k_1d, k_1d, clip=False) - C
    return np.where(gray < threshold_map, 0, 255).astype(np.uint8)


# ==========================================
# 6. Histograms, Equalization & Normalization
# ==========================================

def manual_histogram(gray: np.ndarray) -> np.ndarray:
    """
    h[k] = number of pixels whose intensity equals k (k = 0..255).
    Uses single-pass unbuffered index accumulation instead of 256 full-image scans.
    """
    hist = np.zeros(256, dtype=np.int64)
    np.add.at(hist, gray.ravel(), 1)
    return hist


def manual_distribution(hist: np.ndarray) -> np.ndarray:
    """p[k] = h[k] / N (probability of each intensity)."""
    total = hist.sum()
    return hist / total if total > 0 else np.zeros_like(hist, dtype=np.float64)


def manual_cdf(hist: np.ndarray) -> np.ndarray:
    """Computes cumulative histogram manually."""
    cdf = np.zeros(256, dtype=np.int64)
    cdf[0] = hist[0]
    for k in range(1, 256):
        cdf[k] = cdf[k - 1] + hist[k]
    return cdf


def manual_equalize(image: np.ndarray, hist: np.ndarray = None):
    """Manual histogram equalization. Returns (equalized_image, cdf, mapping)."""
    gray = ensure_gray_from_bgr(image)
    if hist is None:
        hist = manual_histogram(gray)
    N = gray.size

    cdf = manual_cdf(hist)
    nonzero_indices = np.nonzero(hist)[0]
    cdf_min = cdf[nonzero_indices[0]] if nonzero_indices.size > 0 else 0

    if N == cdf_min:
        mapping = np.arange(256, dtype=np.uint8)
    else:
        mapping = np.round((cdf - cdf_min) / (N - cdf_min) * 255.0)
        mapping = np.clip(mapping, 0, 255).astype(np.uint8)

    return mapping[gray], cdf, mapping


def manual_normalize(image: np.ndarray):
    """Manual min-max normalization. Returns (norm_01 float32, norm_255 uint8, min, max)."""
    g = image.astype(np.float32)
    mn, mx = float(g.min()), float(g.max())
    if mx == mn:
        return np.zeros_like(g), image.copy(), int(mn), int(mx)
    norm_01 = (g - mn) / (mx - mn)
    norm_255 = np.round(norm_01 * 255.0).astype(np.uint8)
    return norm_01, norm_255, int(mn), int(mx)


def plot_histograms(image: np.ndarray, save_path: str):
    """Plots the histogram and CDF for each image channel and saves the figure."""
    if image is None:
        raise ValueError("Image is empty.")

    if image.ndim == 2:
        channels = [("Grayscale", image, "#343a40")]
    elif image.ndim == 3:
        n_channels = image.shape[2]
        if n_channels == 3:
            channels = [
                ("Blue Channel", image[:, :, 0], "#0d6efd"),
                ("Green Channel", image[:, :, 1], "#198754"),
                ("Red Channel", image[:, :, 2], "#dc3545"),
            ]
        elif n_channels == 4:
            channels = [
                ("Blue Channel", image[:, :, 0], "#0d6efd"),
                ("Green Channel", image[:, :, 1], "#198754"),
                ("Red Channel", image[:, :, 2], "#dc3545"),
                ("Alpha Channel", image[:, :, 3], "#6c757d"),
            ]
        else:
            raise ValueError("Unsupported number of channels.")
    else:
        raise ValueError("Expected a 2D or 3D image.")

    fig, axes = plt.subplots(
        len(channels), 2,
        figsize=(11, 3.5 * len(channels)),
        squeeze=False,
    )
    bins = np.arange(256)

    for i, (name, channel, color) in enumerate(channels):
        hist = manual_histogram(channel)
        cdf = manual_cdf(hist)
        cdf_normalized = cdf / cdf[-1] if cdf[-1] > 0 else cdf

        axes[i, 0].plot(bins, hist, color=color)
        axes[i, 0].set_title(f"{name} Histogram")
        axes[i, 0].set_xlabel("Intensity")
        axes[i, 0].set_ylabel("Pixel Count")
        axes[i, 0].set_xlim(0, 255)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(bins, cdf_normalized, color=color)
        axes[i, 1].set_title(f"{name} CDF")
        axes[i, 1].set_xlabel("Intensity")
        axes[i, 1].set_ylabel("Cumulative Probability")
        axes[i, 1].set_xlim(0, 255)
        axes[i, 1].set_ylim(0, 1.05)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


# ==========================================
# 7. Active Contour (Greedy Snake)
# ==========================================

def calculate_image_gradients(image: np.ndarray):
    """Sobel gradients (Ix, Iy). Reuses sobel_edge_detection instead of re-implementing Sobel."""
    _, gx, gy, _ = sobel_edge_detection(image, process=False)
    return gx, gy


def calculate_image_energy(image: np.ndarray) -> np.ndarray:
    """E_image = -|grad I|^2, scaled to [-1, 0] so it is comparable to the other terms."""
    ix, iy = calculate_image_gradients(image)
    g2 = ix ** 2 + iy ** 2
    m = g2.max()
    return -g2 / m if m > 0 else -g2


def preprocess_image(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Grayscale -> Gaussian smoothing. Reuses ensure_gray_from_bgr + gaussian_filter."""
    gray = ensure_gray_from_bgr(image)
    if sigma and sigma > 0:
        gray = gaussian_filter(gray, kernel_size=2 * int(np.ceil(3 * sigma)) + 1, sigma=sigma)
    return gray


def average_spacing(contour: np.ndarray) -> float:
    return float(np.linalg.norm(contour - np.roll(contour, 1, axis=0), axis=1).mean())


def resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """Closed polygon -> exactly n_points, equally spaced along its length. (row, col) int."""
    pts = np.asarray(contour, dtype=float)
    closed = np.vstack([pts, pts[:1]])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    if cum[-1] == 0:
        raise ValueError("The drawn contour has zero length.")
    t = np.linspace(0, cum[-1], n_points, endpoint=False)
    out = np.stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])], axis=1)
    return np.rint(out).astype(np.int64)


def generate_candidates(point, search_radius: int, image_shape) -> np.ndarray:
    """(2r+1)^2 neighbourhood inside the image; the current position comes FIRST (wins ties)."""
    r0, c0 = np.rint(point).astype(int)
    d = np.arange(-search_radius, search_radius + 1)
    dr, dc = np.meshgrid(d, d, indexing="ij")
    offs = np.stack([dr.ravel(), dc.ravel()], axis=1)
    offs = np.vstack([[0, 0], offs[np.any(offs != 0, axis=1)]])
    cands = offs + np.array([r0, c0])
    ok = (cands[:, 0] >= 0) & (cands[:, 0] < image_shape[0]) & (cands[:, 1] >= 0) & (cands[:, 1] < image_shape[1])
    return cands[ok]


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return np.zeros_like(x) if rng < 1e-12 else (x - x.min()) / rng


def calculate_candidate_components(image_energy, contour, index, candidates, d_avg=None):
    """Raw (un-weighted) continuity, curvature and image energies for every candidate.
    image_energy=None -> image term is 0 (internal energy only)."""
    contour = np.asarray(contour, dtype=float)
    n = len(contour)
    if d_avg is None:
        d_avg = average_spacing(contour)
    prev_p, next_p = contour[(index - 1) % n], contour[(index + 1) % n]
    Q = np.asarray(candidates, dtype=float)
    e_cont = (np.linalg.norm(Q - prev_p, axis=1) - d_avg) ** 2
    e_curv = np.sum((prev_p - 2 * Q + next_p) ** 2, axis=1)
    if image_energy is None:
        e_img = np.zeros(len(Q))
    else:
        e_img = image_energy[Q[:, 0].astype(int), Q[:, 1].astype(int)]
    return e_cont, e_curv, e_img


def calculate_internal_energy(contour, index, alpha, beta, candidate=None, d_avg=None) -> float:
    """alpha*E_cont + beta*E_curv for point `index` (optionally evaluated AS IF it were at `candidate`)."""
    contour = np.asarray(contour, dtype=float)
    q = contour[index] if candidate is None else np.asarray(candidate, dtype=float)
    ec, ek, _ = calculate_candidate_components(None, contour, index, q[None, :], d_avg)
    return float(alpha * ec[0] + beta * ek[0])


def calculate_candidate_energy(image_energy, contour, index, candidate, alpha, beta, gamma, d_avg=None) -> float:
    """Raw (un-normalised) total energy of a single candidate position."""
    ec, ek, ei = calculate_candidate_components(image_energy, contour, index,
                                                np.asarray(candidate)[None, :], d_avg)
    return float(alpha * ec[0] + beta * ek[0] + gamma * ei[0])


def select_best_candidate(image_energy, contour, index, candidates, alpha, beta, gamma, d_avg=None):
    """Min-max normalise each term across the candidates, combine, take argmin."""
    ec, ek, ei = calculate_candidate_components(image_energy, contour, index, candidates, d_avg)
    total = alpha * _minmax(ec) + beta * _minmax(ek) + gamma * _minmax(ei)
    return candidates[int(np.argmin(total))]


def evolve_contour(image_energy, initial_contour, alpha, beta, gamma, max_iterations=300,
                   search_radius=1, convergence_threshold=0.05, synchronous=True):
    """Greedy snake. Returns (final_contour, history). synchronous=True: neighbours come from the
    previous sweep; False: points are updated in place (sequential)."""
    contour = np.asarray(initial_contour, dtype=np.int64).copy()
    history = [contour.copy()]
    for _ in range(max_iterations):
        d_avg = average_spacing(contour)
        old, new = contour.copy(), contour.copy()
        ref = old if synchronous else new          # `new` aliasing = sequential update
        for i in range(len(contour)):
            cands = generate_candidates(ref[i], search_radius, image_energy.shape)
            new[i] = select_best_candidate(image_energy, ref, i, cands, alpha, beta, gamma, d_avg)
        contour = new
        history.append(contour.copy())
        if np.mean(np.linalg.norm(contour - old, axis=1)) < convergence_threshold:
            break
    return contour, history


# ---- rasterisation, chain code, measurements ----

def bresenham_line(p0, p1):
    r0, c0, r1, c1 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err, pts = dc - dr, []
    while True:
        pts.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr; c0 += sc
        if e2 < dc:
            err += dc; r0 += sr
    return pts


def rasterize_contour(contour) -> np.ndarray:
    """Closed ordered list of 8-connected boundary pixels (start pixel not repeated)."""
    pts = np.rint(np.asarray(contour)).astype(int)
    keep = [0] + [i for i in range(1, len(pts)) if not np.array_equal(pts[i], pts[i - 1])]
    pts = pts[keep]
    if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    boundary = []
    for i in range(len(pts)):
        boundary.extend(bresenham_line(pts[i], pts[(i + 1) % len(pts)])[:-1])
    return np.array(boundary, dtype=np.int64)


# (d_row, d_col) -> Freeman code (0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE)
_DIR_TO_CODE = {(0, 1): 0, (-1, 1): 1, (-1, 0): 2, (-1, -1): 3,
                (0, -1): 4, (1, -1): 5, (1, 0): 6, (1, 1): 7}


def generate_chain_code(contour):
    """Returns (start_pixel, chain_code list, boundary_pixels). Includes the closing step."""
    boundary = rasterize_contour(contour)
    m = len(boundary)
    code = [_DIR_TO_CODE[tuple(int(v) for v in boundary[(i + 1) % m] - boundary[i])] for i in range(m)]
    return tuple(int(v) for v in boundary[0]), code, boundary


def calculate_perimeter(chain_code) -> float:
    """P = N_straight + sqrt(2) * N_diagonal (odd Freeman codes are diagonal)."""
    n_diag = sum(1 for c in chain_code if c % 2 == 1)
    return (len(chain_code) - n_diag) + np.sqrt(2.0) * n_diag


def calculate_area(contour) -> float:
    """Shoelace formula on ordered (row, col) vertices."""
    p = np.asarray(contour, dtype=np.float64)
    r, c = p[:, 0], p[:, 1]
    return 0.5 * abs(np.sum(c * np.roll(r, -1) - np.roll(c, -1) * r))


def draw_contour(image, contour, color=(0, 255, 255), thickness=2, show_points=True) -> np.ndarray:
    """Draws a closed (row, col) contour on a BGR copy of `image` (gray is converted to BGR)."""
    out = np.array(image, dtype=np.uint8, order="C", copy=True)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    xy = np.asarray(contour)[:, ::-1].astype(np.int32)             # (row, col) -> (x, y)
    cv2.polylines(out, [xy.reshape(-1, 1, 2)], True, color, thickness)
    if show_points:
        for x, y in xy:
            cv2.circle(out, (int(x), int(y)), max(2, thickness + 1), (0, 0, 255), -1)
    return out


def snake_segmentation(image, contour_xy, n_points, alpha, beta, gamma, max_iterations=300,
                       search_radius=1, sigma=2.0, convergence_threshold=0.05, synchronous=True):
    """Full pipeline. `contour_xy` = user-drawn [(x, y), ...] in image pixels."""
    smooth = preprocess_image(image, sigma)
    h, w = smooth.shape
    pts = np.asarray(contour_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("Draw at least 3 points for the initial contour.")
    if n_points < 3:
        raise ValueError("Number of contour points must be at least 3.")

    initial = resample_contour(pts[:, ::-1], n_points)             # (x, y) -> (row, col)
    initial[:, 0] = np.clip(initial[:, 0], 0, h - 1)
    initial[:, 1] = np.clip(initial[:, 1], 0, w - 1)

    energy = calculate_image_energy(smooth)
    final, history = evolve_contour(energy, initial, alpha, beta, gamma, max_iterations,
                                    search_radius, convergence_threshold, synchronous)
    start, chain, _ = generate_chain_code(final)
    return {
        "initial": initial, "final": final, "history": history, "iterations": len(history) - 1,
        "smoothed": smooth, "start": start, "chain": chain,
        "perimeter": calculate_perimeter(chain), "area": calculate_area(final),
    }