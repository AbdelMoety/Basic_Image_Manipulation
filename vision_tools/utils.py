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

def _edge_magnitude(image: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Convolves with horizontal and vertical gradient kernels: sqrt(Gx^2 + Gy^2)."""
    gx = convolve(image, kx, clip=False)
    gy = convolve(image, ky, clip=False)
    magnitude = np.hypot(gx, gy)
    return _process_matrix(magnitude)


def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    return _edge_magnitude(image, kx, ky)


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