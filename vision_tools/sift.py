"""
SIFT feature detection + SSD / NCC matching, ported from ``sift_matching.ipynb``.

Same algorithm and the same simplifications as the notebook (no sub-pixel refinement, no image
doubling, every pixel votes into one cell / one bin). The only change is *how* it is computed:
the notebook visits every pixel of every keypoint window in a Python ``for`` loop, which is fine in
Colab but far too slow for a web request, so those loops are replaced by NumPy array operations.
The results are the same (see ``tests.py``, which checks the port against the notebook's maths).

  * Gaussian scale space, DoG, extrema + contrast/edge rejection, orientation, 128-D descriptor: own code
  * Gaussian blur, drawing, RANSAC sanity check: OpenCV
"""
import time
from dataclasses import dataclass

import cv2
import numpy as np

NUM_BINS_ORI = 36   # orientation histogram bins
CELLS = 4           # descriptor window is CELLS x CELLS cells ...
ORI_BINS = 8        # ... with an ORI_BINS-bin orientation histogram each -> 4*4*8 = 128-D
INITIAL_BLUR = 0.5  # assumed blur already present in the camera image

MIN_SIDE = 32       # smaller images leave fewer than 2 octaves
MAX_SIZE_LIMIT = 1000


@dataclass(frozen=True)
class SiftConfig:
    S: int = 3                       # scales per octave in which keypoints are searched
    sigma0: float = 1.6              # blur of the first image of every octave (Lowe)
    contrast_thresh: float = 0.01    # minimum |DoG| (image is in [0, 1])
    edge_ratio: float = 10.0         # Lowe's r for edge rejection

    @property
    def k(self) -> float:
        return 2.0 ** (1.0 / self.S)


# ---------------------------------------------------------------- preparation
def shrink(img: np.ndarray, max_size: int) -> np.ndarray:
    """Shrink big images so the keypoint/matching stages stay fast (never enlarges)."""
    scale = max_size / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def num_octaves_for(img: np.ndarray) -> int:
    return int(np.log2(min(img.shape[:2]))) - 3     # stop when the image gets too small


# ---------------------------------------------------------------- scale space
def compute_sigmas(cfg: SiftConfig):
    return [cfg.sigma0 * (cfg.k ** i) for i in range(cfg.S + 3)]


def build_gaussian_pyramid(img: np.ndarray, num_octaves: int, cfg: SiftConfig):
    """S + 3 progressively blurred images per octave; each octave starts from the image with
    2 * sigma0, down-sampled by 2. Blurring an already blurred image by sigma_diff gives
    sqrt(sigma_prev^2 + sigma_diff^2), so we only ever blur by the *difference*."""
    sigmas = compute_sigmas(cfg)
    pyramid = []

    first_diff = np.sqrt(cfg.sigma0 ** 2 - INITIAL_BLUR ** 2)
    base = cv2.GaussianBlur(img, (0, 0), sigmaX=first_diff)

    for _ in range(num_octaves):
        octave_images = [base]
        for i in range(1, cfg.S + 3):
            sigma_diff = np.sqrt(sigmas[i] ** 2 - sigmas[i - 1] ** 2)
            octave_images.append(cv2.GaussianBlur(octave_images[-1], (0, 0), sigmaX=sigma_diff))
        pyramid.append(octave_images)
        base = np.ascontiguousarray(octave_images[cfg.S][::2, ::2])
    return pyramid


def build_dog_pyramid(gaussian_pyramid):
    """Difference of Gaussians = difference of two neighbouring Gaussian images."""
    return [
        [octave_images[i + 1] - octave_images[i] for i in range(len(octave_images) - 1)]
        for octave_images in gaussian_pyramid
    ]


# ---------------------------------------------------------------- keypoint detection
def _extremum_mask(below: np.ndarray, mid: np.ndarray, above: np.ndarray) -> np.ndarray:
    """True where a pixel (border excluded) is >= or <= all 26 neighbours (8 + 9 + 9).
    Same non-strict test as the notebook (`center == cube.max() or center == cube.min()`)."""
    h, w = mid.shape
    center = mid[1:-1, 1:-1]
    hi = np.full(center.shape, -np.inf, dtype=mid.dtype)
    lo = np.full(center.shape, np.inf, dtype=mid.dtype)
    for layer in (below, mid, above):
        for dy in range(3):
            for dx in range(3):
                if layer is mid and dy == 1 and dx == 1:
                    continue
                nb = layer[dy:h - 2 + dy, dx:w - 2 + dx]
                np.maximum(hi, nb, out=hi)
                np.minimum(lo, nb, out=lo)
    return (center >= hi) | (center <= lo)


def _not_edge(d: np.ndarray, ys: np.ndarray, xs: np.ndarray, edge_ratio: float) -> np.ndarray:
    """Edge rejection with the 2x2 Hessian: keep points with tr(H)^2 / det(H) < (r+1)^2 / r."""
    c = d[ys, xs]
    dxx = d[ys, xs + 1] + d[ys, xs - 1] - 2 * c
    dyy = d[ys + 1, xs] + d[ys - 1, xs] - 2 * c
    dxy = (d[ys + 1, xs + 1] - d[ys + 1, xs - 1] - d[ys - 1, xs + 1] + d[ys - 1, xs - 1]) / 4.0
    trace = dxx + dyy
    det = dxx * dyy - dxy ** 2
    keep = det > 0                                     # det <= 0: curvatures differ in sign -> edge
    safe_det = np.where(keep, det, 1)
    return keep & (trace ** 2 / safe_det < (edge_ratio + 1) ** 2 / edge_ratio)


def detect_keypoints(dog_pyramid, cfg: SiftConfig):
    keypoints = []   # each keypoint: dict(octave, layer, y, x)
    for o, dog_octave in enumerate(dog_pyramid):
        for layer in range(1, cfg.S + 1):              # only layers with a neighbour above and below
            mid = dog_octave[layer]
            if min(mid.shape) < 3:
                continue
            candidate = np.zeros(mid.shape, dtype=bool)
            candidate[1:-1, 1:-1] = (
                _extremum_mask(dog_octave[layer - 1], mid, dog_octave[layer + 1])
                & (np.abs(mid[1:-1, 1:-1]) > cfg.contrast_thresh)
            )
            ys, xs = np.nonzero(candidate)             # row-major, like the notebook's np.where
            if len(ys) == 0:
                continue
            keep = _not_edge(mid, ys, xs, cfg.edge_ratio)
            for y, x in zip(ys[keep], xs[keep]):
                keypoints.append(dict(octave=o, layer=layer, y=int(y), x=int(x)))
    return keypoints


# ---------------------------------------------------------------- orientation
def compute_gradients(img: np.ndarray):
    """Gradient magnitude and direction (degrees, 0-360) of a Gaussian image."""
    gy, gx = np.gradient(img)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.degrees(np.arctan2(gy, gx)) % 360
    return magnitude, angle


def _window(y0: int, x0: int, radius: int, h: int, w: int):
    """Square window around (y0, x0), clipped to the pixels that have all gradient neighbours."""
    y_lo, y_hi = max(y0 - radius, 1), min(y0 + radius, h - 2)
    x_lo, x_hi = max(x0 - radius, 1), min(x0 + radius, w - 2)
    if y_lo > y_hi or x_lo > x_hi:
        return None
    dy = np.arange(y_lo, y_hi + 1)[:, None] - y0
    dx = np.arange(x_lo, x_hi + 1)[None, :] - x0
    return (slice(y_lo, y_hi + 1), slice(x_lo, x_hi + 1)), dy, dx


def assign_orientations(keypoints, gaussian_pyramid, cfg: SiftConfig):
    """36-bin histogram of gradient directions (weighted by magnitude and a Gaussian window of
    1.5 * sigma). The highest peak gives the orientation; other peaks >= 80 % of it add extra
    keypoints at the same place, as in the original paper."""
    grad_cache = {}
    oriented = []

    for kp in keypoints:
        o, layer, y0, x0 = kp["octave"], kp["layer"], kp["y"], kp["x"]
        key = (o, layer)
        if key not in grad_cache:
            grad_cache[key] = compute_gradients(gaussian_pyramid[o][layer])
        magnitude, angle = grad_cache[key]
        h, w = magnitude.shape

        weight_sigma = 1.5 * cfg.sigma0 * (cfg.k ** layer)
        radius = int(round(3 * weight_sigma))

        win = _window(y0, x0, radius, h, w)
        if win is None:
            continue
        sl, dy, dx = win
        weight = np.exp(-(dx ** 2 + dy ** 2) / (2 * weight_sigma ** 2))
        bins = (angle[sl] / 360 * NUM_BINS_ORI).astype(np.int64) % NUM_BINS_ORI
        hist = np.bincount(bins.ravel(), weights=(weight * magnitude[sl]).ravel(),
                           minlength=NUM_BINS_ORI)

        for _ in range(2):                              # circular smoothing
            hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3

        max_val = hist.max()
        if max_val == 0:
            continue
        for b in range(NUM_BINS_ORI):
            left, right = hist[b - 1], hist[(b + 1) % NUM_BINS_ORI]
            if hist[b] > left and hist[b] > right and hist[b] >= 0.8 * max_val:
                new_kp = dict(kp)
                new_kp["angle"] = (b + 0.5) * 360 / NUM_BINS_ORI
                oriented.append(new_kp)
    return oriented


# ---------------------------------------------------------------- descriptor
def compute_descriptors(keypoints, gaussian_pyramid, cfg: SiftConfig):
    """4 x 4 cells x 8 orientation bins = 128 numbers per keypoint. The window is rotated by the
    keypoint angle (rotation invariance) and its cell size is 3 * sigma (scale invariance).
    Normalise, clip at 0.2, normalise again (robust to lighting changes)."""
    grad_cache = {}
    final_keypoints, descriptors = [], []

    for kp in keypoints:
        o, layer, y0, x0 = kp["octave"], kp["layer"], kp["y"], kp["x"]
        key = (o, layer)
        if key not in grad_cache:
            grad_cache[key] = compute_gradients(gaussian_pyramid[o][layer])
        magnitude, angle = grad_cache[key]
        h, w = magnitude.shape

        sigma = cfg.sigma0 * (cfg.k ** layer)
        cell_size = 3 * sigma                           # pixels per cell
        radius = int(cell_size * CELLS / 2 * np.sqrt(2) + 1)

        win = _window(y0, x0, radius, h, w)
        if win is None:
            continue
        sl, dy, dx = win

        theta = np.radians(kp["angle"])
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # rotate the offsets into the keypoint's frame, measured in cells
        x_rot = (cos_t * dx + sin_t * dy) / cell_size
        y_rot = (-sin_t * dx + cos_t * dy) / cell_size
        cx = np.floor(x_rot + CELLS / 2).astype(np.int64)
        cy = np.floor(y_rot + CELLS / 2).astype(np.int64)
        inside = (cx >= 0) & (cx < CELLS) & (cy >= 0) & (cy < CELLS)

        rel_angle = (angle[sl] - kp["angle"]) % 360
        b = (rel_angle / 360 * ORI_BINS).astype(np.int64) % ORI_BINS
        weight = np.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * (CELLS / 2) ** 2))
        flat = (cy * CELLS + cx) * ORI_BINS + b
        desc = np.bincount(flat[inside], weights=(weight * magnitude[sl])[inside],
                           minlength=CELLS * CELLS * ORI_BINS)

        norm = np.linalg.norm(desc)
        if norm == 0:
            continue
        desc = desc / norm
        desc = np.minimum(desc, 0.2)
        desc = desc / np.linalg.norm(desc)

        # position in ORIGINAL image coordinates (octave o is down-sampled by 2^o)
        scale_factor = 2 ** o
        final_keypoints.append(dict(x=x0 * scale_factor, y=y0 * scale_factor,
                                    size=sigma * scale_factor * 2, angle=kp["angle"]))
        descriptors.append(desc)

    return final_keypoints, np.array(descriptors, dtype=np.float32).reshape(-1, CELLS * CELLS * ORI_BINS)


# ---------------------------------------------------------------- full pipeline
def check_size(img: np.ndarray):
    if min(img.shape[:2]) < MIN_SIDE:
        raise ValueError(f"Image is too small for SIFT (the short side must be at least {MIN_SIDE}px).")


def sift(img_gray_uint8: np.ndarray, cfg: SiftConfig = SiftConfig()):
    """Run SIFT on a grayscale uint8 image.
    Returns (keypoints, descriptors, gaussian_pyramid, dog_pyramid, timings)."""
    check_size(img_gray_uint8)
    img = img_gray_uint8.astype(np.float32) / 255.0
    n_oct = num_octaves_for(img)

    t = time.perf_counter()
    gaussian_pyramid = build_gaussian_pyramid(img, n_oct, cfg)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)
    t_pyr = time.perf_counter() - t

    t = time.perf_counter()
    keypoints = detect_keypoints(dog_pyramid, cfg)
    keypoints = assign_orientations(keypoints, gaussian_pyramid, cfg)
    t_det = time.perf_counter() - t

    t = time.perf_counter()
    keypoints, descriptors = compute_descriptors(keypoints, gaussian_pyramid, cfg)
    t_desc = time.perf_counter() - t

    timings = dict(pyramid=t_pyr, detect=t_det, descriptors=t_desc)
    return keypoints, descriptors, gaussian_pyramid, dog_pyramid, timings


# ---------------------------------------------------------------- matching
def match(d1: np.ndarray, d2: np.ndarray, method: str = "ssd", ratio: float = 0.8, chunk: int = 512):
    """Match every descriptor of image 1 to image 2 and keep it only if it passes Lowe's ratio test
    (best / second best < ratio).

    SSD(a, b) = sum (a_i - b_i)^2  -> distance sqrt(SSD)   (smaller = more similar)
    NCC(a, b) = zero-mean, unit-length dot product -> distance 1 - NCC   (closer to 1 = more similar)

    Returns a list of (index in image 1, index in image 2, distance).
    Done in row blocks so memory stays small however many keypoints there are."""
    if method not in ("ssd", "ncc"):
        raise ValueError(method)
    if len(d1) == 0 or len(d2) < 2:
        return []

    a = d1.astype(np.float64)
    b = d2.astype(np.float64)
    if method == "ncc":
        def normalize(d):
            d = d - d.mean(axis=1, keepdims=True)
            return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        a, b = normalize(a), normalize(b)
    else:
        b_sq = (b ** 2).sum(axis=1)

    matches = []
    for start in range(0, len(a), chunk):
        block = a[start:start + chunk]
        if method == "ssd":
            ssd = (block ** 2).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * (block @ b.T)
            dist = np.sqrt(np.maximum(ssd, 0.0))
        else:
            dist = 1.0 - block @ b.T

        two = np.argpartition(dist, 1, axis=1)[:, :2]             # indices of the 2 smallest
        d_two = np.take_along_axis(dist, two, axis=1)
        order = np.argsort(d_two, axis=1)
        two = np.take_along_axis(two, order, axis=1)
        d_two = np.take_along_axis(d_two, order, axis=1)
        best_idx, d_best, d_second = two[:, 0], d_two[:, 0], d_two[:, 1]

        safe = np.where(d_second > 0, d_second, 1.0)
        ok = (d_second > 0) & (d_best / safe < ratio)
        for r in np.nonzero(ok)[0]:
            matches.append((start + int(r), int(best_idx[r]), float(d_best[r])))
    return matches


# ---------------------------------------------------------------- drawing (BGR images for cv2.imwrite)
GREEN, RED, WHITE = (0, 255, 0), (0, 0, 255), (255, 255, 255)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _title_band(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """Put a dark title strip above an image (so a saved result says what it shows)."""
    band_h = 28
    band = np.full((band_h, img_bgr.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(band, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack([band, img_bgr])


def draw_keypoints(img_gray: np.ndarray, kps, title: str = "") -> np.ndarray:
    """Circle = scale, line = orientation."""
    out = _to_bgr(img_gray)
    for kp in kps:
        center = (int(kp["x"]), int(kp["y"]))
        radius = max(int(kp["size"] / 2), 2)
        a = np.radians(kp["angle"])
        end = (int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a)))
        cv2.circle(out, center, radius, GREEN, 1)
        cv2.line(out, center, end, RED, 1)
    return _title_band(out, title) if title else out


def ransac_inliers(kps1, kps2, matches):
    """Quick sanity check: matches that agree with ONE homography are inliers. None if < 4 matches."""
    if len(matches) < 4:
        return None
    pts1 = np.float32([[kps1[i]["x"], kps1[i]["y"]] for i, _, _ in matches])
    pts2 = np.float32([[kps2[j]["x"], kps2[j]["y"]] for _, j, _ in matches])
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else None


def draw_matches(img1, kps1, img2, kps2, matches, label: str, max_draw: int = 60):
    """Side-by-side matches (the `max_draw` best). Returns (BGR image, inliers or None)."""
    cv_kps1 = [cv2.KeyPoint(float(k["x"]), float(k["y"]), float(k["size"])) for k in kps1]
    cv_kps2 = [cv2.KeyPoint(float(k["x"]), float(k["y"]), float(k["size"])) for k in kps2]
    cv_matches = sorted((cv2.DMatch(i, j, d) for i, j, d in matches), key=lambda m: m.distance)[:max_draw]
    inliers = ransac_inliers(kps1, kps2, matches)

    out = cv2.drawMatches(img1, cv_kps1, img2, cv_kps2, cv_matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    extra = f", RANSAC inliers: {inliers}/{len(matches)}" if inliers is not None else ""
    return _title_band(out, f"{label}: {len(matches)} matches{extra}"), inliers


def scale_space_montage(gauss_octave, dog_octave, cfg: SiftConfig, tile: int = 256) -> np.ndarray:
    """Row 1: the S + 3 Gaussian images of an octave; row 2: the S + 2 DoG images (each stretched to
    its own min..max, as matplotlib's imshow does)."""
    def fit(img_u8, caption):
        scale = tile / max(img_u8.shape)
        small = cv2.resize(img_u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_u8
        canvas = np.zeros((tile + 22, tile, 3), dtype=np.uint8)
        canvas[22:22 + small.shape[0], :small.shape[1]] = _to_bgr(small)
        cv2.putText(canvas, caption, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
        return canvas

    sigmas = compute_sigmas(cfg)
    cols = cfg.S + 3
    blank = np.zeros((tile + 22, tile, 3), dtype=np.uint8)

    row1 = [fit(np.clip(g * 255, 0, 255).astype(np.uint8), f"G sigma={sigmas[i]:.2f}")
            for i, g in enumerate(gauss_octave)]
    row2 = []
    for i, d in enumerate(dog_octave):
        lo, hi = float(d.min()), float(d.max())
        stretched = np.zeros(d.shape, np.uint8) if hi == lo else ((d - lo) / (hi - lo) * 255).astype(np.uint8)
        row2.append(fit(stretched, f"DoG {i}"))
    row2 += [blank] * (cols - len(row2))
    return np.vstack([np.hstack(row1), np.hstack(row2)])
