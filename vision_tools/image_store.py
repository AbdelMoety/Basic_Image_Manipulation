"""
Upload-once image store + small in-memory caches.

Why: before this module every "Process" request re-uploaded the file, saved a new copy to
disk, re-decoded it, re-converted it to grayscale and re-ran Canny from scratch -- even when
the user only changed one slider or applied a second operation to the same photo.

Now:
  * an image is saved ONCE under a content-hash id (identical uploads are de-duplicated);
  * later requests send only that id;
  * the decoded BGR image, its grayscale version, the Canny edge map and the gradient
    angles are kept in byte-bounded LRU caches and reused across requests.
"""
import glob
import hashlib
import os
import re
import threading
from collections import OrderedDict

import cv2
from django.core.files.base import ContentFile

from . import utils

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class _LRU:
    """Thread-safe LRU cache bounded by total array bytes."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._d = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        n = value.nbytes
        if n > self.max_bytes:
            return
        with self._lock:
            if key in self._d:
                self._size -= self._d.pop(key).nbytes
            self._d[key] = value
            self._size += n
            while self._size > self.max_bytes and self._d:
                _, old = self._d.popitem(last=False)
                self._size -= old.nbytes


_IMAGES = _LRU(256 * 1024 * 1024)   # decoded BGR / gray images
_RESULTS = _LRU(256 * 1024 * 1024)  # Canny edges / gradient angles


# ---------------------------------------------------------------- storage
def save_upload(fs, uploaded_file):
    """Save an upload once. Returns (image_id, path, url). Same bytes -> same id, no new file."""
    data = b"".join(uploaded_file.chunks())
    image_id = hashlib.sha256(data).hexdigest()[:32]

    found = find(fs, image_id)
    if found:
        return (image_id, *found)

    ext = os.path.splitext(uploaded_file.name or "")[1].lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext):
        ext = ".png"
    saved = fs.save(f"src_{image_id}{ext}", ContentFile(data))
    return image_id, fs.path(saved), fs.url(saved)


def find(fs, image_id):
    """(path, url) of a previously uploaded image, or None. The id is validated (no path tricks)."""
    if not isinstance(image_id, str) or not _ID_RE.match(image_id):
        return None
    matches = glob.glob(os.path.join(fs.location, f"src_{image_id}.*"))
    if not matches:
        return None
    name = os.path.basename(matches[0])
    return fs.path(name), fs.url(name)


def resolve(fs, files, post, file_key, id_key):
    """Prefer the id (no upload needed); fall back to a freshly uploaded file."""
    image_id = post.get(id_key)
    if image_id:
        found = find(fs, image_id)
        if found:
            return (image_id, *found)
    if files.get(file_key):
        return save_upload(fs, files[file_key])
    return None


# ---------------------------------------------------------------- cached loaders
def load_bgr(image_id, path):
    """Decoded image (cv2.imread) -- decoded once, then served from memory. Returns a copy."""
    key = ("bgr", image_id)
    img = _IMAGES.get(key)
    if img is None:
        img = cv2.imread(path)
        if img is None:
            return None
        _IMAGES.put(key, img)
    return img.copy()


def load_gray(image_id, bgr):
    """utils.ensure_gray_from_bgr(bgr), computed once per image. Returns a copy."""
    key = ("gray", image_id)
    g = _IMAGES.get(key)
    if g is None:
        g = utils.ensure_gray_from_bgr(bgr)
        _IMAGES.put(key, g)
    return g.copy()


# ---------------------------------------------------------------- cached intermediates
def get_edges(image_id, image, low, high, sigma, kernel_size):
    """Manual Canny edge map. Re-used when only Hough/draw parameters change."""
    key = ("edges", image_id, float(low), float(high), float(sigma), int(kernel_size))
    edges = _RESULTS.get(key)
    if edges is None:
        edges = utils.manual_canny(image, low, high, sigma, kernel_size)
        _RESULTS.put(key, edges)
    return edges.copy()


def get_theta(image_id, image, sigma, kernel_size):
    """Gradient angles of the smoothed image (needed by the circle/ellipse transforms)."""
    key = ("theta", image_id, float(sigma), int(kernel_size))
    theta = _RESULTS.get(key)
    if theta is None:
        gray = utils.ensure_gray_from_bgr(image)
        smoothed = utils.gaussian_filter(gray, kernel_size, sigma)
        _, _, _, theta = utils.sobel_edge_detection(smoothed, process=False)
        _RESULTS.put(key, theta)
    return theta.copy()
