import shutil
import tempfile

import cv2
import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from . import sift

SIFT_IDS = ["sift_scale_space", "sift_keypoints", "sift_match_ssd", "sift_match_ncc"]


def textured_image(size=256, seed=3):
    """Deterministic blobby texture: SIFT finds plenty of stable keypoints in it."""
    rng = np.random.default_rng(seed)
    noise = rng.random((size, size)).astype(np.float32)
    img = cv2.GaussianBlur(noise, (0, 0), 3) + 0.5 * cv2.GaussianBlur(noise, (0, 0), 8)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)


def rotated_scaled(img, angle=25, scale=0.8):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def png_upload(img, name="img.png"):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return SimpleUploadedFile(name, buf.tobytes(), content_type="image/png")


class SiftAlgorithmTests(TestCase):
    def test_pipeline_shapes(self):
        kps, desc, gauss, dog, timings = sift.sift(textured_image())
        self.assertGreater(len(kps), 20)
        self.assertEqual(desc.shape, (len(kps), 128))
        self.assertTrue(np.allclose(np.linalg.norm(desc, axis=1), 1.0, atol=1e-4))
        cfg = sift.SiftConfig()
        self.assertEqual(len(gauss[0]), cfg.S + 3)
        self.assertEqual(len(dog[0]), cfg.S + 2)

    def test_rotated_scaled_copy_matches_geometrically(self):
        img = textured_image()
        _, d1, *_ = sift.sift(img)
        _, d2, *_ = sift.sift(rotated_scaled(img))
        for method in ("ssd", "ncc"):
            matches = sift.match(d1, d2, method, ratio=0.8)
            self.assertGreater(len(matches), 15, method)

    def test_match_edge_cases(self):
        empty = np.zeros((0, 128), np.float32)
        one = np.ones((1, 128), np.float32)
        self.assertEqual(sift.match(empty, one), [])
        self.assertEqual(sift.match(one, one), [])      # needs >= 2 candidates for the ratio test

    def test_too_small_image(self):
        with self.assertRaises(ValueError):
            sift.sift(np.zeros((16, 16), np.uint8))

    def test_featureless_image_has_no_keypoints(self):
        kps, desc, *_ = sift.sift(np.full((64, 64), 120, np.uint8))
        self.assertEqual(len(kps), 0)
        self.assertEqual(desc.shape, (0, 128))


class SiftApiTests(TestCase):
    def setUp(self):
        self.media = tempfile.mkdtemp()
        self.override = override_settings(MEDIA_ROOT=self.media)
        self.override.enable()
        self.img = textured_image()
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)
        self.addCleanup(self.override.disable)

    def post(self, op, second=None, **extra):
        data = {"filter_type": op, "image": png_upload(self.img), **extra}
        if second is not None:
            data["image_2"] = png_upload(second, "second.png")
        return self.client.post("/api/process/", data)

    def test_sift_section_is_in_the_operations_list(self):
        ops = self.client.get("/api/operations/").json()["operations"]
        sift_ops = [o for o in ops if o["group"] == "Feature Matching (SIFT)"]
        self.assertEqual([o["id"] for o in sift_ops], SIFT_IDS)
        # nothing else was disturbed
        self.assertEqual(len({o["id"] for o in ops}), len(ops))

    def test_single_image_operations(self):
        for op in ("sift_scale_space", "sift_keypoints"):
            res = self.post(op)
            self.assertEqual(res.status_code, 200, res.content)
            body = res.json()
            self.assertIn("processed_image", body, op)
            self.assertTrue(body["info"])

    def test_matching_with_second_image(self):
        for op in ("sift_match_ssd", "sift_match_ncc"):
            res = self.post(op, second=rotated_scaled(self.img))
            self.assertEqual(res.status_code, 200, res.content)
            body = res.json()
            self.assertIn("processed_image", body)
            self.assertIn("uploaded_image_2", body)
            self.assertIn("RANSAC inliers", body["info"])
            for piece in ("image 1: pyramid", "image 2: pyramid", "keypoints", "descriptors",
                          f"{op[-3:].upper()} matching", "total"):
                self.assertIn(piece, body["info"], (op, piece))

    def test_matching_needs_second_image(self):
        res = self.post("sift_match_ssd")
        self.assertEqual(res.status_code, 400)
        self.assertIn("second image", res.json()["error"])

    def test_matching_can_reuse_uploaded_ids(self):
        up1 = self.client.post("/api/upload/", {"image": png_upload(self.img)}).json()["image_id"]
        up2 = self.client.post(
            "/api/upload/", {"image": png_upload(rotated_scaled(self.img), "b.png")}
        ).json()["image_id"]
        res = self.client.post(
            "/api/process/", {"filter_type": "sift_match_ncc", "image_id": up1, "image_2_id": up2}
        )
        self.assertEqual(res.status_code, 200, res.content)

    def test_parameter_validation(self):
        for extra in ({"sift_ratio": "1.5"}, {"sift_contrast": "-1"}, {"sift_edge_ratio": "0.5"},
                      {"sift_max_size": "5000"}, {"sift_max_size": "10"}):
            res = self.post("sift_keypoints", **extra)
            self.assertEqual(res.status_code, 400, extra)

    def test_tiny_image_gives_clean_error(self):
        res = self.client.post(
            "/api/process/",
            {"filter_type": "sift_keypoints", "image": png_upload(np.zeros((20, 20), np.uint8))},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("too small", res.json()["error"])

    def test_existing_operations_still_work(self):
        res = self.post("sobel")
        self.assertEqual(res.status_code, 200, res.content)
        self.assertIn("processed_image", res.json())
