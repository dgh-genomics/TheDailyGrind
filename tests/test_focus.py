"""
Focus check (grind_analyzer.edge_width / focus_warnings) on synthetic photos and
on the repo's sample photo.

Run from the repo root with `python tests/test_focus.py` (or `pytest tests`).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import grind_analyzer as ga  # noqa: E402

FIELD_MM = 30.0  # side of the synthetic square of paper
CELL_MM = 2.0    # one grain per cell, so most grains have paper around them


def synthetic_photo(mm_per_px: float, blur_mm: float = 0.0, seed: int = 0) -> np.ndarray:
    """
    Dark, irregular 0.3–1.4 mm grains on white paper, laid out in mm so the same
    grains can be photographed at any scale. Edges are anti-aliased (drawn at 4×,
    then averaged down). blur_mm is a Gaussian blur's sigma, in mm.
    """
    rng = np.random.default_rng(seed)
    side = int(round(FIELD_MM / mm_per_px))
    up = 4
    mask = np.zeros((side * up, side * up), np.uint8)
    n_cells = int(FIELD_MM / CELL_MM)
    for i in range(n_cells):
        for j in range(n_cells):
            radius = rng.uniform(0.15, 0.7)
            cx, cy = (np.array([i, j]) + 0.5 + rng.uniform(-0.15, 0.15, 2)) * CELL_MM
            angles = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(5, 9)))
            r = radius * rng.uniform(0.7, 1.2, len(angles))
            pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], 1) / mm_per_px * up
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    coffee = cv2.resize(mask, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    paper, grain = np.array([215, 222, 228], np.float32), np.array([45, 60, 85], np.float32)
    image = paper * (1 - coffee[..., None]) + grain * coffee[..., None]
    if blur_mm > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_mm / mm_per_px)
    image += rng.normal(0, 2, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def edge_width_mm(image: np.ndarray, mm_per_px: float) -> float:
    return ga.edge_width(image, ga.segment_particles(image)) * mm_per_px


def test_sharp_photo_passes():
    width = edge_width_mm(synthetic_photo(0.04), 0.04)
    assert width < 0.5 * ga.FOCUS_WARN_MM, width
    assert ga.focus_warnings(width / 0.04, 0.04) == []


def test_blurry_photo_is_flagged():
    image = synthetic_photo(0.04, blur_mm=0.2)
    width_px = ga.edge_width(image, ga.segment_particles(image))
    assert width_px * 0.04 > ga.FOCUS_WARN_MM, width_px * 0.04
    warnings = ga.focus_warnings(width_px, 0.04)
    assert len(warnings) == 1 and "out of focus" in warnings[0]


def test_judged_per_mm_not_per_px():
    # The same blurry grains photographed from twice as far: the edge width in px
    # halves, but in mm (what the check uses) it stays about the same
    near, far = 0.03, 0.06
    width_near = edge_width_mm(synthetic_photo(near, blur_mm=0.15), near)
    width_far = edge_width_mm(synthetic_photo(far, blur_mm=0.15), far)
    assert abs(width_far / width_near - 1) < 0.15, (width_near, width_far)
    assert (width_near / near) / (width_far / far) > 1.6


def test_no_verdict_without_scale_or_grains():
    assert ga.focus_warnings(50.0, 0.0) == []  # uncalibrated: px can't be judged
    assert ga.focus_warnings(float("nan"), 0.04) == []
    blank = np.full((400, 400, 3), 220, np.uint8)
    assert np.isnan(ga.edge_width(blank, ga.segment_particles(blank)))


def test_sample_photo_passes_and_blurred_copy_is_flagged():
    # A 1200 px patch of grounds above the coin in examples/sample_grounds.jpg
    # (0.0550 mm/px from its coin; about 1,000 particles)
    mm_per_px = 0.055
    patch = cv2.imread(str(REPO / "examples" / "sample_grounds.jpg"))[1000:2200, 900:2100]
    assert ga.focus_warnings(ga.edge_width(patch, ga.segment_particles(patch)), mm_per_px) == []
    blurred = cv2.GaussianBlur(patch, (0, 0), 0.2 / mm_per_px)
    assert ga.focus_warnings(ga.edge_width(blurred, ga.segment_particles(blurred)), mm_per_px)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
