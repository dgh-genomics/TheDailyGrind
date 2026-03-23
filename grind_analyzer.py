"""
Coffee Grind Size Analyzer — Core Image Processing
Hybrid approach: CLAHE threshold finds candidate particles, SAM refines each one.
"""

import cv2
import numpy as np
import torch
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

@dataclass
class AnalysisResult:
    """Results from a grind size analysis."""
    diameters_mm: np.ndarray
    areas_px: np.ndarray
    num_particles: int
    pixel_scale: float
    labeled_image: np.ndarray
    threshold_image: np.ndarray
    median_diameter_mm: float
    mean_diameter_mm: float
    std_diameter_mm: float
    d10_mm: float
    d50_mm: float
    d90_mm: float
    fines_pct: float
    boulders_pct: float


REFERENCE_OBJECTS = {
    "US Quarter":       24.26,
    "US Nickel":        21.21,
    "US Penny":         19.05,
    "US Dime":          17.91,
    "Canadian Quarter": 23.88,
    "Canadian Nickel":  21.20,
    "1 Euro":           23.25,
    "2 Euro":           25.75,
    "Custom…":          None,
}

# ── Lazy-loaded SAM predictor (singleton) ─────────────────────────────────
_sam_predictor = None

MODEL_URL = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"

def _download_model(dest: Path):
    """Download MobileSAM weights if not already present."""
    if dest.exists():
        return
    import urllib.request
    print(f"Downloading MobileSAM weights ({dest.name})...")
    urllib.request.urlretrieve(MODEL_URL, str(dest))
    print("Download complete.")

def _get_sam_predictor():
    """Load MobileSAM predictor once, reuse across calls."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    from mobile_sam import sam_model_registry, SamPredictor

    model_path = Path(__file__).parent / "mobile_sam.pt"
    _download_model(model_path)

    torch.set_default_dtype(torch.float32)
    device = "cpu"

    sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
    sam.to(device)
    sam.eval()

    _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def _find_candidate_points(image_bgr: np.ndarray, coin_mask: np.ndarray = None):
    """
    Use CLAHE + adaptive threshold to find ALL dark regions (candidate particles).
    Returns centroids and bounding boxes for each candidate.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # CLAHE to boost local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    # Two-pass adaptive threshold to catch both large and small particles
    block1 = max(51, (min(h, w) // 12) | 1)  # larger block for bigger particles
    block2 = max(15, (min(h, w) // 40) | 1)  # smaller block for fines

    thresh1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block1, 4)
    thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block2, 3)

    # Union of both passes
    binary = cv2.bitwise_or(thresh1, thresh2)

    # Remove coin region
    if coin_mask is not None:
        binary[coin_mask > 0] = 0

    # Light morphological cleanup — just remove single-pixel noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates = []
    total_area = h * w
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter: too small = noise, too large = shadow/background
        if area < 3:
            continue
        if area > total_area * 0.02:
            continue

        cx, cy = centroids[i]
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        candidates.append({
            "centroid": (int(cx), int(cy)),
            "bbox": (bx, by, bx + bw, by + bh),
            "area": area,
            "component_mask": (labels == i),
        })

    return candidates, binary


def _refine_with_sam(image_bgr, candidates, coin_mask=None, use_sam=True, max_sam_candidates=500):
    """
    For each candidate, either use SAM to get a precise mask,
    or fall back to the threshold mask for tiny particles.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_area = h * w

    particle_masks = []

    if use_sam and len(candidates) > 0:
        predictor = _get_sam_predictor()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # Sort by area descending — SAM works better on larger particles
        # Use SAM for the largest candidates, threshold for the rest
        candidates_sorted = sorted(candidates, key=lambda c: c["area"], reverse=True)

        # Batch point prompts for SAM
        sam_candidates = candidates_sorted[:max_sam_candidates]
        threshold_candidates = candidates_sorted[max_sam_candidates:]

        # Process SAM candidates in batches
        for cand in sam_candidates:
            cx, cy = cand["centroid"]
            bx0, by0, bx1, by1 = cand["bbox"]

            # Use both point + box prompt for best results
            point = np.array([[cx, cy]], dtype=np.float32)
            label = np.array([1], dtype=np.int32)  # 1 = foreground

            # Add padding to bbox
            pad = max(5, int(max(bx1 - bx0, by1 - by0) * 0.15))
            box = np.array([
                max(0, bx0 - pad), max(0, by0 - pad),
                min(w, bx1 + pad), min(h, by1 + pad)
            ], dtype=np.float32)

            try:
                masks, scores, _ = predictor.predict(
                    point_coords=point,
                    point_labels=label,
                    box=box[None, :],  # SAM expects [1, 4]
                    multimask_output=True,
                )

                # Pick best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                area = int(np.sum(mask))

                # Validate: not too big, not background
                if area > total_area * 0.02 or area < 2:
                    # Fall back to threshold mask
                    mask = cand["component_mask"]
                    area = cand["area"]

                # Validate: must be darker than surroundings
                if not _is_dark_particle(mask, gray, coin_mask):
                    continue

                particle_masks.append({
                    "segmentation": mask,
                    "area": area,
                    "source": "sam",
                })
            except Exception:
                # SAM failed — use threshold mask
                mask = cand["component_mask"]
                if _is_dark_particle(mask, gray, coin_mask):
                    particle_masks.append({
                        "segmentation": mask,
                        "area": cand["area"],
                        "source": "threshold",
                    })

        # Add threshold-only candidates (too many for SAM)
        for cand in threshold_candidates:
            mask = cand["component_mask"]
            if _is_dark_particle(mask, gray, coin_mask):
                particle_masks.append({
                    "segmentation": mask,
                    "area": cand["area"],
                    "source": "threshold",
                })
    else:
        # No SAM — pure threshold
        for cand in candidates:
            mask = cand["component_mask"]
            if _is_dark_particle(mask, gray, coin_mask):
                particle_masks.append({
                    "segmentation": mask,
                    "area": cand["area"],
                    "source": "threshold",
                })

    return particle_masks


def _is_dark_particle(mask, gray, coin_mask=None):
    """Check that a mask region is darker than its surroundings."""
    if np.sum(mask) < 2:
        return False

    # Skip if overlaps with coin
    if coin_mask is not None:
        overlap = np.sum(mask & (coin_mask > 0))
        if overlap > np.sum(mask) * 0.3:
            return False

    mask_pixels = gray[mask]
    mean_inside = np.mean(mask_pixels)

    # Dilate to get surroundings
    seg_uint8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(seg_uint8, kernel, iterations=2)
    surround = (dilated > 0) & (~mask)

    if np.sum(surround) < 5:
        # Can't determine surroundings — accept if dark enough globally
        return mean_inside < 180

    mean_outside = np.mean(gray[surround])

    # Particle must be noticeably darker than surroundings
    return mean_inside < mean_outside * 0.95


def _draw_outlines(image_bgr: np.ndarray, masks: list) -> np.ndarray:
    """Draw colored outlines on the original image for each detected particle."""
    output = image_bgr.copy()
    rng = np.random.RandomState(42)
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Color by source: cyan for SAM, green for threshold-only
        if mask.get("source") == "sam":
            color = rng.randint(80, 255, size=3).tolist()
        else:
            color = [0, 255, 128]  # green for threshold-detected
        cv2.drawContours(output, contours, -1, color, 1)
    return output


def _make_threshold_image(masks: list, h: int, w: int) -> np.ndarray:
    """Create a binary mask showing all detected particles."""
    binary = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        binary[mask["segmentation"]] = 255
    return binary


def detect_reference_circle(image_bgr: np.ndarray, known_diameter_mm: float) -> tuple:
    """Try to auto-detect a circular reference object."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape

    min_r = int(w * 0.02)
    max_r = int(w * 0.15)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(w * 0.1),
        param1=100, param2=40, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return 0.0, None

    best = circles[0][0]
    cx, cy, radius_px = int(best[0]), int(best[1]), best[2]
    diameter_px = 2 * radius_px
    pixel_scale = known_diameter_mm / diameter_px

    coin_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(coin_mask, (cx, cy), int(radius_px * 1.1), 255, -1)

    return float(pixel_scale), coin_mask


def analyze_image(
    image_bgr: np.ndarray,
    pixel_scale: float = 0.0,
    min_area_px: int = 3,
    coin_mask: np.ndarray = None,
) -> AnalysisResult:
    """
    Hybrid analysis pipeline:
      1. CLAHE + adaptive threshold to find ALL dark candidate regions
      2. SAM predictor refines each candidate with point+box prompts
      3. Threshold fallback for tiny particles / overflow
      4. Measure and compute statistics
    """
    h, w = image_bgr.shape[:2]

    # 1. Find candidates via threshold
    candidates, binary_preview = _find_candidate_points(image_bgr, coin_mask)

    # 2. Refine with SAM (point-prompted) + threshold fallback
    particle_masks = _refine_with_sam(image_bgr, candidates, coin_mask,
                                       use_sam=True, max_sam_candidates=500)

    # 3. Measure
    areas_px = np.array([m["area"] for m in particle_masks], dtype=np.float64)
    if len(areas_px) == 0:
        areas_px = np.array([0.0])

    diameters_px = 2.0 * np.sqrt(areas_px / np.pi)

    if pixel_scale > 0:
        diameters_mm = diameters_px * pixel_scale
    else:
        diameters_mm = diameters_px

    # 4. Visuals
    labeled = _draw_outlines(image_bgr, particle_masks)
    threshold = _make_threshold_image(particle_masks, h, w)

    # 5. Stats
    if len(diameters_mm) == 0 or (len(diameters_mm) == 1 and diameters_mm[0] == 0):
        diameters_mm = np.array([0.0])

    result = AnalysisResult(
        diameters_mm=diameters_mm,
        areas_px=areas_px,
        num_particles=int(len(particle_masks)),
        pixel_scale=pixel_scale,
        labeled_image=labeled,
        threshold_image=threshold,
        median_diameter_mm=float(np.median(diameters_mm)),
        mean_diameter_mm=float(np.mean(diameters_mm)),
        std_diameter_mm=float(np.std(diameters_mm)),
        d10_mm=float(np.percentile(diameters_mm, 10)),
        d50_mm=float(np.percentile(diameters_mm, 50)),
        d90_mm=float(np.percentile(diameters_mm, 90)),
        fines_pct=float(np.mean(diameters_mm < np.percentile(diameters_mm, 10)) * 100),
        boulders_pct=float(np.mean(diameters_mm > np.percentile(diameters_mm, 90)) * 100),
    )
    return result
