"""
Coffee Grind Size Analyzer — Core Image Processing

Every particle is measured the same way, so results don't depend on how
densely the grounds are spread:
  1. Blue channel divided by a smooth estimate of the paper brightness
  2. Threshold at 58.8% of the paper (as coffeegrindsize does) — about half-way
     between coffee and paper, so blurred edges are neither grown nor shrunk
  3. Distance-transform watershed splits touching grains
  4. Grains cut off by the image edge or the coin are dropped
  5. Optional: MobileSAM refines larger grains (pip install -r requirements-sam.txt)
"""

import cv2
import numpy as np
import warnings
from dataclasses import dataclass
from pathlib import Path
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

warnings.filterwarnings("ignore")

@dataclass
class AnalysisResult:
    """Results from a grind size analysis.

    Sizes are in mm when calibrated (pixel_scale > 0), otherwise in px.
    Percentiles, mean, std and fines/boulders are weighted by `weighting`.
    """
    diameters_mm: np.ndarray   # equivalent-circle diameter of each particle
    areas_px: np.ndarray
    volumes_mm3: np.ndarray    # ellipsoid volume of each particle (px³ if uncalibrated)
    weights: np.ndarray        # each particle's weight under `weighting`
    weighting: str
    num_particles: int
    pixel_scale: float
    labeled_image: np.ndarray
    threshold_image: np.ndarray
    median_diameter_mm: float  # weighted median, same as d50_mm
    mean_diameter_mm: float
    std_diameter_mm: float
    d10_mm: float
    d50_mm: float
    d90_mm: float
    d50_se_mm: float           # bootstrap standard error of D50 (particle counting noise)
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

# Fines / boulders are defined relative to D50, so they mean the same thing for
# every grind size and whether or not the image is calibrated.
FINES_RATIO = 0.5     # fines:    smaller than half of D50
BOULDERS_RATIO = 2.0  # boulders: larger than twice D50

# Larger photos are downscaled. Resolution limits accuracy, so keep this high.
MAX_IMAGE_DIM = 4000

# A pixel is coffee if its blue value is below this fraction of the paper's
DEFAULT_THRESHOLD = 0.588

# Touching grains are split where the neck between them is narrower than this
# fraction of the smaller grain's half-width
SPLIT_RATIO = 0.5

# coffeegrindsize's extraction model: water reaches ~0.1 mm into each grain
EXTRACTION_DEPTH_MM = 0.1

WEIGHTINGS = {
    "available mass": "Mass that water can reach (outer 0.1 mm of each grain), coffeegrindsize's default",
    "mass":           "Grain volume, comparable to sieve or laser-diffraction results",
    "surface":        "Projected grain area",
    "count":          "Every grain counts equally, so fines dominate",
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

    import torch
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


def prepare_image(image_bgr: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    """Downscale very large images so their longest side is at most max_dim."""
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return image_bgr
    scale = max_dim / max(h, w)
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _normalize_to_paper(image_bgr: np.ndarray, exclude: np.ndarray = None) -> np.ndarray:
    """Blue channel divided by a smooth estimate of the local paper brightness (paper ≈ 1)."""
    blue = image_bgr[:, :, 0]
    h, w = blue.shape

    # The paper estimate is smooth, so compute it at quarter resolution
    small = cv2.resize(blue, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA).astype(np.float32)
    keep = np.ones(small.shape, bool)
    if exclude is not None:
        keep = cv2.resize(exclude, small.shape[::-1], interpolation=cv2.INTER_NEAREST) == 0
    otsu, _ = cv2.threshold(small.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper_mask = (small > otsu) & keep

    # Normalized convolution: average only nearby paper pixels, so grains and
    # the coin don't drag the estimate down, whatever their size
    sigma = max(small.shape) / 50
    for _ in range(2):
        weight = cv2.GaussianBlur(paper_mask.astype(np.float32), (0, 0), sigma)
        paper = cv2.GaussianBlur(small * paper_mask, (0, 0), sigma) / np.maximum(weight, 1e-3)
        paper_mask = (small > 0.8 * paper) & keep

    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR)
    return blue.astype(np.float32) / np.maximum(paper, 1.0)


def _segment_threshold(image_bgr, coin_mask=None, threshold=DEFAULT_THRESHOLD,
                       min_area_px=5, split_touching=True, split_ratio=SPLIT_RATIO):
    """
    Threshold the paper-normalized blue channel and split touching grains.
    Returns particles whose "segmentation" is cropped to "bbox" (x0, y0, x1, y1),
    plus the full binary mask.
    """
    h, w = image_bgr.shape[:2]
    binary = _normalize_to_paper(image_bgr, coin_mask) < threshold
    if coin_mask is not None:
        binary &= coin_mask == 0

    components, n_components = ndi.label(binary, structure=np.ones((3, 3)))
    if split_touching:
        # One marker per distance-transform peak whose neck to a neighbouring peak
        # is narrower than split_ratio × its own half-width. On log distance this
        # is scale-free, so big irregular grains aren't chopped up at their bumps.
        dist = ndi.distance_transform_edt(binary)
        log_dist = np.log(np.maximum(dist, 1.0))
        markers, n_markers = ndi.label(h_maxima(log_dist, -np.log(split_ratio)), structure=np.ones((3, 3)))
        # Components without a peak (e.g. 1-px-wide specks) keep a single marker
        has_marker = np.zeros(n_components + 1, bool)
        has_marker[components[markers > 0]] = True
        missing = binary & ~has_marker[components]
        markers[missing] = components[missing] + n_markers
        labels = watershed(-dist, markers, mask=binary, connectivity=2)
        # h_maxima keeps both of two equal-height peaks, cutting one grain in two;
        # re-apply the neck rule to the regions themselves
        labels = _merge_wide_necks(labels, dist, split_ratio)
    else:
        labels = components

    # Grains cut off by the image edge or the coin would read too small
    cut = np.zeros((h, w), bool)
    cut[[0, -1], :] = True
    cut[:, [0, -1]] = True
    if coin_mask is not None:
        cut |= cv2.dilate(coin_mask, np.ones((3, 3), np.uint8)) > 0
    rejected = set(np.unique(labels[cut & (labels > 0)]).tolist())

    particles = []
    max_area = h * w * 0.02  # larger = shadow or background, not a grain
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None or i in rejected:
            continue
        mask = labels[sl] == i
        area = int(np.count_nonzero(mask))
        if area < min_area_px or area > max_area:
            continue
        particles.append({
            "segmentation": mask,
            "bbox": (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop),
            "area": area,
            "source": "threshold",
        })

    return particles, binary


def _merge_wide_necks(labels: np.ndarray, dist: np.ndarray, split_ratio: float) -> np.ndarray:
    """
    Merge neighbouring regions whose shared boundary (neck) is wider than
    split_ratio × the smaller region's peak half-width: they are one grain.
    """
    n = int(labels.max())
    if n < 2:
        return labels
    peak = ndi.maximum(dist, labels, index=np.arange(n + 1))

    # Every horizontally / vertically adjacent pixel pair that crosses a boundary
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    d = np.concatenate([np.minimum(dist[:, :-1], dist[:, 1:]).ravel(),
                        np.minimum(dist[:-1, :], dist[1:, :]).ravel()])
    sel = (a != b) & (a > 0) & (b > 0)
    if not sel.any():
        return labels
    lo, hi, d = np.minimum(a[sel], b[sel]), np.maximum(a[sel], b[sel]), d[sel]

    # Neck half-width of each touching pair = largest distance value on its boundary
    key = lo.astype(np.int64) * (n + 1) + hi
    order = np.argsort(key, kind="stable")
    pairs, start = np.unique(key[order], return_index=True)
    neck = np.maximum.reduceat(d[order], start)
    lo, hi = pairs // (n + 1), pairs % (n + 1)
    merge = neck > split_ratio * np.minimum(peak[lo], peak[hi])

    parent = np.arange(n + 1)
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(lo[merge], hi[merge]):
        parent[root(x)] = root(y)
    return np.array([root(i) for i in range(n + 1)])[labels]


def _crop_to_mask(mask):
    """Crop a full-image boolean mask to its bounding box. Returns (crop, bbox)."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    # copy() so the full-size SAM output can be freed
    return mask[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def _refine_with_sam(image_bgr, particles, coin_mask=None, min_area_px=50):
    """
    Replace the threshold mask of every particle of at least min_area_px with a
    MobileSAM mask (point + box prompt). Which particles get refined depends
    only on their own size, never on how many others are in the photo.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total_area = h * w

    predictor = _get_sam_predictor()
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    refined = []
    for p in particles:
        if p["area"] < min_area_px:
            refined.append(p)
            continue

        bx0, by0, bx1, by1 = p["bbox"]
        ys, xs = np.nonzero(p["segmentation"])
        point = np.array([[bx0 + xs.mean(), by0 + ys.mean()]], dtype=np.float32)
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
        except Exception:
            refined.append(p)  # SAM failed — keep threshold mask
            continue

        mask = masks[np.argmax(scores)]
        area = int(np.sum(mask))

        # Validate: not too big, not background, darker than surroundings
        if area > total_area * 0.02 or area < 2:
            refined.append(p)
            continue
        crop, bbox = _crop_to_mask(mask)
        if not _is_dark_particle(crop, bbox, gray, coin_mask):
            refined.append(p)
            continue

        refined.append({"segmentation": crop, "bbox": bbox, "area": area, "source": "sam"})

    return refined


def _is_dark_particle(mask, bbox, gray, coin_mask=None, pad=5):
    """Check that a mask region (cropped to bbox) is darker than its surroundings."""
    area = np.count_nonzero(mask)
    if area < 2:
        return False

    # Work in a padded window so the dilation below (reach 4 px) has room to grow
    x0, y0, x1, y1 = bbox
    h, w = gray.shape
    wx0, wy0 = max(0, x0 - pad), max(0, y0 - pad)
    wx1, wy1 = min(w, x1 + pad), min(h, y1 + pad)
    local = np.zeros((wy1 - wy0, wx1 - wx0), dtype=bool)
    local[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0] = mask
    gray_local = gray[wy0:wy1, wx0:wx1]

    # Skip if overlaps with coin
    if coin_mask is not None:
        overlap = np.count_nonzero(local & (coin_mask[wy0:wy1, wx0:wx1] > 0))
        if overlap > area * 0.3:
            return False

    mean_inside = np.mean(gray_local[local])

    # Dilate to get surroundings
    seg_uint8 = local.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(seg_uint8, kernel, iterations=2)
    surround = (dilated > 0) & (~local)

    if np.sum(surround) < 5:
        # Can't determine surroundings — accept if dark enough globally
        return mean_inside < 180

    mean_outside = np.mean(gray_local[surround])

    # Particle must be noticeably darker than surroundings
    return mean_inside < mean_outside * 0.95


def _axis_ratio(mask) -> float:
    """Minor/major axis ratio of the ellipse with the same second moments."""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 1.0
    # + 1/12: the variance of a single pixel, so 1-px-wide grains aren't flat
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))) + 1 / 12
    return float(np.clip(np.sqrt(ev[0] / ev[1]), 0.1, 1.0))


def _draw_outlines(image_bgr: np.ndarray, masks: list) -> np.ndarray:
    """Draw randomly coloured outlines (so split grains stand out) on the image."""
    output = image_bgr.copy()
    rng = np.random.RandomState(42)
    for mask in masks:
        # 1-px zero border so particles touching the crop edge get closed contours
        seg = np.pad(mask["segmentation"], 1).astype(np.uint8) * 255
        x0, y0 = mask["bbox"][:2]
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                       offset=(int(x0) - 1, int(y0) - 1))
        color = rng.randint(80, 255, size=3).tolist()
        cv2.drawContours(output, contours, -1, color, 1)
    return output


def _make_threshold_image(masks: list, h: int, w: int) -> np.ndarray:
    """Create a binary mask showing all detected particles."""
    binary = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        x0, y0, x1, y1 = mask["bbox"]
        binary[y0:y1, x0:x1][mask["segmentation"]] = 255
    return binary


def _refine_coin_edge(gray: np.ndarray, cx: float, cy: float, r: float, n_rays: int = 360):
    """
    Measure a candidate coin's outer edge along n_rays rays. On each ray the edge
    is where the profile rises through half-way between the coin rim and the
    paper (the same half-way rule used for grains), taking the crossing nearest
    the candidate radius so grains just outside the coin are ignored. An ellipse
    is fitted, since a tilted phone sees the coin slightly oval.
    Returns ((cx, cy), (axis1, axis2), angle, consistency) or None, where
    consistency is the fraction of rays within 1% of the fitted edge.
    """
    radii = np.arange(0.75 * r, 1.4 * r, 0.5)
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    xs = cx + np.outer(np.cos(angles), radii)
    ys = cy + np.outer(np.sin(angles), radii)
    profiles = ndi.map_coordinates(gray, [ys, xs], order=1, mode="nearest")

    paper = np.percentile(profiles[:, radii >= 1.2 * r], 75, axis=1)
    rim = profiles[:, (radii >= 0.85 * r) & (radii <= 1.1 * r)].min(axis=1)
    half = (paper + rim) / 2
    points = []
    for i in range(n_rays):
        if paper[i] - rim[i] < 15:  # no visible edge on this ray
            continue
        below = profiles[i] < half[i]
        rises = np.flatnonzero(below[:-1] & ~below[1:])  # dark → light going outward
        if rises.size == 0:
            continue
        j = rises[np.argmin(np.abs(radii[rises] - r))]
        # Interpolate where the profile crosses half-way between samples j and j+1
        p0, p1 = profiles[i, j], profiles[i, j + 1]
        rho = radii[j] + (half[i] - p0) / (p1 - p0) * (radii[j + 1] - radii[j])
        points.append((cx + rho * np.cos(angles[i]), cy + rho * np.sin(angles[i])))

    if len(points) < n_rays * 0.5:
        return None
    points = np.array(points, dtype=np.float32)

    # Robust ellipse fit: drop points far from the current fit (grains on the rim, glare)
    keep = np.ones(len(points), bool)
    for _ in range(6):
        if keep.sum() < 5:
            return None
        (ex, ey), (a1, a2), angle = cv2.fitEllipse(points[keep])
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        dx, dy = points[:, 0] - ex, points[:, 1] - ey
        u, v = dx * c + dy * s, -dx * s + dy * c
        resid = np.abs(np.sqrt((2 * u / a1) ** 2 + (2 * v / a2) ** 2) - 1) * np.sqrt(a1 * a2) / 2
        keep = resid < max(1.5, 3 * 1.4826 * np.median(resid[keep]))
    consistency = float(np.sum(resid < max(1.5, 0.01 * np.sqrt(a1 * a2) / 2)) / n_rays)
    return (ex, ey), (a1, a2), angle, consistency


def find_reference_circle(image_bgr: np.ndarray) -> tuple:
    """Try to auto-detect a circular reference object. Returns (diameter_px, coin_mask)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find candidates on a ~1000 px copy (fast); the edge fit restores full precision
    f = min(1.0, 1000 / max(h, w))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (9, 9), 2)
    s = min(small.shape)
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(s * 0.1),
        param1=100, param2=40, minRadius=int(s * 0.03), maxRadius=int(s * 0.4),
    )
    if circles is None:
        return 0.0, None

    # The coin is the candidate whose edge is consistent all the way round;
    # clusters of grains can form Hough circles but not a clean edge
    gray_f = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    best = None
    for x, y, r in circles[0][:10]:
        fit = _refine_coin_edge(gray_f, x / f, y / f, r / f)
        if fit is not None and fit[3] >= 0.6 and (best is None or fit[3] > best[3]):
            best = fit
    if best is None:
        return 0.0, None

    center, (a1, a2), angle, _ = best
    coin_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(coin_mask, (center, (a1 * 1.1, a2 * 1.1), angle), 255, -1)

    # Equivalent-area diameter: the right average scale if the coin looks oval
    return float(np.sqrt(a1 * a2)), coin_mask


def detect_reference_circle(image_bgr: np.ndarray, known_diameter_mm: float) -> tuple:
    """Try to auto-detect a circular reference object. Returns (mm_per_px, coin_mask)."""
    diameter_px, coin_mask = find_reference_circle(image_bgr)
    if diameter_px <= 0:
        return 0.0, None
    return float(known_diameter_mm / diameter_px), coin_mask


def segment_particles(
    image_bgr: np.ndarray,
    coin_mask: np.ndarray = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_area_px: int = 5,
    use_sam: bool = False,
) -> list:
    """The slow part of the pipeline (independent of calibration scale and weighting)."""
    particles, _ = _segment_threshold(image_bgr, coin_mask, threshold, min_area_px)
    if use_sam:
        particles = _refine_with_sam(image_bgr, particles, coin_mask)
    return particles


def particle_sizes(particle_masks: list, pixel_scale: float = 0.0) -> tuple:
    """
    Per-particle (diameters, areas_px, volumes). Diameter is the equivalent-circle
    diameter; volume treats each grain as an ellipsoid whose hidden (vertical)
    axis equals its short axis, as coffeegrindsize does. Units are mm / mm³ when
    pixel_scale > 0, otherwise px / px³.
    """
    areas_px = np.array([m["area"] for m in particle_masks], dtype=np.float64)
    ratios = np.array([_axis_ratio(m["segmentation"]) for m in particle_masks], dtype=np.float64)

    scale = pixel_scale if pixel_scale > 0 else 1.0
    diameters = 2.0 * np.sqrt(areas_px / np.pi) * scale
    semi_major = np.sqrt(areas_px / (np.pi * np.maximum(ratios, 1e-6))) * scale
    semi_minor = semi_major * ratios
    volumes = 4.0 / 3.0 * np.pi * semi_major * semi_minor ** 2
    return diameters, areas_px, volumes


def particle_weights(areas_px, volumes, weighting: str, pixel_scale: float = 0.0) -> tuple:
    """Per-particle weights for a weighting in WEIGHTINGS. Returns (weights, weighting_used).

    "available mass" needs real units, so it falls back to "mass" when uncalibrated.
    """
    if weighting == "available mass" and pixel_scale <= 0:
        weighting = "mass"
    if weighting == "count":
        return np.ones_like(areas_px), weighting
    if weighting == "surface":
        return np.asarray(areas_px, dtype=np.float64), weighting
    if weighting == "mass":
        return np.asarray(volumes, dtype=np.float64), weighting
    if weighting == "available mass":
        # Equivalent-sphere radius; only the outer EXTRACTION_DEPTH_MM shell is reachable
        radii = (3.0 * volumes / (4.0 * np.pi)) ** (1 / 3)
        core = 4.0 / 3.0 * np.pi * np.maximum(radii - EXTRACTION_DEPTH_MM, 0) ** 3
        return volumes - core, weighting
    raise ValueError(f"Unknown weighting {weighting!r}; expected one of {list(WEIGHTINGS)}")


def weighted_percentile(values, weights, q):
    """Percentile(s) q (0–100) of values under weights (midpoint-rule CDF)."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(np.asarray(q) / 100.0, cdf, v)


def size_stats(diameters, weights, n_boot: int = 200, seed: int = 0) -> dict:
    """Weighted D10/D50/D90, mean, std, fines/boulders % of weight, and D50 standard error."""
    if len(diameters) == 0:
        return dict(d10=0.0, d50=0.0, d90=0.0, mean=0.0, std=0.0, d50_se=0.0,
                    fines_pct=0.0, boulders_pct=0.0)
    diameters, weights = np.asarray(diameters), np.asarray(weights)
    d10, d50, d90 = (float(v) for v in weighted_percentile(diameters, weights, [10, 50, 90]))
    mean = float(np.average(diameters, weights=weights))
    std = float(np.sqrt(np.average((diameters - mean) ** 2, weights=weights)))
    total = np.sum(weights)

    # Counting noise: how much D50 moves when the same photo's grains are resampled
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diameters), len(diameters))
        boots.append(weighted_percentile(diameters[idx], weights[idx], 50))

    return dict(
        d10=d10, d50=d50, d90=d90, mean=mean, std=std,
        d50_se=float(np.std(boots)),
        fines_pct=float(np.sum(weights[diameters < FINES_RATIO * d50]) / total * 100),
        boulders_pct=float(np.sum(weights[diameters > BOULDERS_RATIO * d50]) / total * 100),
    )


def measure_particles(
    image_bgr: np.ndarray,
    particle_masks: list,
    pixel_scale: float = 0.0,
    weighting: str = "available mass",
) -> AnalysisResult:
    """Measure segmented particles and compute weighted statistics (fast)."""
    h, w = image_bgr.shape[:2]

    diameters, areas_px, volumes = particle_sizes(particle_masks, pixel_scale)
    weights, weighting = particle_weights(areas_px, volumes, weighting, pixel_scale)
    stats = size_stats(diameters, weights)

    return AnalysisResult(
        diameters_mm=diameters,
        areas_px=areas_px,
        volumes_mm3=volumes,
        weights=weights,
        weighting=weighting,
        num_particles=int(len(particle_masks)),
        pixel_scale=pixel_scale,
        labeled_image=_draw_outlines(image_bgr, particle_masks),
        threshold_image=_make_threshold_image(particle_masks, h, w),
        median_diameter_mm=stats["d50"],
        mean_diameter_mm=stats["mean"],
        std_diameter_mm=stats["std"],
        d10_mm=stats["d10"],
        d50_mm=stats["d50"],
        d90_mm=stats["d90"],
        d50_se_mm=stats["d50_se"],
        fines_pct=stats["fines_pct"],
        boulders_pct=stats["boulders_pct"],
    )


def analyze_image(
    image_bgr: np.ndarray,
    pixel_scale: float = 0.0,
    min_area_px: int = 5,
    coin_mask: np.ndarray = None,
    weighting: str = "available mass",
    use_sam: bool = False,
) -> AnalysisResult:
    """Full pipeline: segment_particles() then measure_particles()."""
    particle_masks = segment_particles(image_bgr, coin_mask, min_area_px=min_area_px, use_sam=use_sam)
    return measure_particles(image_bgr, particle_masks, pixel_scale, weighting)
