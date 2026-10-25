"""
Measurement log: one JSON line per saved measurement, in a local file.

Each record holds what the photo measured, how it was taken (phone and camera
mode from EXIF) and what the user says about the grind and the cup. It never
stores the photo or its location: camera fields are picked from EXIF by name,
and GPS is not among them. The format is meant to grow into pooled data across
phones, grinders, brewers and beans, so it is versioned (SCHEMA_VERSION).

The log path defaults to measurements.jsonl in the working directory; set the
TDG_MEASUREMENT_LOG environment variable to change it.
"""

import datetime
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

from PIL import ExifTags, Image

import grind_analyzer as ga

SCHEMA_VERSION = 1

BREW_METHODS = ["Drip machine", "Pour-over", "French press / immersion", "AeroPress", "Espresso",
                "Cold brew", "Other"]
TASTES = ["Not tasted", "Sour / thin (under-extracted)", "Balanced", "Bitter", "Astringent / drying",
          "Bitter and astringent"]


def log_path() -> Path:
    return Path(os.environ.get("TDG_MEASUREMENT_LOG", "measurements.jsonl"))


def camera_info(photo_bytes: bytes) -> dict:
    """Phone and camera mode from EXIF. Deliberately excludes GPS and other location data."""
    try:
        with Image.open(io.BytesIO(photo_bytes)) as im:
            base = im.getexif()
            exif = base.get_ifd(ExifTags.IFD.Exif)
    except Exception:
        return {}

    def text(value):
        if value is None:
            return None
        return str(value).strip("\x00 ").strip() or None

    flash = exif.get(ExifTags.Base.Flash)
    focal = exif.get(ExifTags.Base.FocalLengthIn35mmFilm)
    zoom = exif.get(ExifTags.Base.DigitalZoomRatio)
    info = {
        "make": text(base.get(ExifTags.Base.Make)),
        "model": text(base.get(ExifTags.Base.Model)),
        "focal_length_35mm": int(focal) if focal else None,
        "digital_zoom": round(float(zoom), 2) if zoom else None,
        "flash_fired": bool(int(flash) & 1) if flash is not None else None,
        "taken_at": text(exif.get(ExifTags.Base.DateTimeOriginal)),
    }
    return {k: v for k, v in info.items() if v is not None}


def _app_version() -> str:
    """Short git commit of this checkout, so records can be tied to the code that made them."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _round(x, digits=4):
    return None if x is None else round(float(x), digits)


def build_record(photo_bytes: bytes, result: "ga.AnalysisResult", *, coin_diameter_px: float,
                 evenness: float, warnings: list, context: dict) -> dict:
    """
    One measurement as a JSON-ready dict. `context` holds what the user entered
    (bean, grinder, setting, brew method, recipe, taste, TDS, notes).
    """
    calibrated = result.pixel_scale > 0
    d, w = result.diameters_mm, result.weights
    share_under_06 = (100 * w[d < 0.6].sum() / w.sum()) if calibrated and len(d) and w.sum() > 0 else None
    return {
        "schema": SCHEMA_VERSION,
        "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "app_version": _app_version(),
        "photo_sha256": hashlib.sha256(photo_bytes).hexdigest(),
        "camera": camera_info(photo_bytes),
        "capture": {
            "mm_per_px": _round(result.pixel_scale, 6),
            "coin_diameter_px": _round(coin_diameter_px, 1),
            "lighting_evenness_pct": _round(100 * evenness, 1),
            "clumped_pct": _round(result.clumped_pct, 1),
            "grains_measured": int(result.num_particles),
        },
        "size": {
            "weighting": result.weighting,
            "unit": "mm" if calibrated else "px",
            "d10": _round(result.d10_mm),
            "d50": _round(result.d50_mm),
            "d50_se": _round(result.d50_se_mm),
            "d90": _round(result.d90_mm),
            "d50_clump_adjusted": _round(ga.clump_adjusted(result.d50_mm, result.clumped_pct)),
            "fines_pct": _round(result.fines_pct, 2),
            "boulders_pct": _round(result.boulders_pct, 2),
            "share_under_0_6mm_pct": _round(share_under_06, 2),
        },
        "warnings": list(warnings),
        "context": context,
    }


def append_record(record: dict, path: Path = None) -> int:
    """Append one record; returns how many records the log now holds."""
    path = Path(path or log_path())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
