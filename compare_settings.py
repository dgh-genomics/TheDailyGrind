"""
Compare grinder settings: can photos tell neighbouring settings apart?

Put the photos in one folder per grinder setting (names sort numerically):

    photos/
      12/  IMG_0001.jpg  IMG_0002.jpg  ...
      13/  ...
      14/  grind_a/IMG_0101.jpg  grind_b/IMG_0107.jpg  ...   <- optional: one
                                                               subfolder per separate grind

    python compare_settings.py photos/ --coin "US Quarter" --out results/

For each size weighting it reports:
  - per setting: mean D50 and its photo-to-photo SD, next to the counting noise
    of a single photo (so you can see whether more grains per photo would help)
  - for each pair of neighbouring settings: the D50 difference in units of the
    photo-to-photo SD ("separation"), how often one photo would be assigned to
    the wrong one of the pair, and how many photos per measurement would push
    the separation to 3 (about 7% mix-ups)
  - leave-one-out: how often each photo lands on its true setting (or within ±1)

Sizes come from single grains (clumps of touching grains are left out). A final
block repeats the available-mass analysis with D50 adjusted to no clumping, using
how D50 moves with clumping within settings in this experiment (or the
provisional default when clumping barely varies).

Each photo is checked for flash / ultra-wide lens (EXIF), clumping and uneven
light; warnings are printed and listed in the report.

Writes to --out: photos.csv (one row per photo, incl. clumped_pct,
lighting_evenness_pct and warnings), particles/<setting>/... .csv (one row per
single grain), and report.html (plots).

Note: repeat photos of one spread share the same grains, so they understate the
spread-to-spread scatter; use separate spreads (and grinds) as replicates.

HEIC photos (iPhone default) must be converted to JPEG first.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import grind_analyzer as ga

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MIN_PARTICLES = 50     # photos with fewer grains are skipped
TARGET_SEPARATION = 3.0


def setting_key(name: str):
    """Sort settings numerically when they look like numbers (2, 10, 10.5), else by name."""
    try:
        return (0, float(name), "")
    except ValueError:
        return (1, 0.0, name)


def find_photos(root: Path) -> list:
    photos = []
    for setting_dir in sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: setting_key(d.name)):
        for path in sorted(setting_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                rel = path.relative_to(setting_dir)
                photos.append({
                    "setting": setting_dir.name,
                    "sample": rel.parts[0] if len(rel.parts) > 1 else "",
                    "path": path,
                })
    return photos


def measure_photo(path: Path, args):
    """Segment one photo. Returns (mm_per_px, clumped_pct, evenness, diameters, areas_px, volumes) or None."""
    image = cv2.imread(str(path))
    if image is None:
        print(f"  ! could not read {path}")
        return None
    image = ga.prepare_image(image, args.max_dim)

    if args.mm_per_px:
        scale, coin_mask = args.mm_per_px, None
    else:
        coin_px, coin_mask = ga.find_reference_circle(image)
        if coin_px <= 0:
            print(f"  ! no coin found in {path.name}; skipped (or pass --mm-per-px)")
            return None
        scale = args.coin_mm / coin_px

    particles = ga.segment_particles(image, coin_mask, threshold=args.threshold, use_sam=args.sam)
    # Sizes come from single grains; clumps (merged touching grains) only count toward clumped_pct
    clumps = ga.clump_flags(particles)
    singles = [p for p, c in zip(particles, clumps) if not c]
    return (scale, ga.clumped_percent(particles, clumps), ga.lighting_evenness(image, coin_mask),
            *ga.particle_sizes(singles, scale))


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pooled_sd(groups: list) -> float:
    """Pooled standard deviation over groups of values (groups of 1 add nothing)."""
    dof = sum(len(g) - 1 for g in groups if len(g) > 1)
    if dof == 0:
        return float("nan")
    return math.sqrt(sum((len(g) - 1) * np.var(g, ddof=1) for g in groups if len(g) > 1) / dof)


def clump_coefficient(rows: list, key: str) -> tuple:
    """
    Slope of log(key) on clumped_pct within settings (pooled), so it is not
    confounded with the settings themselves. Falls back to the provisional
    default when clumping barely varies within settings.
    """
    xs, ys = [], []
    for setting in dict.fromkeys(r["setting"] for r in rows):
        sel = [r for r in rows if r["setting"] == setting]
        if len(sel) > 1:
            x = np.array([r["clumped_pct"] for r in sel])
            y = np.log([r[key] for r in sel])
            xs.append(x - x.mean())
            ys.append(y - y.mean())
    if xs:
        x, y = np.concatenate(xs), np.concatenate(ys)
        dof = len(x) - len(xs)
        if dof >= 6 and np.sum(x ** 2) >= 50:
            return float(np.sum(x * y) / np.sum(x ** 2)), f"fitted on this experiment, {dof} within-setting d.o.f."
    return ga.CLUMP_ADJUST_PER_PCT, "provisional default; clumping varies too little within settings here"


def summarize(rows: list, weighting: str) -> dict:
    """Per-setting D50 statistics and neighbour separation for one weighting."""
    key = f"d50_{weighting}"
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    d50 = {s: np.array([r[key] for r in rows if r["setting"] == s]) for s in settings}
    se = {s: np.array([r[f"d50_se_{weighting}"] for r in rows if r["setting"] == s]) for s in settings}

    per_setting = [dict(
        setting=s, n=len(d50[s]), mean=float(d50[s].mean()),
        sd=float(np.std(d50[s], ddof=1)) if len(d50[s]) > 1 else float("nan"),
        se=float(se[s].mean()),
    ) for s in settings]
    sd = pooled_sd(list(d50.values()))

    pairs = []
    for a, b in zip(per_setting, per_setting[1:]):
        sep = abs(b["mean"] - a["mean"]) / sd if sd > 0 else float("nan")
        pairs.append(dict(
            a=a["setting"], b=b["setting"], delta=b["mean"] - a["mean"], separation=sep,
            mixup=normal_cdf(-sep / 2) if not math.isnan(sep) else float("nan"),
            photos_needed=math.ceil((TARGET_SEPARATION / sep) ** 2) if sep > 0 else None,
        ))

    # Leave-one-out: assign each photo to the setting whose mean (without it) is closest
    exact = within_one = total = 0
    for i, r in enumerate(rows):
        means = {}
        for s in settings:
            others = [rr[key] for j, rr in enumerate(rows) if rr["setting"] == s and j != i]
            if others:
                means[s] = np.mean(others)
        if r["setting"] not in means:
            continue
        guess = min(means, key=lambda s: abs(means[s] - r[key]))
        total += 1
        exact += guess == r["setting"]
        within_one += abs(settings.index(guess) - settings.index(r["setting"])) <= 1

    # With grind subfolders: split photo-to-photo SD into same-grind and grind-to-grind parts
    nested = None
    if any(r["sample"] for r in rows):
        grinds = {}
        for r in rows:
            grinds.setdefault((r["setting"], r["sample"]), []).append(r[key])
        grind_means = [[np.mean(v) for (s, _), v in grinds.items() if s == setting] for setting in settings]
        nested = dict(same_grind_sd=pooled_sd(list(grinds.values())), grind_to_grind_sd=pooled_sd(grind_means))

    mean_se2 = np.mean([r[f"d50_se_{weighting}"] ** 2 for r in rows])
    return dict(
        weighting=weighting, per_setting=per_setting, sd=sd, pairs=pairs, nested=nested,
        loo_exact=exact / total if total else float("nan"),
        loo_within_one=within_one / total if total else float("nan"),
        counting_share=float(mean_se2 / sd ** 2) if sd > 0 else float("nan"),
    )


def format_summary(s: dict, unit: str) -> str:
    lines = [f"== {s['weighting']} ==",
             f"{'setting':>10} {'photos':>6}   {'D50 mean ± photo SD':>22} {'CV':>6}   {'counting SE':>11}"]
    for p in s["per_setting"]:
        cv = f"{100 * p['sd'] / p['mean']:5.1f}%" if p["n"] > 1 else "   n/a"
        sd = f"{p['sd']:.3f}" if p["n"] > 1 else "  n/a"
        lines.append(f"{p['setting']:>10} {p['n']:>6}   {p['mean']:>10.3f} ± {sd} {unit:<3} {cv}   {p['se']:>11.3f}")
    lines.append(f"pooled photo-to-photo SD: {s['sd']:.3f} {unit}; (counting SE / photo SD)² = "
                 f"{s['counting_share']:.2f}  (≈1: more grains per photo would help; "
                 f"≪1: spread, lighting or focus dominate)")
    if s["nested"]:
        lines.append(f"  same-grind photo SD {s['nested']['same_grind_sd']:.3f}, "
                     f"grind-to-grind SD {s['nested']['grind_to_grind_sd']:.3f} {unit}")
    lines.append(f"\n{'neighbours':>16} {'ΔD50':>8} {'separation':>11} {'1-photo mix-ups':>16} "
                 f"{f'photos for {TARGET_SEPARATION:g}×':>15}")
    for p in s["pairs"]:
        needed = str(p["photos_needed"]) if p["photos_needed"] else "n/a"
        lines.append(f"{p['a'] + ' → ' + p['b']:>16} {p['delta']:>+8.3f} {p['separation']:>10.1f}× "
                     f"{100 * p['mixup']:>15.0f}% {needed:>15}")
    lines.append(f"leave-one-out: {100 * s['loo_exact']:.0f}% of photos on the right setting, "
                 f"{100 * s['loo_within_one']:.0f}% within ±1")
    return "\n".join(lines)


def make_report(rows: list, particles: dict, summaries: list, best: str, unit: str, text: str) -> str:
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    x_of = {s: i for i, s in enumerate(settings)}
    rng = np.random.default_rng(0)

    strip = make_subplots(rows=1, cols=len(summaries), subplot_titles=[s["weighting"] for s in summaries])
    for c, s in enumerate(summaries, start=1):
        key = f"d50_{s['weighting']}"
        strip.add_trace(go.Scatter(
            x=[x_of[r["setting"]] + rng.uniform(-0.15, 0.15) for r in rows], y=[r[key] for r in rows],
            mode="markers", marker=dict(size=6, color="#c9a96e", opacity=0.7),
            text=[f"{r['setting']} {r['photo']}" for r in rows], hoverinfo="text+y", showlegend=False,
        ), row=1, col=c)
        strip.add_trace(go.Scatter(
            x=list(range(len(settings))), y=[p["mean"] for p in s["per_setting"]],
            error_y=dict(array=[p["sd"] if p["n"] > 1 else 0 for p in s["per_setting"]]),
            mode="lines+markers", marker=dict(size=9, color="#ff6b6b"), line=dict(color="#ff6b6b", width=1),
            showlegend=False,
        ), row=1, col=c)
        strip.update_xaxes(tickvals=list(range(len(settings))), ticktext=settings, title_text="setting",
                           row=1, col=c)
    strip.update_yaxes(title_text=f"D50 ({unit})", row=1, col=1)
    strip.update_layout(height=420, template="plotly_white",
                        title="D50 per photo (dots) and setting mean ± photo-to-photo SD")

    cdf = go.Figure()
    colors = sample_colorscale("Viridis", [i / max(1, len(settings) - 1) for i in range(len(settings))])
    for s, color in zip(settings, colors):
        d = np.concatenate([p[0] for p in particles[s]])
        a = np.concatenate([p[1] for p in particles[s]])
        v = np.concatenate([p[2] for p in particles[s]])
        w, _ = ga.particle_weights(a, v, best.split(",")[0], pixel_scale=1.0)  # sizes are already in mm
        order = np.argsort(d)
        cdf.add_trace(go.Scatter(x=d[order], y=np.cumsum(w[order]) / w.sum() * 100, mode="lines",
                                 name=f"setting {s}", line=dict(color=color, width=2)))
    cdf.update_layout(height=420, template="plotly_white", xaxis_type="log",
                      title=f"Pooled cumulative distribution per setting ({best} weighting)",
                      xaxis_title=f"diameter ({unit})", yaxis_title=f"cumulative % of {best}")

    return ("<!doctype html><html><head><meta charset='utf-8'><title>Grinder setting comparison</title></head>"
            "<body style='font-family:system-ui;margin:24px'>"
            "<h2>Grinder setting comparison</h2>"
            + strip.to_html(full_html=False, include_plotlyjs=True)
            + cdf.to_html(full_html=False, include_plotlyjs=False)
            + f"<pre style='font-size:13px'>{text}</pre></body></html>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", type=Path, help="folder with one subfolder per grinder setting")
    ap.add_argument("--out", type=Path, default=Path("results"), help="output folder (default: results)")
    coins = [k for k, v in ga.REFERENCE_OBJECTS.items() if v]
    ap.add_argument("--coin", default="US Quarter", choices=coins, help="reference coin in every photo")
    ap.add_argument("--coin-mm", type=float, help="reference diameter in mm (overrides --coin)")
    ap.add_argument("--mm-per-px", type=float, help="fixed scale for a camera on a stand (no coin needed)")
    ap.add_argument("--max-dim", type=int, default=ga.MAX_IMAGE_DIM, help="downscale longer side to this")
    ap.add_argument("--threshold", type=float, default=ga.DEFAULT_THRESHOLD,
                    help="coffee if blue < this fraction of the paper (default %(default)s)")
    ap.add_argument("--sam", action="store_true", help="refine larger grains with MobileSAM")
    args = ap.parse_args()
    args.coin_mm = args.coin_mm or ga.REFERENCE_OBJECTS[args.coin]
    unit = "mm"

    photos = find_photos(args.photos)
    if not photos:
        raise SystemExit(f"No photos found in subfolders of {args.photos}")
    print(f"{len(photos)} photos in {len(set(p['setting'] for p in photos))} settings")

    rows, particles = [], {}
    for photo in photos:
        measured = measure_photo(photo["path"], args)
        if measured is None:
            continue
        scale, clumped, evenness, d, a, v = measured
        if len(d) < MIN_PARTICLES:
            print(f"  ! only {len(d)} grains in {photo['path'].name}; skipped")
            continue
        warnings = (ga.capture_warnings(photo["path"]) + ga.spread_warnings(clumped)
                    + ga.lighting_warnings(evenness))
        for warning in warnings:
            print(f"  ! {photo['path'].name}: {warning}")
        row = dict(setting=photo["setting"], sample=photo["sample"], photo=photo["path"].name,
                   n_particles=len(d), mm_per_px=round(scale, 6), clumped_pct=round(clumped, 1),
                   lighting_evenness_pct=round(100 * evenness), warnings=" | ".join(warnings))
        for weighting in ga.WEIGHTINGS:
            w, _ = ga.particle_weights(a, v, weighting, scale)
            stats = ga.size_stats(d, w)
            for k in ("d10", "d50", "d90", "d50_se"):
                row[f"{k}_{weighting}"] = round(stats[k], 5)
        rows.append(row)
        particles.setdefault(photo["setting"], []).append((d, a, v))
        print(f"  {photo['setting']:>8} {photo['path'].name:<28} {len(d):6d} grains  "
              f"D50 (available mass) {row['d50_available mass']:.3f} ± {row['d50_se_available mass']:.3f} mm")

        out_csv = args.out / "particles" / photo["setting"] / photo["sample"] / f"{photo['path'].stem}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["diameter_mm", "area_px", "volume_mm3"])
            writer.writerows(zip(np.round(d, 5), a.astype(int), np.round(v, 7)))

    if len(set(r["setting"] for r in rows)) < 2:
        raise SystemExit("Need usable photos from at least two settings to compare.")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "photos.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = [summarize(rows, w) for w in ga.WEIGHTINGS]

    # Clump covariate: how D50 moves with clumping within each setting, from this
    # experiment if clumping varies enough, else the provisional default
    coef, source = clump_coefficient(rows, "d50_available mass")
    adjusted = "available mass, clump-adjusted"
    for r in rows:
        factor = math.exp(-coef * r["clumped_pct"])
        r[f"d50_{adjusted}"] = round(r["d50_available mass"] * factor, 5)
        r[f"d50_se_{adjusted}"] = round(r["d50_se_available mass"] * factor, 5)
    summaries.append(summarize(rows, adjusted))
    text = (f"Clump adjustment: D50 changes {100 * coef:+.2f}% per point of clumping ({source}); "
            "the last block shows available-mass D50 adjusted to no clumping.\n\n")
    text += "\n\n".join(format_summary(s, unit) for s in summaries)
    seps = {s["weighting"]: np.nanmedian([p["separation"] for p in s["pairs"]]) for s in summaries}
    best = max(seps, key=lambda k: -1 if math.isnan(seps[k]) else seps[k])
    verdict = (f"Best weighting: {best} (median neighbour separation {seps[best]:.1f}× the photo-to-photo SD; "
               f"aim for ≥ {TARGET_SEPARATION:g}).")
    text += "\n\n" + verdict
    flagged = [r["photo"] for r in rows if r["warnings"]]
    if flagged:
        text += (f"\n\n{len(flagged)} of {len(rows)} photos have warnings (flash, ultra-wide lens, clumped "
                 f"grains or uneven light) and are included above: {', '.join(flagged)}. Their sizes are "
                 "biased; consider reshooting them. See the warnings column in photos.csv.")
    print("\n" + text)

    (args.out / "report.html").write_text(make_report(rows, particles, summaries, best, unit, text))
    print(f"\nWrote {args.out / 'photos.csv'}, {args.out / 'report.html'} and per-grain CSVs.")


if __name__ == "__main__":
    main()
