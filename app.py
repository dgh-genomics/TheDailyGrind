"""
☕ Coffee Grind Size Analyzer
Drop a photo of your coffee grounds → get a particle size distribution.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from grind_analyzer import (
    segment_particles,
    measure_particles,
    find_reference_circle,
    prepare_image,
    capture_warnings,
    spread_warnings,
    clump_adjusted,
    lighting_evenness,
    lighting_warnings,
    REFERENCE_OBJECTS,
    FINES_RATIO,
    BOULDERS_RATIO,
    UNDER_DETECTED_PX,
    WEIGHTINGS,
    AnalysisResult,
)
from grind_analyzer import shape_comparison
from measurement_log import BREW_METHODS, TASTES, append_record, build_record, camera_info, describe, log_path, read_records

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coffee Grind Analyzer",
    page_icon="☕",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark coffee-themed styling */
    .block-container { max-width: 1200px; padding-top: 2rem; }
    h1 { color: #c9a96e; }
    .stMetric label { font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #c9a96e44;
        border-radius: 12px;
        padding: 1rem;
    }
    .brew-guide {
        background: #1e1e1e;
        color: #f0e6d6;
        border-left: 4px solid #c9a96e;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Brew guide lookup ────────────────────────────────────────────────────────
BREW_GUIDES = {
    "Turkish":      (0.0, 0.2),
    "Espresso":     (0.2, 0.5),
    "Moka Pot":     (0.4, 0.6),
    "AeroPress":    (0.4, 0.8),
    "Pour Over":    (0.5, 0.8),
    "Drip":         (0.6, 1.0),
    "Chemex":       (0.7, 1.0),
    "French Press": (0.8, 1.2),
    "Cold Brew":    (1.0, 1.5),
}


def classify_grind(median_mm: float) -> str:
    """Return the most likely brew method based on median particle size."""
    if median_mm <= 0:
        return "Unknown"
    for method, (lo, hi) in BREW_GUIDES.items():
        if lo <= median_mm <= hi:
            return method
    if median_mm < 0.2:
        return "Extra Fine (Turkish)"
    return "Extra Coarse (Cold Brew+)"


def make_histogram(result: AnalysisResult, unit_label: str) -> go.Figure:
    """Build a Plotly histogram of particle diameters, weighted by result.weighting."""
    d, w = result.diameters_mm, result.weights
    if len(d) <= 1:
        fig = go.Figure()
        fig.add_annotation(text="No particles detected", showarrow=False, font_size=18)
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Particle Size Distribution", "Cumulative Distribution"),
    )

    # Histogram on a log size axis (bins of equal width in log size, so the fine
    # end isn't squashed), as % of total weight. Error bars are counting
    # (Poisson) noise: a bin built from n grains is uncertain by 1/√n of its height.
    # On a log axis, plotly takes shapes (lines, bands) in data units but
    # annotations in log10 units.
    lg = np.log10
    nbins = min(60, max(20, int(np.sqrt(len(d)) * 1.5)))
    edges = np.logspace(lg(d.min()), lg(d.max() * 1.0001), nbins + 1)
    counts, _ = np.histogram(d, edges)
    pct = np.histogram(d, edges, weights=w)[0] / np.sum(w) * 100
    err = np.where(counts > 0, pct / np.sqrt(np.maximum(counts, 1)), 0)
    fig.add_trace(
        go.Bar(
            x=edges[:-1], y=pct, width=np.diff(edges), offset=0,
            error_y=dict(type="data", array=err, color="#8b6914", thickness=1, width=0),
            marker_color="#c9a96e", marker_line_color="#8b6914",
            marker_line_width=0.5, opacity=0.85,
            customdata=np.column_stack([np.sqrt(edges[:-1] * edges[1:]), counts]),
            hovertemplate="%{customdata[0]:.2f}: %{y:.1f}% (%{customdata[1]} grains)<extra></extra>",
        ),
        row=1, col=1,
    )

    # Grains under ~7 px across are too small to detect reliably
    limit = UNDER_DETECTED_PX * (result.pixel_scale if result.pixel_scale > 0 else 1.0)
    if limit > edges[0]:
        fig.add_vrect(x0=edges[0], x1=limit, fillcolor="#888", opacity=0.12, line_width=0, row=1, col=1)
        fig.add_annotation(x=lg(edges[0]), y=0.98, yref="paper", xanchor="left", yanchor="top",
                           text=f"under-detected<br>(< {limit:.2f} {unit_label})", showarrow=False,
                           font=dict(color="#999", size=10))

    # Vertical lines for D10 / D50 / D90
    for val, label, color in [
        (result.d10_mm, "D10", "#4ecdc4"),
        (result.d50_mm, "D50", "#ff6b6b"),
        (result.d90_mm, "D90", "#ffe66d"),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5, row=1, col=1)
        fig.add_annotation(
            x=lg(val), y=1, yref="paper", text=f"{label}: {val:.2f}",
            showarrow=False, font=dict(color=color, size=11), yshift=10,
        )

    # Brew-method shading
    for method, (lo, hi) in BREW_GUIDES.items():
        fig.add_vrect(
            x0=max(lo, edges[0]), x1=hi, fillcolor="#c9a96e", opacity=0.04,
            line_width=0, row=1, col=1,
        )

    # Cumulative
    order = np.argsort(d)
    sorted_d = d[order]
    cumulative = np.cumsum(w[order]) / np.sum(w) * 100
    fig.add_trace(
        go.Scatter(
            x=sorted_d, y=cumulative,
            mode="lines", line=dict(color="#c9a96e", width=2),
            name="Cumulative %",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,107,107,0.33)", row=2, col=1)

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        showlegend=False,
        margin=dict(t=40, b=40),
    )
    ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5) if edges[0] <= t <= edges[-1]] if unit_label == "mm" else None
    fig.update_xaxes(type="log", tickvals=ticks, row=1, col=1)
    fig.update_xaxes(type="log", tickvals=ticks, title_text=f"Particle Diameter ({unit_label}, log scale)", row=2, col=1)
    fig.update_yaxes(title_text=f"% of {result.weighting}", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=2, col=1)

    return fig


def preview(image: np.ndarray, max_width: int = 1600) -> np.ndarray:
    """Downscale an image for display (analysis runs at full resolution)."""
    w = image.shape[1]
    if w <= max_width:
        return image
    return cv2.resize(image, None, fx=max_width / w, fy=max_width / w, interpolation=cv2.INTER_AREA)


@st.cache_data(max_entries=4, show_spinner="Detecting particles…")
def segment_upload(file_bytes: bytes, use_calibration: bool):
    """Decode an upload, find the coin and segment particles.

    Cached on the file bytes, so reruns (any click, changing the reference coin
    or weighting) skip segmentation. Returns None if the image can't be decoded.
    """
    image_bgr = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    image_bgr = prepare_image(image_bgr)

    coin_diameter_px, coin_mask = find_reference_circle(image_bgr) if use_calibration else (0.0, None)
    particle_masks = segment_particles(image_bgr, coin_mask)
    return image_bgr, coin_diameter_px, particle_masks, lighting_evenness(image_bgr, coin_mask)


# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("☕ Coffee Grind Size Analyzer")
st.caption("Drop a photo of your grounds on a white background. Get your particle size distribution in seconds.")

# Sidebar — calibration
with st.sidebar:
    st.header("⚙️ Calibration")
    st.markdown("Place a coin next to your grounds for real-world measurements.")

    ref_choice = st.selectbox("Reference object", list(REFERENCE_OBJECTS.keys()), index=0)
    if ref_choice == "Custom…":
        custom_dia = st.number_input("Diameter (mm)", value=25.0, min_value=1.0, step=0.1)
        ref_diameter_mm = custom_dia
    else:
        ref_diameter_mm = REFERENCE_OBJECTS[ref_choice]

    use_calibration = st.checkbox("Enable calibration", value=True,
                                  help="Uncheck if there's no reference object in the image")

    st.divider()
    st.header("⚖️ Size Weighting")
    weighting = st.selectbox(
        "Weight particles by", list(WEIGHTINGS), index=0,
        help="\n\n".join(f"**{k}**: {v}" for k, v in WEIGHTINGS.items()),
    )

    st.divider()
    st.header("🎯 Brew Guide")
    for method, (lo, hi) in BREW_GUIDES.items():
        st.markdown(f"**{method}**: {lo}–{hi} mm")

# File uploader
uploaded = st.file_uploader(
    "Upload a photo of your coffee grounds",
    type=["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
    help="Best results: white paper background, even lighting, no shadows",
)

if uploaded is not None:
    segmented = segment_upload(uploaded.getvalue(), use_calibration)
    if segmented is None:
        st.error("Could not read image. Please try a different file.")
        st.stop()
    image_bgr, coin_diameter_px, particle_masks, evenness = segmented

    # Calibrate
    pixel_scale = ref_diameter_mm / coin_diameter_px if coin_diameter_px > 0 else 0.0
    if use_calibration:
        if pixel_scale > 0:
            st.success(f"Calibrated: **{pixel_scale:.4f} mm/px** using {ref_choice}")
        else:
            st.warning("Could not detect a circular reference object. Showing results in pixels. "
                       "Keep the coin fully in frame with a finger-width of clear paper around it, and avoid glare.")
    for warning in capture_warnings(uploaded.getvalue()):
        st.warning(warning, icon="📷")
    for warning in lighting_warnings(evenness):
        st.warning(warning, icon="💡")

    unit = "mm" if pixel_scale > 0 else "px"

    result = measure_particles(image_bgr, particle_masks, pixel_scale, weighting)
    for warning in spread_warnings(result.clumped_pct):
        st.warning(warning, icon="🫘")
    if result.weighting != weighting:
        st.info(f"“{weighting}” needs calibration, so sizes are weighted by {result.weighting}.")

    # ── Results ──────────────────────────────────────────────────────────
    st.markdown("---")

    # Metrics row (D10/D50/D90, mean and fines/boulders are weighted by `weighting`)
    cols = st.columns([1, 1, 1.8, 1, 1, 1, 1])  # wider D50 column for its ± value
    cols[0].metric("Particles", f"{result.num_particles:,}")
    cols[1].metric(f"D10 ({unit})", f"{result.d10_mm:.2f}")
    cols[2].metric(f"D50 ({unit})", f"{result.d50_mm:.2f} ± {result.d50_se_mm:.2f}",
                   help="± is counting noise: how much D50 moves when this photo's grains are resampled. "
                        "Photo-to-photo differences (spread, lighting) come on top.")
    cols[3].metric(f"D90 ({unit})", f"{result.d90_mm:.2f}")
    cols[4].metric(f"Mean ({unit})", f"{result.mean_diameter_mm:.2f}")
    cols[5].metric("Fines", f"{result.fines_pct:.0f}%",
                   help=f"Share of {result.weighting} in grains smaller than {FINES_RATIO:g}× D50")
    cols[6].metric("Boulders", f"{result.boulders_pct:.0f}%",
                   help=f"Share of {result.weighting} in grains larger than {BOULDERS_RATIO:g}× D50")
    if result.clumped_pct >= 1 and result.num_particles:
        st.caption(f"{result.clumped_pct:.0f}% of the coffee is in clumps of touching grains (red outlines), "
                   f"left out of the sizes above. Adjusted to no clumping, D50 ≈ "
                   f"{clump_adjusted(result.d50_mm, result.clumped_pct):.2f} {unit} "
                   "(provisional adjustment, measured on one bean).")

    # Brew recommendation
    if result.num_particles == 0:
        st.warning("No particles detected. Spread the grounds in a thin layer on white paper "
                   "with even lighting, and try again.")
    elif pixel_scale > 0:
        brew = classify_grind(result.median_diameter_mm)
        uniformity = result.std_diameter_mm / result.mean_diameter_mm if result.mean_diameter_mm > 0 else 0
        uni_label = "Excellent" if uniformity < 0.3 else "Good" if uniformity < 0.5 else "Fair" if uniformity < 0.7 else "Poor"

        st.markdown(f"""
        <div class="brew-guide">
            <b>Best match:</b> {brew} &nbsp;|&nbsp;
            <b>Uniformity:</b> {uni_label} (CV = {uniformity:.2f}) &nbsp;|&nbsp;
            <b>Spread:</b> D90/D10 = {result.d90_mm / result.d10_mm:.1f}x
        </div>
        """, unsafe_allow_html=True)

    # Tabs: histogram / images
    tab_hist, tab_img, tab_thresh = st.tabs(["📊 Distribution", "🔍 Detected Particles", "⬛ Threshold"])

    with tab_hist:
        fig = make_histogram(result, unit)
        st.plotly_chart(fig, use_container_width=True)

    with tab_img:
        # Convert BGR → RGB for display
        labeled_rgb = cv2.cvtColor(preview(result.labeled_image), cv2.COLOR_BGR2RGB)
        st.image(labeled_rgb, caption=f"{result.num_particles} particles detected", use_container_width=True)

    with tab_thresh:
        st.image(preview(result.threshold_image), caption="Binary threshold mask (white = coffee)",
                 use_container_width=True)

    # Download CSV
    st.markdown("---")
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([f"diameter_{unit}", "area_px", f"volume_{unit}3", f"weight_{result.weighting.replace(' ', '_')}"])
    for d_val, a_val, v_val, w_val in zip(result.diameters_mm, result.areas_px, result.volumes_mm3, result.weights):
        writer.writerow([f"{d_val:.4f}", f"{a_val:.1f}", f"{v_val:.6g}", f"{w_val:.6g}"])
    st.download_button(
        "📥 Download particle data (CSV)",
        data=buf.getvalue(),
        file_name="grind_analysis.csv",
        mime="text/csv",
    )

    # Comparisons and the log always use available-mass sizes, so records stay comparable
    am_result = result if result.weighting == "available mass" else \
        measure_particles(image_bgr, particle_masks, pixel_scale, "available mass")

    def share_under(r, mm=0.6):
        d, w = r.diameters_mm, r.weights
        return 100 * w[d < mm].sum() / w.sum() if pixel_scale > 0 and len(d) and w.sum() > 0 else None

    # Compare with a saved measurement, e.g. a calibration grind
    references = [r for r in read_records() if r.get("shape")]
    with st.expander("🔍 Compare with a saved measurement"):
        if not references:
            st.caption("Save a measurement first (below); it can then be used as a reference, "
                       "e.g. the grind that tastes right for a coffee.")
        else:
            ref = st.selectbox("Reference", references[::-1], format_func=describe, key="compare_ref")
            ref_size, ref_unit = ref.get("size", {}), ref.get("size", {}).get("unit", "mm")
            d50_ratio = am_result.d50_mm / ref_size["d50"] if unit == ref_unit and ref_size.get("d50") else None
            check = shape_comparison(am_result.shape, ref["shape"], d50_ratio)
            here_under, ref_under = share_under(am_result), ref_size.get("share_under_0_6mm_pct")
            rows = [("D50 (available mass)", f"{am_result.d50_mm:.2f} {unit}", f"{ref_size.get('d50', float('nan')):.2f} {ref_unit}"),
                    ("Coffee in grains < 0.6 mm", f"{here_under:.1f}%" if here_under is not None else "–",
                     f"{ref_under:.1f}%" if ref_under is not None else "–"),
                    ("Clumped", f"{am_result.clumped_pct:.0f}%", f"{ref.get('capture', {}).get('clumped_pct', float('nan')):.0f}%")]
            st.table({"": [r[0] for r in rows], "This photo": [r[1] for r in rows], "Reference": [r[2] for r in rows]})
            if d50_ratio:
                st.markdown(f"**D50 is {100 * (d50_ratio - 1):+.1f}%** compared with the reference.")
            if check:
                dist, level, msg = check
                {"same": st.success, "somewhat different": st.info, "clearly different": st.warning}[level](
                    f"Shape distance {dist:.3f}. {msg}")
            here_cam, ref_cam = camera_info(uploaded.getvalue()).get("model"), ref.get("camera", {}).get("model")
            if here_cam and ref_cam and here_cam != ref_cam:
                st.caption(f"Different phones ({here_cam} vs {ref_cam}): sizes may be offset between them.")

    # Save to the local measurement log (see measurement_log.py)
    with st.expander("📝 Save this measurement"):
        st.caption(f"Adds one line to `{log_path()}` on this computer: the sizes (available-mass weighting), "
                   "the phone model and camera mode from the photo, and what you enter here. "
                   "The photo itself and its location are not saved.")
        with st.form("save_measurement"):
            c1, c2 = st.columns(2)
            bean = c1.text_input("Bean / roast", key="log_bean", placeholder="e.g. Ethiopia Guji, roasted 2026-10-20")
            grinder = c2.text_input("Grinder", key="log_grinder", placeholder="e.g. Baratza Encore")
            setting = c1.text_input("Grinder setting", key="log_setting")
            method = c2.selectbox("Brew method", BREW_METHODS, key="log_method")
            dose = c1.number_input("Dose (g)", min_value=0.0, step=0.5, key="log_dose")
            water = c2.number_input("Water (g)", min_value=0.0, step=10.0, key="log_water")
            taste = c1.selectbox("How did it taste?", TASTES, key="log_taste")
            tds = c2.number_input("TDS % (0 = not measured)", min_value=0.0, max_value=30.0, step=0.01,
                                  format="%.2f", key="log_tds")
            notes = st.text_input("Notes", key="log_notes")
            if st.form_submit_button("Save"):
                logged = am_result
                context = {
                    "bean": bean or None, "grinder": grinder or None, "setting": setting or None,
                    "brew_method": method, "dose_g": dose or None, "water_g": water or None,
                    "taste": taste, "tds_pct": tds or None, "notes": notes or None,
                    "reference_object": ref_choice if pixel_scale > 0 else None,
                }
                warnings_all = (capture_warnings(uploaded.getvalue()) + lighting_warnings(evenness)
                                + spread_warnings(logged.clumped_pct))
                record = build_record(uploaded.getvalue(), logged, coin_diameter_px=coin_diameter_px,
                                      evenness=evenness, warnings=warnings_all, context=context)
                st.success(f"Saved. `{log_path()}` now holds {append_record(record)} measurement(s).")

else:
    # Landing page
    st.markdown("""
    ### How to use
    1. **Spread** your coffee grounds on a white sheet of paper
    2. **Place** a coin (e.g. US Quarter) next to the grounds for scale
    3. **Photograph** from above with even lighting — avoid shadows
    4. **Upload** the image above

    The analyzer will automatically:
    - Detect and separate individual particles (even touching ones)
    - Measure each particle's diameter
    - Show you the full size distribution
    - Suggest the best brew method for your grind
    """)

    # Show example placeholder
    st.info("👆 Upload an image to get started!")
