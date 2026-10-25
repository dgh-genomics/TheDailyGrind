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
    REFERENCE_OBJECTS,
    FINES_RATIO,
    BOULDERS_RATIO,
    WEIGHTINGS,
    AnalysisResult,
)

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

    # Histogram, as % of total weight. Error bars are counting (Poisson) noise:
    # a bin built from n grains is uncertain by 1/√n of its height.
    nbins = min(80, max(20, int(np.sqrt(len(d)) * 2)))
    edges = np.histogram_bin_edges(d, bins=nbins)
    counts, _ = np.histogram(d, edges)
    pct = np.histogram(d, edges, weights=w)[0] / np.sum(w) * 100
    err = np.where(counts > 0, pct / np.sqrt(np.maximum(counts, 1)), 0)
    fig.add_trace(
        go.Bar(
            x=(edges[:-1] + edges[1:]) / 2, y=pct, width=np.diff(edges),
            error_y=dict(type="data", array=err, color="#8b6914", thickness=1, width=0),
            marker_color="#c9a96e", marker_line_color="#8b6914",
            marker_line_width=0.5, opacity=0.85,
            customdata=counts, hovertemplate="%{x:.2f}: %{y:.1f}% (%{customdata} grains)<extra></extra>",
        ),
        row=1, col=1,
    )

    # Vertical lines for D10 / D50 / D90
    for val, label, color in [
        (result.d10_mm, "D10", "#4ecdc4"),
        (result.d50_mm, "D50", "#ff6b6b"),
        (result.d90_mm, "D90", "#ffe66d"),
    ]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, line_width=1.5, row=1, col=1)
        fig.add_annotation(
            x=val, y=1, yref="paper", text=f"{label}: {val:.2f}",
            showarrow=False, font=dict(color=color, size=11), yshift=10,
        )

    # Brew-method shading
    for method, (lo, hi) in BREW_GUIDES.items():
        fig.add_vrect(
            x0=lo, x1=hi, fillcolor="#c9a96e", opacity=0.04,
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
    fig.update_xaxes(title_text=f"Particle Diameter ({unit_label})", row=2, col=1)
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
    return image_bgr, coin_diameter_px, particle_masks


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
    image_bgr, coin_diameter_px, particle_masks = segmented

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
