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
    analyze_image,
    detect_reference_circle,
    REFERENCE_OBJECTS,
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
    """Build a Plotly histogram of particle diameters."""
    d = result.diameters_mm
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

    # Histogram
    nbins = min(80, max(20, int(np.sqrt(len(d)) * 2)))
    fig.add_trace(
        go.Histogram(
            x=d, nbinsx=nbins,
            marker_color="#c9a96e", marker_line_color="#8b6914",
            marker_line_width=0.5, opacity=0.85,
            name="Count",
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
    sorted_d = np.sort(d)
    cumulative = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
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
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=2, col=1)

    return fig


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
    # Load image
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read image. Please try a different file.")
        st.stop()

    # Downscale very large images for speed
    h, w = image_bgr.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Calibrate
    pixel_scale = 0.0
    coin_mask = None
    if use_calibration and ref_diameter_mm:
        with st.spinner("Detecting reference object…"):
            pixel_scale, coin_mask = detect_reference_circle(image_bgr, ref_diameter_mm)
        if pixel_scale > 0:
            st.success(f"Calibrated: **{pixel_scale:.4f} mm/px** using {ref_choice}")
        else:
            st.warning("Could not detect a circular reference object. Showing results in pixels.")

    unit = "mm" if pixel_scale > 0 else "px"

    # Analyze
    with st.spinner("Running MobileSAM segmentation (first run loads model)…"):
        result = analyze_image(image_bgr, pixel_scale=pixel_scale, coin_mask=coin_mask)

    # ── Results ──────────────────────────────────────────────────────────
    st.markdown("---")

    # Metrics row
    cols = st.columns(6)
    cols[0].metric("Particles", f"{result.num_particles:,}")
    cols[1].metric(f"Median ({unit})", f"{result.median_diameter_mm:.2f}")
    cols[2].metric(f"Mean ({unit})", f"{result.mean_diameter_mm:.2f}")
    cols[3].metric("D10", f"{result.d10_mm:.2f}")
    cols[4].metric("D50", f"{result.d50_mm:.2f}")
    cols[5].metric("D90", f"{result.d90_mm:.2f}")

    # Brew recommendation
    if pixel_scale > 0:
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
        labeled_rgb = cv2.cvtColor(result.labeled_image, cv2.COLOR_BGR2RGB)
        st.image(labeled_rgb, caption=f"{result.num_particles} particles detected", use_container_width=True)

    with tab_thresh:
        st.image(result.threshold_image, caption="Binary threshold mask (white = coffee)", use_container_width=True)

    # Download CSV
    st.markdown("---")
    import io, csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([f"diameter_{unit}", f"area_px"])
    for d_val, a_val in zip(result.diameters_mm, result.areas_px):
        writer.writerow([f"{d_val:.4f}", f"{a_val:.1f}"])
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
