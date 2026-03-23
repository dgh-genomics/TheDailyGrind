# TheDailyGrind

A modern, easy-to-use tool for analyzing coffee grind particle size distributions from photographs. Upload a photo of your grounds on white paper, get a full size distribution and brew recommendation in seconds.

Uses a hybrid **MobileSAM + adaptive threshold** pipeline for robust particle detection across a wide range of grind sizes.

## Features

- **Drag-and-drop** web interface (Streamlit)
- **Automatic coin calibration** — place a US Quarter, Dime, Euro, etc. for real-world measurements
- **Hybrid segmentation** — MobileSAM for precise particle boundaries, adaptive threshold for fine particles
- **Full statistics** — median, mean, D10/D50/D90, uniformity, fines/boulders percentage
- **Brew recommendation** — suggests the best brew method based on your grind size
- **CSV export** — download particle-level data for further analysis
- **No parameter tuning needed** — just upload and go

## Quick Start

```bash
git clone https://github.com/dgh-genomics/TheDailyGrind.git
cd TheDailyGrind
pip install -r requirements.txt
streamlit run app.py
```

The MobileSAM model weights (~39MB) are downloaded automatically on first run.

## How to Take a Good Photo

1. **Spread** grounds in a thin, single layer on white paper
2. **Place** a reference coin (e.g., US Quarter) in the frame
3. **Photograph** straight down with even, diffuse lighting
4. **Avoid** shadows, piled-up grounds, and glossy surfaces

## How It Works

1. **CLAHE + adaptive threshold** finds every dark candidate region (two block sizes for large particles and fines)
2. **MobileSAM predictor** refines each candidate with point + bounding box prompts for precise masks
3. **Threshold fallback** handles tiny particles that don't need SAM refinement
4. **Darkness filter** rejects non-particle regions (shadows, background artifacts)
5. **Hough circle detection** finds the reference coin for pixel-to-mm calibration

## Requirements

- Python 3.9+
- PyTorch (CPU is fine — no GPU needed)
- See `requirements.txt` for full list

## Inspired By

- [coffeegrindsize](https://github.com/jgagneastro/coffeegrindsize) by Jonathan Gagne — the original coffee grind analyzer that pioneered image-based particle size analysis for coffee
- [coffeegrindsize v1.0.0](https://github.com/csatt/coffeegrindsize/releases/tag/v1.0.0) by csatt — a refined fork with improved usability

## License

MIT
