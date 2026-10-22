# TheDailyGrind

A modern, easy-to-use tool for analyzing coffee grind particle size distributions from photographs. Upload a photo of your grounds on white paper, get a full size distribution and brew recommendation in seconds.

Every grain is measured the same way — a paper-normalized **blue-channel threshold** with **watershed splitting** of touching grains — so results don't depend on how densely you spread the grounds. MobileSAM refinement is available as an option.

## Features

- **Drag-and-drop** web interface (Streamlit)
- **Automatic coin calibration** — place a US Quarter, Dime, Euro, etc. for real-world measurements
- **Size weighting** — by available mass (coffeegrindsize's default), mass, surface area or count
- **Error bars** — counting noise on every histogram bar, and D50 ± its standard error
- **Full statistics** — D10/D50/D90, mean, uniformity, fines/boulders share
- **Brew recommendation** — suggests the best brew method based on your grind size
- **Grinder-setting comparison** — batch script that tests whether photos can tell settings apart
- **CSV export** — download particle-level data for further analysis

## Quick Start

```bash
git clone https://github.com/dgh-genomics/TheDailyGrind.git
cd TheDailyGrind
pip install -r requirements.txt
streamlit run app.py
```

Optional MobileSAM refinement (adds PyTorch; weights, ~39MB, download on first use):

```bash
pip install -r requirements-sam.txt
```

## How to Take a Good Photo

1. **Spread** grounds in a thin, single layer on white paper
2. **Place** a reference coin (e.g., US Quarter) in the frame
3. **Photograph** straight down with even, diffuse lighting
4. **Avoid** shadows, piled-up grounds, and glossy surfaces

Resolution limits accuracy most: fill the frame with the paper, and use your camera's full resolution (photos are analyzed at up to 4000 px). HEIC photos must be converted to JPEG for the comparison script.

## Comparing Grinder Settings

`compare_settings.py` answers "can photos tell neighbouring grinder settings apart?". Put photos in one folder per setting, optionally with one subfolder per separate grind:

```
photos/
  12/  IMG_0001.jpg  IMG_0002.jpg  ...
  13/  grind_a/IMG_0101.jpg  grind_b/IMG_0107.jpg  ...
  14/  ...
```

```bash
python compare_settings.py photos/ --coin "US Quarter" --out results/
```

For each weighting it reports each setting's mean D50 and photo-to-photo SD, the **separation** between neighbouring settings (ΔD50 ÷ photo-to-photo SD; aim for ≥ 3, about 7% single-photo mix-ups), how many photos per measurement would reach that, and leave-one-out accuracy. It writes `photos.csv`, per-grain CSVs and `report.html`.

A suggested first experiment: one grinder, five adjacent settings plus two far-apart settings as a positive control; per setting, 3 separate grinds × 3 spreads × 2 photos.

## How It Works

1. **Paper normalization** — the blue channel (coffee's strongest contrast against white paper) is divided by a smooth estimate of the local paper brightness, so uneven lighting cancels out
2. **Threshold** at 58.8% of the paper brightness — about half-way between coffee and paper, so blurred edges are neither grown nor shrunk
3. **Watershed** on the distance transform splits touching grains
4. **Edge rejection** drops grains cut off by the image edge or the coin
5. **Measurement** — equivalent-circle diameter; volume as an ellipsoid whose hidden axis equals its short axis; available mass assumes water reaches 0.1 mm into each grain
6. **Hough circle detection** finds the reference coin for pixel-to-mm calibration

## Requirements

- Python 3.9+
- See `requirements.txt` (and `requirements-sam.txt` for the MobileSAM option)

## Inspired By

- [coffeegrindsize](https://github.com/jgagneastro/coffeegrindsize) by Jonathan Gagne — the original coffee grind analyzer that pioneered image-based particle size analysis for coffee; the blue-channel threshold, edge rejection, ellipsoid volumes and available-mass weighting come from it
- [coffeegrindsize v1.0.0](https://github.com/csatt/coffeegrindsize/releases/tag/v1.0.0) by csatt — a refined fork with improved usability

## License

MIT
