# Spatial Filter-Based Noise Reduction and Its Impact on Edge Detection

## 1. What this project does
Denoising before edge detection is standard advice. We test it quantitatively:
clean image -> add known noise -> denoise with several filters -> run edge detectors -> compare the edges with the clean image's edges.

**Hypotheses**
- **H1:** edge maps from a *denoised* image are closer to the clean image's edges than those from the *noisy* image.
- **H2:** denoising only pays off above some noise level (valuable for space/low-light images, unnecessary for clean photos).

## 2. Repository structure
- `notebooks/denoise_impact_on_edges.ipynb` — executed notebook with outputs
- `data/real_images/` — real noisy photos
- `outputs/` — all figures (.png) and result tables (.csv) produced by the run
- `docs/` — final report and slides

## 3. Dataset
| Data | Source | Use |
|---|---|---|
| 4 reference images (Camera Man, Astronaut, Coins, Chelsea Cat) | bundled with scikit-image (`skimage.data`), converted to 512x512 grayscale | ground truth; noise is added synthetically |
| Real noisy photos | downloaded from the internet (camera model / ISO are in the file names) | reference-free analysis (no clean version) |

## 4. How to run
### Option A — Kaggle (recommended)
1. Create a new Kaggle notebook and import `notebooks/denoise_impact_on_edges.ipynb`.
2. Add your real photos as a Kaggle dataset and set `REAL_IMAGE_DIR = '/kaggle/input/<dataset-name>'` in the real-photo cell (Section 12). If the folder does not exist, that section is skipped and everything else still runs.
3. Click **Save Version -> Save & Run All**. Results appear in `/kaggle/working/outputs`.

### Option B — Local machine (Python 3.11)
```bash
git clone https://github.com/waquar003/Section_D_Team_2.git
cd Section_D_Team_2
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# edit REAL_IMAGE_DIR in src/denoise_impact_on_edges.py -> 'data/real_images'
python src/denoise_impact_on_edges.py   # run from the repo root
```
Everything is saved in `./outputs/`. All random generators are seeded (`SEED = 42`), so results are reproducible.

## 5. Implementation details
**Noise models:** salt-and-pepper (5%), Gaussian (sigma = 25) and mixed (sigma = 15 + 2% impulses) in the main experiment; photon-limited noise (Poisson shot noise + read noise + hot pixels, for space/low-light) in the dedicated sweep.

**Denoisers (6):** mean 5x5, median 5x5, Gaussian 5x5, bilateral, non-local means, and a median -> bilateral cascade. Parameters are fixed (not tuned per noise type) for a fair comparison; a sensitivity study checks how much this matters.

**Edge detectors (3):** Sobel, Canny (after explicit Gaussian smoothing) and Laplacian of Gaussian (LoG). Thresholds are calibrated **once on the clean image and then frozen**, so a noisy image cannot be rescued by retuning.

**Metrics:** 
- PSNR and SSIM (image quality)
- Edge precision, recall and F1 with 1-pixel tolerance against the clean image's edges (edge quality)
- Real photos have no clean reference, so they use reference-free measures: `Immerkaer's noise-sigma estimate`, `edge density relative to the typical clean-image density`, and `isolated-edge "speck" counts`.

**Sections of the script:**
| Section | What it does |
|---|---|
| 1 | Load and prepare the 4 reference images |
| 2 | Noise models (seeded, reproducible) |
| 3 | Metrics: PSNR, SSIM, edge precision / recall / F1 |
| 4 | The six denoising filters |
| 5 | Edge detectors and calibration on the clean image |
| 6 | Main experiment: 4 images x 3 noises x 6 filters x 3 detectors |
| 7 | Visual results: denoising grids and edge-map grids |
| 8 | Quantitative summary: bar charts and edge-F1 heatmaps |
| 9 | Hypothesis test: best-PSNR filter vs noisy, per detector; best filter by PSNR / SSIM / edge F1 |
| 10 | Noise-level sweep (S&P density, Gaussian sigma) |
| 11 | Parameter sensitivity of each filter |
| 12 | Real noisy photographs (reference-free analysis) |
| 13 | Crossover analysis: the noise level above which denoising beats doing nothing |
| 14 | Photon-limited (space / low-light) noise sweep |
| 15 | Robustness check with per-version adaptive thresholds |


## 6. Key findings
1. **H1 holds, but depends on the detector.** Using the best-PSNR filter for each (image, noise), edge F1 improved in 27 of 36 cases (75%, mean gain +0.088): Sobel 12/12 (+0.164), Canny 8/12 (+0.065), LoG 7/12 (+0.033). Sobel has no smoothing of its own, so it gains most.
2. **The best denoiser depends on the noise.** Salt-and-pepper: median is best on PSNR, SSIM and edge F1 (30.0 dB, Canny F1 0.80). Gaussian: bilateral has the best PSNR (28.3 dB) but non-local means has the best edge F1 (Canny 0.81). Mixed: median has the best PSNR (28.8 dB) but the Gaussian filter has the best Canny and LoG F1. **The best filter by PSNR is not always the best for edges.**
3. **Strong edge-preserving filters can hurt on impulses.** Bilateral treats impulse pixels as edges and keeps them (22.9 dB, LoG F1 0.33 vs 0.56 for the noisy image).
4. **H2: a crossover exists.** With Gaussian 5x5, denoising improves PSNR from sigma about 8 and SSIM from about 4, but improves edges only from sigma about 15-20 (Sobel 16.4, LoG 15.0, Canny 20.2). At sigma = 2 it *lowers* Canny F1 from 0.96 to 0.80 (a "denoising tax" caused by blur). For impulses, median pays off from 0.3% (PSNR) to 2.8% (LoG) corrupted pixels.
5. **Space / low-light noise.** For photon-limited noise, denoising helps edges from about 84 photons (Canny) to 163 photons (Sobel), equivalent to an estimated sigma of about 14-19.
6. **Robust to the threshold choice.** With per-image adaptive thresholds the crossovers are similar (Sobel 15.4, Canny 20.0, LoG 8.8), and denoising still costs 0.06-0.11 F1 at sigma = 2, so the finding is not an artefact of frozen thresholds.
7. **Practical rule:** 
   - Estimate sigma first. 
   - For clean photos (sigma about 1-2) do not denoise before edge detection.
   - Use a median filter for impulse noise.
   - Use spatial smoothing or non-local means (with h scaled to sigma) for strong or space/low-light noise.
   - Choose the denoiser with an edge metric, not PSNR.

## 7. Limitations
- 4 test images
- Synthetic noise with one seeded realisation per setting
- The clean image's edges stand in for human-annotated edges
- Real photos have no ground truth, so edge density is only a proxy (a density below 1 can also mean lost edges). Non-local means uses a fixed h = 15, which is too weak at high noise.

## 8. Report and presentation
- Report: `docs\Noise_Reduction_Edge_Detection_Report.pdf`  
- Slides: `docs\Noise_Reduction_Edge_Detection_Presentation.pptx` 
