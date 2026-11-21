# Galaxy Morphology Evolution Lab

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights%20%26%20Data-FFD21E)](https://huggingface.co/Krishna2805/galaxy-morphology-weights)
![PyTorch](https://img.shields.io/badge/PyTorch-tested%202.15-EE4C2C?logo=pytorch)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python)

An end-to-end deep learning and observational astrophysics pipeline designed to investigate **how galaxy morphology evolves across cosmic lookback time**. By combining optical multi-spectral image cutouts from the Sloan Digital Sky Survey (SDSS DR7 / Galaxy Zoo 2) with 6 spectroscopic metadata features, the system classifies galaxy morphology and measures how each class's centroid in the learned 256-d embedding space shifts across redshift bins (`low_z → mid_z → high_z`).

---

## 🔬 Scientific Purpose & Core Findings

As optical surveys capture galaxies at higher cosmological redshift (z), two observational biases emerge:
1. **Cosmological Surface Brightness Dimming** (scaling with `(1 + z)^4`).
2. **Spatial Resolution Degradation**: Faint disk structures and subtle spiral arms blur together, which can lead image-only networks to misclassify distant disk and spiral galaxies as featureless smooth/elliptical systems.

### Our Approach:
1. **Multimodal Late Fusion (`MultimodalFusionNet`)**: Combines optical images (via pretrained ResNet-18 or EfficientNet-B0) with SDSS spectroscopic metadata (redshift z, color indices `u-r` and `g-i`, De Vaucouleurs profile fraction `fracdev_r`, half-light radius `petror50_r_kpc`, and surface brightness `mu50_r`).
2. **256-Dimensional Latent Bottleneck**: Extracts dense morphological representations prior to the classification head.
3. **Centroid Drift Analysis**: Tracks morphological feature drift across science-driven redshift bins (`low_z: 0.01-0.06`, `mid_z: 0.06-0.10`, `high_z: 0.10-0.14`, `ext_z: 0.14-0.25`) using Euclidean distance and Cosine angular distance.
4. **Manifold Projections**: Reduces embeddings to 2D manifolds via **UMAP** and **t-SNE** with interactive Plotly visual maps.

### Results (seed-42 stratified test split, n = 7,064; source: `outputs/test_metrics.json`, written by `evaluate_models.py`)

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| ResNet-18 (image only) | 76.7% | 0.688 |
| EfficientNet-B0 (image only) | 78.8% | 0.695 |
| Multimodal ResNet-18 | 78.5% | **0.706** |
| Multimodal EfficientNet-B0 | 75.3% | 0.674 |

### Key Findings:
- **Metadata fusion helps ResNet-18, not EfficientNet-B0**: Multimodal ResNet-18 has the best macro-F1 (0.706), +1.8 pts accuracy and +1.8 pts macro-F1 over image-only ResNet-18. Multimodal EfficientNet-B0 is 3.5 pts accuracy and 2.2 pts macro-F1 *below* image-only EfficientNet-B0, which has the highest accuracy overall (78.8%).
- **Disk galaxies show the largest centroid shift**: 4.07 (Euclidean) from low_z → mid_z with the ResNet-18 model; spiral mid_z → high_z shows the smallest, 0.99 (`outputs/drift_resnet18/centroid_drift.csv`).
- **Redshift-input control**: redshift is one of the model's inputs, so it moves the embeddings by itself. Changing *only* the redshift input from z = 0.035 to 0.08 on the same 256 galaxies moves the mean embedding by 1.56 (ResNet-18) / 1.21 (EfficientNet-B0) (`outputs/drift_redshift_control.json`), against low_z → mid_z class drifts of 2.68–4.07 (ResNet-18). The drift numbers therefore combine image-driven change with the effect of the redshift input; separating the two (e.g. drift on image-branch-only embeddings) is future work.

---

## 📂 Project Structure

```
Galaxy-Morphology-Evolution/
├── README.md                           # Project documentation & quickstart guide
├── pyproject.toml / requirements.txt   # Dependencies & environment configuration
├── .gitignore                          # Git exclusions (datasets, binaries, weights)
├── dataset.py                          # PyTorch Image Dataset loader & stratified splits
├── dataset_multimodal.py               # Multimodal (Image + Metadata) Dataset loader
├── baseline_model.py                   # ResNet-18 vision baseline architecture
├── efficientnet_model.py               # EfficientNet-B0 vision baseline architecture
├── multimodal_model.py                 # Custom MultimodalFusionNet late-fusion architecture
├── train_baseline.py                   # ResNet-18 baseline training script
├── train_efficientnet.py               # EfficientNet-B0 baseline training script
├── train_multimodal.py                 # Multimodal fusion model training script
├── drift_analysis.py                   # Centroid drift analysis across redshift bins
├── phase4_visualize.py                 # UMAP & t-SNE 2D manifold generation script
├── evaluate_models.py                  # Test-split metrics + redshift-input control → outputs/*.json
├── galaxy_app.py                       # Streamlit interactive dashboard & live classifier
├── outputs/                            # Figures, drift tables, 2D maps the dashboard reads (~9.6 MB, committed)
└── sample_images/                      # 32 test-split cutouts so the Live Classifier works without images_gz2/
```

---

## ☁️ Hugging Face Model Weights & Dataset Auto-Downloader

Trained model checkpoints and dataset tables are hosted on the Hugging Face model repository:  
👉 [**`Krishna2805/galaxy-morphology-weights`**](https://huggingface.co/Krishna2805/galaxy-morphology-weights)

On first run the app downloads these three files (the only ones hosted there) to `checkpoints/` and the project root via `huggingface_hub`:
- **`multimodal_resnet18.pth`** (~45.5 MB): Pretrained PyTorch weights for `MultimodalFusionNet`
- **`metadata_stats.pth`** (~1.9 KB): Normalization means and standard deviations for tabular spectroscopic metadata
- **`galaxy_master_dataset.csv`** (~10.2 MB): Full 47,089-galaxy dataset table containing SDSS dr7objid, spectroscopic redshift, morphology classes, and photometric parameters

`multimodal_efficientnet_b0.pth` (~17.8 MB) is also hosted; the app downloads it the first time you pick EfficientNet-B0 in the Live Classifier. Not hosted: the image-only baselines (train them with Step 1 below). Everything else the dashboard shows ships in the repo (`outputs/`, `sample_images/`).

**1-Line Verification Command** (exit code 1 if any file is still missing):
```bash
python -c "import sys, galaxy_app; ok = galaxy_app.ensure_weights_and_data(); print('OK: weights and dataset present' if ok else 'MISSING: see messages above'); sys.exit(0 if ok else 1)"
```

---

## 📥 Dataset Acquisition & Cutouts

- **Master Table**: Automatically downloaded via the Hugging Face auto-downloader as documented above.
- **Raw Optical Cutouts (`images_gz2/`)**: needed only for training or re-running the pipeline. The 424×424 RGB JPEG cutouts (resized to 224×224 for the models) are named by Galaxy Zoo 2 `asset_id`: all 47,089 (`dr7objid`, filename) pairs in `galaxy_master_dataset.csv` match `gz2_filename_mapping.csv` in [Galaxy Zoo 2: Images from Original Sample](https://doi.org/10.5281/zenodo.3565489), whose `images_gz2.zip` (3.35 GB) holds the images. Extract it so the JPEGs sit directly in `images_gz2/` in the repo root (e.g. `images_gz2/110939.jpg`).

### Data credits
Images and morphology labels come from Galaxy Zoo 2 (Willett et al. 2013, MNRAS 435, 2835; Hart et al. 2016, MNRAS 461, 3663), built on Sloan Digital Sky Survey DR7 imaging. The cutouts in `images_gz2/` and the 32 in `sample_images/` are from the Galaxy Zoo team's image release, [doi:10.5281/zenodo.3565489](https://doi.org/10.5281/zenodo.3565489) (CC BY 4.0).

---

## ⚙️ Installation & Setup

Python: the `pip` path works on Python 3.11+ (tested on 3.11.9 with PyTorch 2.15.0, CPU). The `uv` path follows `pyproject.toml` (Python 3.13+, `torch>=2.10.0`); its lockfile installs PyTorch 2.10.0 with CUDA 12.8, a version not re-tested here.

### Option A: Using `uv` (Recommended)

```bash
git clone https://github.com/Krishna2805/Galaxy-Morphology-Evolution.git
cd Galaxy-Morphology-Evolution
uv sync
```

### Option B: Using Standard `pip`

```bash
git clone https://github.com/Krishna2805/Galaxy-Morphology-Evolution.git
cd Galaxy-Morphology-Evolution
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## 🚀 Execution Pipeline

### 1. Launch the Interactive Dashboard
The Streamlit app provides multi-tab exploration of exploratory data analysis (EDA), model benchmarks, redshift centroid drift curves, interactive UMAP/t-SNE maps, and real-time morphological inference:

```bash
# Using uv
uv run streamlit run galaxy_app.py

# Using standard Python
streamlit run galaxy_app.py
```

### 2. Training Models from Scratch

**Step 1: Train Single-Modal Vision Baselines**
```bash
python train_baseline.py --epochs 20 --fine-tune
python train_efficientnet.py --epochs 20 --fine-tune
```

**Step 2: Train Multimodal Late-Fusion Network**
```bash
python train_multimodal.py --backbone resnet18 --epochs 20 --fine-tune
python train_multimodal.py --backbone efficientnet_b0 --epochs 20 --fine-tune
```

**Step 3: Quantify Morphological Drift Across Redshift Bins**
```bash
python drift_analysis.py --backbone resnet18
python drift_analysis.py --backbone efficientnet_b0
```

**Step 4: Generate UMAP & t-SNE Manifold Visualizations**
```bash
python phase4_visualize.py --backbone resnet18 --method both
python phase4_visualize.py --backbone efficientnet_b0 --method both
```

**Step 5: Test-Split Metrics and Redshift-Input Control** (writes `outputs/test_metrics.json`, `outputs/drift_redshift_control.json`)
```bash
python evaluate_models.py                     # ~35 min on CPU for all four models
python evaluate_models.py --redshift-control  # ~1 min
```

---

## 💻 Hardware & System Notes

- **Deterministic Splits**: Dataset partitioning uses stratified 70/15/15 splits with a fixed seed (`random_state=42`) to prevent data leakage across epochs.
- **Windows Acceleration**: DataLoader `--num-workers 0` is set by default to avoid multiprocessing socket deadlocks on Windows systems.
- **GPU Acceleration**: CUDA devices are automatically selected if available; runs seamlessly on CPU for inference.
