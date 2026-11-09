# Galaxy Morphology Evolution Lab

![Streamlit Status](https://img.shields.io/badge/Streamlit-App_Live-FF4B4B?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)
![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python)

An end-to-end deep learning and observational astrophysics pipeline designed to investigate **how galaxy morphology evolves across cosmic lookback time**. By combining optical multi-spectral image cutouts from the Sloan Digital Sky Survey (SDSS DR7 / Galaxy Zoo 2) with 6 spectroscopic metadata features, the system mitigates cosmological resolution degradation bias and quantifies high-dimensional latent space centroid drift across cosmic redshift bins (`low_z → mid_z → high_z`).

---

## 🔬 Scientific Purpose & Core Findings

As optical surveys capture galaxies at higher cosmological redshift (z), two observational biases emerge:
1. **Cosmological Surface Brightness Dimming** (scaling with `(1 + z)^4`).
2. **Spatial Resolution Degradation**: Faint disk structures and subtle spiral arms blur together, causing single-modal convolutional vision networks to systematically misclassify distant disk and spiral galaxies as featureless smooth/elliptical systems.

### Our Approach:
1. **Multimodal Late Fusion (`MultimodalFusionNet`)**: Combines optical images (via pretrained ResNet-18 or EfficientNet-B0) with SDSS spectroscopic metadata (redshift z, color indices `u-r` and `g-i`, De Vaucouleurs profile fraction `fracdev_r`, half-light radius `petror50_r_kpc`, and surface brightness `mu50_r`).
2. **256-Dimensional Latent Bottleneck**: Extracts dense morphological representations prior to the classification head.
3. **Centroid Drift Analysis**: Tracks morphological feature drift across science-driven redshift bins (`low_z: 0.01-0.06`, `mid_z: 0.06-0.10`, `high_z: 0.10-0.14`, `ext_z: 0.14-0.25`) using Euclidean distance and Cosine angular distance.
4. **Manifold Projections**: Reduces embeddings to 2D manifolds via **UMAP** and **t-SNE** with interactive Plotly visual maps.

### Key Findings:
- **Disk Galaxies Drift the Most**: Exhibit the highest Euclidean centroid drift (4.07 from low_z → mid_z in ResNet-18), showing that observational blurring distorts both feature magnitude and representation angle.
- **Spiral Galaxies Remain Stable**: Exhibit minimal latent drift (0.99 from mid_z → high_z), proving that prominent spiral arm features resist resolution degradation across lookback distance.
- **Multimodal Superiority**: Incorporating spectroscopic metadata outperforms image-only backbones by ~2.1% in accuracy and +4.3% in macro F1-score.

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
└── galaxy_app.py                       # Streamlit interactive dashboard & live classifier
```

---

## 📥 Dataset Acquisition & Setup

Due to GitHub repository limits, raw cutout images and full dataset tables are excluded from version control.

### 1. Required Files
To run the full training or extraction pipelines, place the following in the project root:
- **`galaxy_master_dataset.csv`**: Metadata CSV with SDSS galaxy IDs (`dr7objid`), morphology classes (`disk`, `edge_on`, `smooth`, `spiral`), spectroscopic redshift (`redshift`), and photometry.
- **`images_gz2/`**: Subfolder containing 224x224 RGB JPEG galaxy cutouts named by ID (e.g. `110939.jpg`).

### 2. Sources
- **Dataset**: Galaxy Zoo 2 (SDSS DR7)
- **Primary Data Portal**: [Galaxy Zoo 2 Data Tables (Data-GZ2)](https://data.galaxyzoo.org/)
- **Kaggle Mirror**: [Galaxy Zoo Challenge Data](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data)

---

## ⚙️ Installation & Setup

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

---

## 💻 Hardware & System Notes

- **Deterministic Splits**: Dataset partitioning uses stratified 70/15/15 splits with a fixed seed (`random_state=42`) to prevent data leakage across epochs.
- **Windows Acceleration**: DataLoader `--num-workers 0` is set by default to avoid multiprocessing socket deadlocks on Windows systems.
- **GPU Acceleration**: CUDA devices are automatically selected if available; runs seamlessly on CPU for inference.
