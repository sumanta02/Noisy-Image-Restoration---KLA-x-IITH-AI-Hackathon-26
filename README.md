# Physically-Motivated Compound-Degradation Image Restoration 🏆

> [**2nd Place Overall (Team Name: TAUIG)**](https://drive.google.com/file/d/1J8r_s9TB1i0g8wlXXWLe6oYmtwyoM9_I/view?usp=sharing) at the KLA AI Hackathon, IIT Hyderabad '26
> - [**Public Leaderboard:**](https://drive.google.com/file/d/1ljUQov6SCQWBecOtH92l9pNomgWWjmEg/view?usp=sharing) Rank 13 (Score: 0.88614)
> - [**Private Leaderboard:**](https://drive.google.com/file/d/1yg0PYCcsbPbGYoGU24Yfj9zb5nMNqvGM/view?usp=sharing) Rank 10 (Score: 0.88303)
> - [**Final Rank (After Presentation):**](https://www.linkedin.com/feed/update/urn:li:activity:7462877666421923840/) Rank 2 🥈

## 📖 Overview

This repository contains our award-winning solution for reconstructing high-quality images from degraded, noisy, and low-resolution `.npy` inputs. The implementation includes the training loop, inference pipeline, Kaggle submission helper, experiment configs, and notebooks used during development.

![Noisy vs Ground Truth](visualisations/noisy_gt_comp.png)

Real-world physical degradation is complex. We tackled the challenge of **compound corruptions**—simultaneously handling Speckle Noise, Additive White Gaussian Noise (AWGN), and Block Mean Downsampling.

For a deeper technical walkthrough, see [`docs/methodology.md`](docs/methodology.md).

## 🚀 Key Innovations

### 1. NAFNetH2SR Architecture
We adapted the highly efficient **Nonlinear Activation Free Network (NAFNet)** for joint Super-Resolution ($2\times$) and compound denoising in grayscale.
- Replaced standard nonlinear activations (ReLU, GELU) with **SimpleGate** (element-wise multiplication).
- Achieved state-of-the-art restoration with drastically reduced MACs and computational complexity.

### 2. Physically-Motivated Loss Landscape
To prevent over-smoothing and enforce structural fidelity, we developed a dynamically weighted loss function:
- **Reconstruction:** Official NAFNet PSNR Loss + Charbonnier Loss for outlier robustness.
- **Perceptual \& Structural:** SSIM + LPIPS losses.
- **Heteroscedastic Data Consistency (DC):** A novel heavy-tailed Student-t formulation to robustly handle outlier pixels without variance collapse.

### 3. Curriculum-Based OOD Augmentation
Our model climbed from Rank 13 to Rank 10 on the private leaderboard thanks to our generalization strategy:
- **Phase 1:** Heavy synthetic OOD degradations (affine shifts, dynamic Gamma, 3x3 reflection blur, impulse noise, spatial cutouts) with early layers frozen.
- **Phase 2 \& 3:** Linear decay of transforms and complete un-freezing for convergence on the pure dataset distribution.

### 4. Inference Optimization
To squeeze maximum performance out of the model during evaluation:
- **Test-Time Augmentation (TTA):** Averaged predictions across 8-way spatial transformations (flips + rotations) to reduce variance.
- **Test-Time Refinement:** Employed a local Adam optimizer at inference to perform gradient steps directly on the predicted HR output, minimizing the DC loss against the specific test LR image.

## 📊 Results

- **Local Validation (2% Held-Out):** PSNR: 28.5 dB | SSIM: 0.77
- **Training Time:** ~75 minutes for a full 120-epoch run on 4× NVIDIA RTX A5000 GPUs.
- **OOD Generalization:** The model successfully generalized to unseen, severe degradations in the private test set, proving the effectiveness of our curriculum learning and physically-motivated loss.

![Denoised Samples](visualisations/contact_sheet_6_samples.png)

## 🗂️ Repository Structure

- `src/` - dataset loading, model wrapper, DDP helpers, and restoration losses.
- `configs/` - reproducible NAFNet training and inference presets.
- `scripts/` - setup, checkpoint inspection, dataset conversion, and visualization helpers.
- `notebooks/` - Kaggle-oriented exploratory and submission notebooks.
- `docs/` - technical methodology notes.
- `visualisations/` - lightweight figures used in the README.

Large artifacts are intentionally not tracked. Put local datasets in `data/`, trained checkpoints in `checkpoints/`, downloaded NAFNet weights in `nafnet/weights/`, and generated predictions in `outputs/`.

## ⚙️ Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/setup_nafnet_official.sh
bash scripts/download_nafnet_sidd_width64.sh
```

The active model wrapper uses the official NAFNet implementation at `nafnet/official`, which the setup script clones locally. That directory is ignored so the repository stays focused on the competition-specific code.

## 🏋️ Training

Expected training data layout:

```text
data/train/
├── GT/
│   └── 000000.npy
└── NoisyLR/
    └── 000000.npy
```

Run the default DDP training config:

```bash
torchrun --nproc_per_node=4 train_nafnet_ddp.py --config configs/train_nafnet_h2.yaml
```

For a single-process smoke run, set `--nproc_per_node=1` and reduce `epochs`, `batch_size`, and `num_workers` in the config or via CLI overrides.

## 🔎 Inference and Submission

Run local inference from a trained checkpoint:

```bash
python3 infer_nafnet_ddp.py --config configs/infer_nafnet_h2.yaml --checkpoint checkpoints/nafnet_h2/best_infer.pt
```

Create a Kaggle-style submission CSV from prediction `.npy` files:

```bash
python3 make_submission_csv.py --submission-dir outputs/nafnet_h2/test_predictions --output-csv submission.csv
```

On Kaggle, use the prepared inference wrapper:

```bash
python3 infer_nafnet_kaggle_best_infer.py
```

## 🧑‍💻 Team

**Team Name: TAUIG | Department of Artificial Intelligence, Indian Institute of Technology, Hyderabad**
- Supriyo Banerjea (AI24MTECH12005)
- Debanjan Das (AI24MTECH12009)
- Sumanta Manna (AI24MTECH12011)

## 📚 References
- Chen, L. et al., "Simple Baselines for Image Restoration (NAFNet)," ECCV, 2022.
- Zamir, S. W. et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR, 2022.
