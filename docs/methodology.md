# Technical Methodology: NAFNet H2 Restoration Pipeline

This note is the implementation-level companion to the [project README](../README.md). It documents the mathematical assumptions, training behavior, and inference choices encoded in the repository. Where the code makes a practical approximation, this document says so rather than presenting it as a recovered physical truth.

## 1. Data contract and notation

Let `x` be a clean high-resolution image, `y` a noisy low-resolution observation, and `x_hat = f_theta(y)` the predicted restoration. Under the default 2x setting,

```math
x, \hat{x} \in [0,1]^{H \times W}, \qquad y \in \mathbb{R}^{(H/2) \times (W/2)}.
```

Training pairs are discovered by matching `.npy` filenames in `data/train/GT/` and `data/train/NoisyLR/`. The loader accepts `H x W`, `C x H x W`, and `H x W x C` arrays, canonicalizes them to `C x H x W`, aligns the pair spatially, and samples matched crops. HR crop sizes are multiples of `scale * 8`, keeping LR patches compatible with NAFNet's four downsampling levels. GT is clipped to `[0,1]`; the noisy LR input is otherwise kept as supplied.

The held-out validation split is a deterministic 10% filename split under seed 42. Validation uses no spatial or synthetic augmentation.

## 2. Grayscale adaptation of the official NAFNet

[`src/model.py`](../src/model.py) wraps the official NAFNet implementation rather than replacing its restoration blocks. The one-channel LR input is first bilinearly resized, repeated to RGB to match the SIDD-pretrained checkpoint, restored by NAFNet, averaged back to one channel, cropped to the exact target shape, and clamped:

```math
\hat{x} = \mathrm{clip}_{[0,1]}\left[
\frac{1}{3}\sum_{c=1}^{3}
\mathcal{B}_{\theta}\left(\mathrm{rep}_3\left(U_2(y)\right)\right)_c
\right].
```

`U_2` is bilinear interpolation with `align_corners=False`, `rep_3` repeats the grayscale channel, and `B_theta` is the official RGB NAFNet backbone. The default `sidd-width64` preset uses encoder block counts `(2, 2, 4, 8)`, 12 middle blocks, and decoder counts `(2, 2, 2, 2)`.

Each official NAFBlock uses LayerNorm, pointwise expansion, depthwise `3 x 3` convolution, `SimpleGate` channel multiplication, simplified channel attention, and residual scalars initialized to zero. The backbone retains its own RGB global residual path and U-Net skip connections. The wrapper's repeat-and-average adapters preserve checkpoint compatibility while allowing end-to-end fine-tuning for a grayscale target.

Pretrained loading accepts common NAFNet state-dict containers (`params_ema`, `params`, `model`, or a root state dict), removes `module.` and `backbone.` prefixes, and reports missing or unexpected keys. `strict_pretrained: false` allows diagnostic partial loading; inference checkpoints written by this repository load strictly.

## 3. H2 forward model and heteroscedasticity

The data-consistency branch assumes a 2x block-mean forward operator. Let `D_s` denote average pooling over non-overlapping `s x s` blocks. The implementation evaluates

```math
\mu = D_s(\hat{x}), \qquad
q = \frac{D_s(\hat{x} \odot \hat{x})}{s^2}, \qquad
v = \max(b + a q, 10^{-8}).
```

For the supplied configuration, `s = 2`, `a = 0.14160734`, and `b = 0.000063383`. The `q` expression includes an additional division by `s^2`; it is the pipeline's scaled local-energy statistic, not an ordinary second moment with a renamed variable.

The variance `v` changes with predicted local energy, which is why this is a *heteroscedastic* formulation. The `a q` term encodes a signal-dependent component consistent with speckle-like degradation, while `b` establishes an intensity-independent floor consistent with residual read/AWGN-like noise. A homoscedastic reprojection objective would penalize a mismatch in a dark smooth area and a bright high-energy area equally even when the assumed noise scale differs.

This is a fixed calibration model, not a learned uncertainty head. `a` and `b` are supplied hyperparameters. The training loop can optionally perturb them multiplicatively for calibration robustness, but the checked-in config disables this with `h2_jitter: 0.0`.

### Formal Student-t density versus the optimized objective

[`src/losses.py`](../src/losses.py) contains a complete Student-t negative log likelihood:

```math
-\log p(y_i \mid \mu_i, v_i) = C(\nu) + \frac{1}{2}\log v_i
+ \frac{\nu + 1}{2}\log\left(1 + \frac{(y_i-\mu_i)^2}{\nu v_i}\right).
```

That is not the term optimized by `combined_restoration_loss`. The active data-consistency regularizer is instead

```math
\mathcal{L}_{\mathrm{DC}} = \frac{1}{N}\sum_i
\log\left(1 + \frac{(y_i - \mu_i)^2}
{\nu\,\mathrm{stopgrad}(\max(v_i,10^{-4}))}\right).
```

The default config uses `nu = 3`. The active term omits the `log(v)` contribution and detaches `v` before it becomes a residual weight. Consequently, the prediction cannot lower this part of the loss by changing `q` to manipulate a differentiable variance path. It retains robust, heavy-tailed weighting in LR reprojection space without exposing a variance-collapse or variance-inflation shortcut. It is a regularizer alongside paired HR supervision, not the only learning signal.

## 4. Composite restoration objective

The generic objective implemented in [`src/losses.py`](../src/losses.py) is

```math
\begin{aligned}
\mathcal{L} ={}& \lambda_P\mathcal{L}_{\mathrm{PSNR}}
+ \lambda_1\mathcal{L}_1
+ \lambda_2\mathcal{L}_2
+ \lambda_C\mathcal{L}_{\mathrm{Charb}}
+ \lambda_F\mathcal{L}_{\mathrm{FFT}} \\
&+ \lambda_D\mathcal{L}_{\mathrm{DC}}
+ \lambda_S(1-\mathrm{SSIM})
+ \lambda_L\mathcal{L}_{\mathrm{LPIPS}}
+ \lambda_E\mathcal{L}_{\mathrm{edge}}.
\end{aligned}
```

Not every term is active in every experiment:

- `L_PSNR = (10 / ln 10) mean(log(MSE + 1e-8))`; minimizing it is equivalent to maximizing PSNR.
- `L_Charb = mean(sqrt((x_hat - x)^2 + epsilon^2))` is a smooth robust pixel penalty.
- `L_FFT` is L1 distance between log magnitudes of orthonormal 2D real FFTs, so relative structure is emphasized without allowing bright frequencies to dominate.
- `1 - SSIM` is evaluated on clamped `[0,1]` images. A nonzero requested term requires `pytorch-msssim`.
- LPIPS repeats grayscale to RGB and maps it to `[-1,1]` for its frozen comparison network. Edge loss is L1 difference of horizontal and vertical finite differences.

### Default 120-epoch schedule

The `roi_psnr_120e` preset in [`train_nafnet_ddp.py`](../train_nafnet_ddp.py) overrides the static YAML weights. The table gives anchor values; intermediate values are linearly interpolated.

| Epoch anchor | PSNR | L1 | L2 | Charbonnier | FFT | DC | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 to 19 | 1.0 | 0.050 | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 |
| 20 | 1.0 | 0.050 | 0.010 | 0.010 | 0.000 | 0.000 | 0.000 |
| 90 | 1.0 | 0.030 | 0.005 | 0.008 | 0.010 | 0.010 | 0.015 |
| final | 1.0 | 0.020 | 0.000 | 0.005 | 0.015 | 0.008 | 0.020 |

LPIPS and edge loss are available but disabled in the checked-in config. The schedule establishes accurate pixel reconstruction before introducing stronger frequency, consistency, and structural constraints. The optional adaptive DC scheduler responds to validation PSNR, but the supplied preset uses the stage-defined DC values.

## 5. Augmentation and optimization

Geometric augmentation applies independent horizontal flips, vertical flips, and rotations by multiples of 90 degrees to both elements of a pair. The OOD augmentation is input-only: it composes a random subset of affine gain/bias, gamma, Gaussian noise, multiplicative speckle, Poisson noise, reflected `3 x 3` blur, stripe noise, impulse noise, and cutout corruption on the LR observation, while leaving the GT clean.

For the default `max` profile over 120 epochs:

- Epochs 0 to 19 use the `strong` profile, OOD probability capped at 0.70, and at most 3 synthetic operators.
- Epochs 20 to 89 use `max`, OOD probability 0.95, and at most 5 operators.
- Epochs 90 onward return to `strong`, with probability decaying from 0.80 toward 0.50 and the operator count from 3 toward 2.

The code supports staged freezing of the NAFNet introduction and early encoder blocks. It is disabled by default (`staged_freeze: false`), so the provided run fine-tunes all backbone parameters throughout.

Optimization uses AdamW with `beta=(0.9, 0.9)`, learning rate `5e-4`, zero weight decay, a three-epoch warm-up from 5% of the base rate, and cosine annealing to `1e-7`. The checked-in training config disables AMP. Distributed runs use `DistributedSampler`, rank-specific seed `42 + rank`, and reduced validation metrics. `best.pt` is selected by validation PSNR and `best_infer.pt` is a compact inference export.

## 6. Inference and submission serialization

With `checkpoint: auto`, inference tries `best_lpips_infer.pt`, `best_infer.pt`, `best_lpips.pt`, `best.pt`, `latest_infer.pt`, then `latest.pt`. Every output is validated as finite, one-channel, and `256 x 256` before it is saved.

The quality presets are:

| Preset | Test-time augmentation | Refinement |
| --- | --- | --- |
| `fast` | none | none |
| `balanced` | four flip variants | 3 steps at `8e-4` |
| `high` | eight flip/transpose variants | 8 steps at `5e-4` |

TTA averages predictions after applying each transform and its inverse. Refinement initializes an optimizable HR tensor from the NAFNet prediction and runs Adam directly on it to minimize the same robust data-consistency term against `y`, clamping after every step. It has no additional image prior; the initial NAFNet output is the prior, so the step count is deliberately small.

The submission writer sorts prediction filenames, verifies shape and finite values, serializes each array with `numpy.save`, base64-encodes the bytes, and writes one-indexed `id,npy_base64` rows. This retains exact NumPy array serialization instead of flattening pixel values into CSV text.

## 7. Limits and reproducibility boundary

The six checked-in panels in [`outputs/presentation_qualitative_png/`](../outputs/presentation_qualitative_png/) show the pipeline's visual output on test observations. They do not contain test ground truth and are not a metric claim. Datasets, checkpoints, raw predictions, and submissions are intentionally excluded from version control; reproducing a run requires the competition data and the downloaded NAFNet SIDD checkpoint described in the [README](../README.md).
