# Methodology: Super Resolution + Denoising for Semiconductor Imagery

This document provides a technical overview of the methodology employed for the KLA AI Hackathon task: joint **Super Resolution (SR)** and **Denoising**.

## 1. Task Objective
The mission was to reconstruct high-fidelity, high-resolution (HR) semiconductor images from noisy, low-resolution (LR) inputs. The dataset presented unique challenges characteristic of electron microscopy or high-end fabrication inspection tools, where noise is signal-dependent and resolution is limited by sensor constraints.

## 2. Model Architecture: NAFNet (Non-linear Activation Free Network)
Our solution utilizes **NAFNet** as the primary restoration engine. NAFNet is a state-of-the-art model that achieves superior performance by simplifying the network architecture:
- **Activation-Free**: It replaces standard activations (like ReLU) with multiplication-based gates, reducing complexity while enhancing feature representational power.
- **SR adaptation**: We extended the base NAFNet (standardized on the SIDD-width64 configuration) with a bilinear upsampling stem to perform the 2x super-resolution mapping.
- **Grayscale Processing**: The network was optimized for single-channel input/output to match the grayscale nature of the semiconductor data.

## 3. Probabilistic Noise Modeling: The H2 Framework
A critical component of our methodology is the **Heteroscedastic Noise Model (H2)**. Unlike simple Gaussian noise, semiconductor imagery noise scales with signal intensity ($I$):
$$\sigma^2 = a \cdot I + b$$
Where $a$ and $b$ are parameters representing Poisson (shot) and Gaussian (electronic) noise components.

### Forward Consistency Branch
We implemented a **Forward Consistency** objective to ensure the model's high-resolution predictions are physically plausible. The network's HR prediction ($\hat{x}$) is projected back into the LR domain using area-downsampling. This re-projected signal is then compared to the original noisy input ($y$) using a robust likelihood function:
- **Robust Data Consistency (DC) Loss**: We use a Student-t Negative Log-Likelihood (NLL) term. This provides higher robustness to outliers in the noise distribution compared to standard MSE.

## 4. Multi-Faceted Optimization Objective
The training process was guided by a composite loss function designed to balance pixel accuracy, structural patterns, and frequency details:

| Loss Term | Description |
| :--- | :--- |
| **Restoration (PSNR/L1)** | Direct pixel-wise convergence to the ground truth. |
| **Frequency (FFT)** | L1 loss on Fourier magnitude spectra to recover global patterns and periodic textures. |
| **Data Consistency (DC)** | Enforces the H2 noise model in the re-projected LR space. |
| **Structural (SSIM)** | Enhances edge preservation and structural similarity. |
| **Perceptual (LPIPS)** | Late-stage refinement using AlexNet-based perceptual features for visual "crispness". |

## 5. Training Strategy
- **Curriculum Learning**: We employed an augmentation curriculum that starts with basic spatial transforms and ramps up to heavy Out-of-Distribution (OOD) noise and blur transforms.
- **Adaptive Scheduling**: The weight of the Data Consistency term ($\lambda_{dc}$) was dynamically adjusted based on validation PSNR performance to prevent over-smoothing.
- **DDP Training**: Leveraging PyTorch’s Distributed Data Parallel (DDP) for high-batch training stability beyond 400,000 iterations.
- **Staged Freezing**: During fine-tuning on high-resolution targets, earlier layers were progressively frozen to allow the upsampling layers specialized learning.

## 6. Inference Pipeline
For competition submission, the model acts as an end-to-end Restoration-SR pipeline. For images exceeding standard patch sizes, a tiled inference strategy with overlapping regions and windowed averaging is used to prevent boundary artifacts.
