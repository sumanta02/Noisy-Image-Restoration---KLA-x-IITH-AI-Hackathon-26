from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _msssim  # type: ignore[import-not-found]

    HAS_MSSSIM = True
except Exception:
    _msssim = None
    HAS_MSSSIM = False

try:
    import lpips as _lpips_pkg  # type: ignore[import-not-found]

    HAS_LPIPS = True
except Exception:
    _lpips_pkg = None
    HAS_LPIPS = False


@dataclass
class LossConfig:
    scale: int = 2
    h2_a: float = 0.14160734
    h2_b: float = 6.3383e-05
    student_t_nu: float = 9.0
    lambda_psnr: float = 1.0
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    lambda_charbonnier: float = 0.0
    lambda_fft: float = 0.0
    lambda_ssim: float = 0.10
    lambda_dc: float = 0.05
    ssim_contribute_to_loss: bool = True
    lambda_lpips: float = 0.0
    pixel_loss_type: str = "l1"
    charbonnier_eps: float = 1e-3
    lambda_edge: float = 0.0


def _as_tensor(value: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(value, device=ref.device, dtype=ref.dtype)


def forward_consistency_h2(x_hat: torch.Tensor, scale: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    H2 forward branch from inferred pipeline:
      mu = D(x_hat)
      q  = D(x_hat^2) / (scale^2)
    where D is area-like downsampling (avg pooling here).
    """

    mu = F.avg_pool2d(x_hat, kernel_size=scale, stride=scale)
    q = F.avg_pool2d(x_hat * x_hat, kernel_size=scale, stride=scale) / float(scale * scale)
    return mu, q


def heteroscedastic_variance_h2(q: torch.Tensor, h2_a: float | torch.Tensor, h2_b: float | torch.Tensor) -> torch.Tensor:
    a = _as_tensor(h2_a, q)
    b = _as_tensor(h2_b, q)
    return torch.clamp(b + a * q, min=1e-8)


def student_t_nll(y: torch.Tensor, mu: torch.Tensor, variance: torch.Tensor, nu: float = 9.0) -> torch.Tensor:
    if nu <= 2.0:
        raise ValueError("student_t_nu must be > 2")

    variance = torch.clamp(variance, min=1e-8)
    resid2 = (y - mu) ** 2

    nu_t = torch.tensor(nu, dtype=variance.dtype, device=variance.device)
    log_norm = torch.lgamma((nu_t + 1.0) / 2.0) - torch.lgamma(nu_t / 2.0) - 0.5 * torch.log(nu_t * torch.tensor(torch.pi, device=variance.device, dtype=variance.dtype))

    log_pdf = (
        log_norm
        - 0.5 * torch.log(variance)
        - 0.5 * (nu_t + 1.0) * torch.log1p(resid2 / (nu_t * variance))
    )
    return -log_pdf.mean()


def dc_loss_robust(y: torch.Tensor, mu: torch.Tensor, variance: torch.Tensor, nu: float = 9.0) -> torch.Tensor:
    """
    Robust Student-t-style data term without log(v), using detached variance weights.

    This prevents the model from exploiting variance-collapse pathways while retaining
    heavy-tail robustness in LR reprojection space.
    """

    if nu <= 0.0:
        raise ValueError("student_t_nu must be > 0")

    resid2 = (y - mu) ** 2
    v_w = torch.clamp(variance.detach(), min=1e-4)
    return torch.log1p(resid2 / (nu * v_w)).mean()


def ssim_loss(x_hat: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    if not HAS_MSSSIM or _msssim is None:
        return torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
    x_hat = x_hat.clamp(0.0, 1.0)
    x_gt = x_gt.clamp(0.0, 1.0)
    return 1.0 - _msssim(x_hat, x_gt, data_range=1.0, size_average=True)


def build_lpips_model(net: str = "alex", device: torch.device | None = None) -> torch.nn.Module | None:
    if not HAS_LPIPS or _lpips_pkg is None:
        return None

    model = _lpips_pkg.LPIPS(net=net, verbose=False)
    if device is not None:
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def lpips_loss(x_hat: torch.Tensor, x_gt: torch.Tensor, lpips_net: torch.nn.Module | None) -> torch.Tensor:
    if lpips_net is None:
        return torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

    x_hat = x_hat.clamp(0.0, 1.0)
    x_gt = x_gt.clamp(0.0, 1.0)

    # LPIPS expects RGB tensors in [-1, 1].
    x_hat = x_hat * 2.0 - 1.0
    x_gt = x_gt * 2.0 - 1.0
    if x_hat.shape[1] == 1:
        x_hat = x_hat.repeat(1, 3, 1, 1)
        x_gt = x_gt.repeat(1, 3, 1, 1)

    score = lpips_net(x_hat, x_gt)
    return score.mean()


def charbonnier_loss(x_hat: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("charbonnier_eps must be > 0")
    diff = x_hat - x_gt
    eps_t = torch.tensor(eps, device=diff.device, dtype=diff.dtype)
    return torch.sqrt(diff * diff + eps_t * eps_t).mean()


def edge_loss(x_hat: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    grad_x_hat = x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1]
    grad_x_gt = x_gt[:, :, :, 1:] - x_gt[:, :, :, :-1]
    grad_y_hat = x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :]
    grad_y_gt = x_gt[:, :, 1:, :] - x_gt[:, :, :-1, :]
    return F.l1_loss(grad_x_hat, grad_x_gt) + F.l1_loss(grad_y_hat, grad_y_gt)


def fourier_magnitude_loss(x_hat: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if eps <= 0:
        raise ValueError("eps must be > 0")

    x_hat_fft = torch.fft.rfft2(x_hat, norm="ortho")
    x_gt_fft = torch.fft.rfft2(x_gt, norm="ortho")
    mag_hat = torch.abs(x_hat_fft)
    mag_gt = torch.abs(x_gt_fft)

    # Log compression emphasizes relative structure and stabilizes bright-frequency dominance.
    return F.l1_loss(torch.log(mag_hat + eps), torch.log(mag_gt + eps))


def psnr_loss_official(x_hat: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
        """
        Match official NAFNet PSNRLoss objective:
            (10 / ln(10)) * ln(MSE + 1e-8)
        averaged across batch.
        """

        mse = ((x_hat - x_gt) ** 2).mean(dim=(1, 2, 3))
        scale = 10.0 / math.log(10.0)
        return scale * torch.log(mse + 1e-8).mean()


def combined_restoration_loss(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    y_lr: torch.Tensor,
    cfg: LossConfig,
    use_dc: bool,
    lpips_net: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    l_psnr = psnr_loss_official(x_hat, x_gt)
    l_l1 = F.l1_loss(x_hat, x_gt)
    l_l2 = F.mse_loss(x_hat, x_gt)
    l_charb = charbonnier_loss(x_hat, x_gt, eps=cfg.charbonnier_eps)
    l_fft = fourier_magnitude_loss(x_hat, x_gt)

    weighted_recon = (
        cfg.lambda_psnr * l_psnr
        + cfg.lambda_l1 * l_l1
        + cfg.lambda_l2 * l_l2
        + cfg.lambda_charbonnier * l_charb
    )

    # Backward compatibility: if no mixed recon weights are provided, use legacy one-of pixel_loss_type.
    if (cfg.lambda_psnr + cfg.lambda_l1 + cfg.lambda_l2 + cfg.lambda_charbonnier) > 0:
        recon = weighted_recon
    elif cfg.pixel_loss_type == "charbonnier":
        recon = l_charb
    elif cfg.pixel_loss_type == "psnr":
        recon = l_psnr
    else:
        recon = l_l1

    l_ssim = ssim_loss(x_hat, x_gt)
    if cfg.lambda_lpips > 0:
        l_lpips = lpips_loss(x_hat, x_gt, lpips_net=lpips_net)
    else:
        l_lpips = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
    if cfg.lambda_edge > 0:
        l_edge = edge_loss(x_hat, x_gt)
    else:
        l_edge = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

    if use_dc:
        mu, q = forward_consistency_h2(x_hat, scale=cfg.scale)
        v = heteroscedastic_variance_h2(q, h2_a=cfg.h2_a, h2_b=cfg.h2_b)
        l_dc = dc_loss_robust(y_lr, mu, v, nu=cfg.student_t_nu)
    else:
        l_dc = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

    total = recon + cfg.lambda_fft * l_fft + cfg.lambda_dc * l_dc
    if cfg.ssim_contribute_to_loss:
        total = total + cfg.lambda_ssim * l_ssim
    if cfg.lambda_lpips > 0:
        total = total + cfg.lambda_lpips * l_lpips
    if cfg.lambda_edge > 0:
        total = total + cfg.lambda_edge * l_edge

    logs = {
        "loss_total": float(total.detach().item()),
        "loss_recon": float(recon.detach().item()),
        "loss_psnr": float(l_psnr.detach().item()),
        "loss_l1_metric": float(l_l1.detach().item()),
        "loss_l1": float(l_l1.detach().item()),
        "loss_l2": float(l_l2.detach().item()),
        "loss_charbonnier": float(l_charb.detach().item()),
        "loss_fft": float(l_fft.detach().item()),
        "loss_ssim": float(l_ssim.detach().item()),
        "loss_dc": float(l_dc.detach().item()),
        "loss_lpips": float(l_lpips.detach().item()),
        "loss_edge": float(l_edge.detach().item()),
        "lambda_psnr": float(cfg.lambda_psnr),
        "lambda_l1": float(cfg.lambda_l1),
        "lambda_l2": float(cfg.lambda_l2),
        "lambda_charbonnier": float(cfg.lambda_charbonnier),
        "lambda_fft": float(cfg.lambda_fft),
        "lambda_dc": float(cfg.lambda_dc),
        "lambda_ssim": float(cfg.lambda_ssim),
        "lambda_lpips": float(cfg.lambda_lpips),
        "lambda_edge": float(cfg.lambda_edge),
        "pixel_loss_type": cfg.pixel_loss_type,
    }
    return total, logs
