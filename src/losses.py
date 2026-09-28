"""
losses.py
=========
训练损失：
  1. EDM video diffusion loss（可做嘴部空间加权 + 逐帧 mask）—— 把 AR 对齐到视频/像素视角，
     是口型与细节的主信号。
  2. motion regression 锚定 loss（生成 emb vs GT emb）—— 把 motion 钉在 motion_encoder 的
     合法分布上，防止 X-Nemo/SVD 冻结时被 reward-hack。
  3. 时序速度 loss（可选）—— 抑制 emb 抖动、保动态范围（缓解 over-smoothing）。

所有 loss 都支持 frame_mask（[B,T] 的有效帧 mask），以正确处理 pad 帧。
"""

from typing import Optional
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EDM scalings（与 Document 5 一致；抽出来供 forward 与 loss 复用）
# ---------------------------------------------------------------------------
def edm_scalings(sigmas: torch.Tensor):
    """sigmas: [B,1,1,1,1]。返回 c_skip, c_out, c_in, c_noise([B]), loss_weight。"""
    c_skip = 1.0 / (sigmas ** 2 + 1.0)
    c_out = -sigmas / (sigmas ** 2 + 1.0) ** 0.5
    c_in = 1.0 / (sigmas ** 2 + 1.0) ** 0.5
    c_noise = (sigmas.log() / 4).reshape(sigmas.shape[0])
    loss_weight = (sigmas ** 2 + 1.0) / sigmas ** 2
    return c_skip, c_out, c_in, c_noise, loss_weight


def sample_sigmas(batch_size: int, device, mode: str = "standard",
                  p_mean: float = 0.0, p_std: float = 1.0):
    """采样 EDM 噪声 sigma。
    mode:
      standard    : exp(N(0,1))         —— Document 5 默认
      detail      : exp(N(-1.2,1))      —— 偏低噪，强调口型/纹理等高频细节
      motion      : exp(N(0.7,1.6))     —— 偏高噪，强调大幅度运动结构
      custom      : exp(N(p_mean,p_std))
    口型精调阶段推荐 detail。
    """
    n = torch.randn([batch_size, 1, 1, 1, 1], device=device)
    if mode == "standard":
        return n.exp()
    if mode == "detail":
        return (n - 1.2).exp()
    if mode == "motion":
        return (n * 1.6 + 0.7).exp()
    if mode == "custom":
        return (n * p_std + p_mean).exp()
    raise ValueError(f"unknown sigma mode: {mode}")


# ---------------------------------------------------------------------------
# 嘴部空间权重图（latent 分辨率）
# ---------------------------------------------------------------------------
def build_mouth_weight_map(mode: str, latent_h: int, latent_w: int,
                           mouth_gain: float, device,
                           masks: Optional[torch.Tensor] = None,
                           region=(0.55, 0.92, 0.28, 0.72)) -> Optional[torch.Tensor]:
    """返回空间权重 w，形状 [1,1,1,H,W] 或 [B,T,1,H,W]；None 表示均匀（不加权）。
    最终对 (denoised-target)^2 逐元素乘 w。w = 1 + mouth_gain * mouth_mask。

    mode:
      uniform : 不加权（返回 None）
      region  : 下半脸中部矩形启发式（无需额外数据；假设人脸大致居中且占满画面，
                这对裁剪过的说话头数据成立）。region=(h0,h1,w0,w1) 为归一化比例。
      mask    : 用传入的逐帧嘴部掩码 masks（[B,T,1,H,W]，已下采样到 latent 分辨率），最精确。
    """
    if mode == "uniform" or mouth_gain == 0:
        return None
    if mode == "region":
        m = torch.zeros(1, 1, 1, latent_h, latent_w, device=device)
        h0, h1, w0, w1 = region
        m[..., int(latent_h * h0):int(latent_h * h1), int(latent_w * w0):int(latent_w * w1)] = 1.0
        return 1.0 + mouth_gain * m
    if mode == "mask":
        assert masks is not None, "mode='mask' 需要传入逐帧嘴部掩码"
        return 1.0 + mouth_gain * masks.to(device)
    raise ValueError(f"unknown mouth weight mode: {mode}")


# ---------------------------------------------------------------------------
# EDM video loss
# ---------------------------------------------------------------------------
def edm_video_loss(model_pred: torch.Tensor,
                   noisy_latents: torch.Tensor,
                   target_latents: torch.Tensor,
                   sigmas: torch.Tensor,
                   weight_map: Optional[torch.Tensor] = None,
                   frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    model_pred / noisy_latents / target_latents: [B,T,C,H,W]
    sigmas: [B,1,1,1,1]
    weight_map: [1,1,1,H,W] 或 [B,T,1,H,W]，None=均匀
    frame_mask: [B,T]，None=全有效
    """
    c_skip, c_out, _, _, loss_weight = edm_scalings(sigmas)
    denoised = c_out * model_pred + c_skip * noisy_latents        # [B,T,C,H,W]
    se = (denoised - target_latents) ** 2                         # [B,T,C,H,W]
    se = se * loss_weight                                         # EDM 权重 [B,1,1,1,1]

    if weight_map is not None:
        se = se * weight_map                                      # 嘴部空间加权
        norm = weight_map.expand_as(se)
    else:
        norm = torch.ones_like(se)

    if frame_mask is not None:
        fm = frame_mask.view(frame_mask.shape[0], frame_mask.shape[1], 1, 1, 1)
        se = se * fm
        norm = norm * fm

    return se.sum() / norm.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# DDPM/DDIM video loss（X-Nemo backbone）—— epsilon 或 v 预测，latent 排布 [B,C,F,H,W]
# ---------------------------------------------------------------------------
def ddpm_video_loss(model_pred: torch.Tensor,
                    target: torch.Tensor,
                    weight_map: Optional[torch.Tensor] = None,
                    frame_mask: Optional[torch.Tensor] = None,
                    frame_dim: int = 2) -> torch.Tensor:
    """
    model_pred / target: [B,C,F,H,W]（X-Nemo denoising_unet 的预测与目标，epsilon 或 v）
    weight_map: [1,1,1,H,W] 或 [B,1,F,H,W]，None=均匀（嘴部加权用）
    frame_mask: [B,F]，None=全有效；frame_dim 指明 F 所在维（X-Nemo 为 2）
    """
    se = (model_pred - target) ** 2                              # [B,C,F,H,W]
    if weight_map is not None:
        se = se * weight_map
        norm = weight_map.expand_as(se)
    else:
        norm = torch.ones_like(se)
    if frame_mask is not None:
        B, Fr = frame_mask.shape
        shape = [1] * se.dim()
        shape[0] = B
        shape[frame_dim] = Fr
        fm = frame_mask.view(shape)
        se = se * fm
        norm = norm * fm
    return se.sum() / norm.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# motion regression 锚定 loss
# ---------------------------------------------------------------------------
def motion_regression_loss(gen_motion: torch.Tensor,
                           gt_motion: torch.Tensor,
                           frame_mask: Optional[torch.Tensor] = None,
                           kind: str = "smooth_l1") -> torch.Tensor:
    """gen_motion / gt_motion: [B,T,motion_dim]，必须在同一空间（建议都为归一化空间）。
    kind: mse | smooth_l1（smooth_l1 对离群更稳，常用于回归锚定）。
    """
    if gen_motion.shape != gt_motion.shape:
        T = min(gen_motion.shape[1], gt_motion.shape[1])
        gen_motion, gt_motion = gen_motion[:, :T], gt_motion[:, :T]
        if frame_mask is not None:
            frame_mask = frame_mask[:, :T]

    if kind == "mse":
        per = (gen_motion - gt_motion) ** 2
    elif kind == "smooth_l1":
        per = F.smooth_l1_loss(gen_motion, gt_motion, reduction="none", beta=1.0)
    else:
        raise ValueError(kind)
    per = per.mean(dim=-1)                                        # [B,T]

    if frame_mask is not None:
        return (per * frame_mask).sum() / frame_mask.sum().clamp(min=1.0)
    return per.mean()


def temporal_velocity_loss(gen_motion: torch.Tensor,
                           gt_motion: torch.Tensor,
                           frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """一阶差分对齐：让生成 motion 的逐帧变化（速度/动态范围）贴近 GT。
    缓解 over-smoothing（std_ratio 偏低），对口型动态尤其有用。"""
    T = min(gen_motion.shape[1], gt_motion.shape[1])
    g = gen_motion[:, :T]; t = gt_motion[:, :T]
    dv_g = g[:, 1:] - g[:, :-1]
    dv_t = t[:, 1:] - t[:, :-1]
    per = ((dv_g - dv_t) ** 2).mean(dim=-1)                       # [B,T-1]
    if frame_mask is not None:
        fm = (frame_mask[:, 1:T] * frame_mask[:, :T - 1])
        return (per * fm).sum() / fm.sum().clamp(min=1.0)
    return per.mean()


@torch.no_grad()
def std_ratio(gen_motion: torch.Tensor, gt_motion: torch.Tensor) -> float:
    """监控指标：生成 motion 与 GT 的逐维时间 std 之比（≈1 为佳）。"""
    T = min(gen_motion.shape[1], gt_motion.shape[1])
    gs = gen_motion[:, :T].float().std(dim=1).mean()
    ts = gt_motion[:, :T].float().std(dim=1).mean()
    return (gs / (ts + 1e-8)).item()