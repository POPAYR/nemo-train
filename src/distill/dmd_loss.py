"""
Phase 3: DMD2 蒸馏的 ε/DDPM 空间核心数学（DECODER_DISTILL_PLAN.md §Phase 3 / §6）
=================================================================================
X-Nemo 是 ε-pred DDPM，不用 Self-Forcing 的 flow。DMD 在 x0 空间，所以：
  ① 网络出 ε → 转 x0；② DMD grad = (x0_fake − x0_real_cfg)/norm；
  ③ L_G = 0.5·MSE(x0, sg(x0 − grad))；④ critic = ε-MSE 去噪 loss。
本模块只放纯函数（不依赖具体网络），便于用 dummy 单测数学正确性。
"""
import torch
import torch.nn.functional as F


def eps_to_x0(x_t, eps, t, alphas_cumprod):
    """DDPM 反演：x0 = (x_t − sqrt(1−ā_t)·ε) / sqrt(ā_t)。
    fp32 内算（高 t 下 sqrt(ā) 很小，bf16 会丢精度），结果转回 x_t.dtype。"""
    a = alphas_cumprod[t].view([-1] + [1] * (x_t.dim() - 1)).float()
    x0 = (x_t.float() - (1 - a).sqrt() * eps.float()) / a.sqrt()
    return x0.to(x_t.dtype)


def x0_to_eps(x_t, x0, t, alphas_cumprod):
    """逆向：ε = (x_t − sqrt(ā_t)·x0) / sqrt(1−ā_t)。"""
    a = alphas_cumprod[t].view([-1] + [1] * (x_t.dim() - 1)).float()
    eps = (x_t.float() - a.sqrt() * x0.float()) / (1 - a).sqrt()
    return eps.to(x_t.dtype)


def dmd_kl_grad(x0, x0_fake, x0_real, normalize=True):
    """DMD grad（DMD eq.7-8）：grad = (x0_fake − x0_real) / mean|x0 − x0_real|。
    全部已是 x0 空间张量 [B,F,C,H,W]。返回 grad（与 x0 同形）。"""
    grad = x0_fake - x0_real
    if normalize:
        norm = (x0 - x0_real).abs().mean(dim=list(range(1, x0.dim())), keepdim=True).clamp_min(1e-8)
        grad = grad / norm
    return torch.nan_to_num(grad)


def real_score_cfg_x0(x_t, t, eps_cond, eps_uncond, alphas_cumprod, guidance_scale):
    """teacher(real_score) 带 CFG：ε_cfg = ε_uncond + s·(ε_cond − ε_uncond) → x0。"""
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps_to_x0(x_t, eps, t, alphas_cumprod)


def dmd_generator_loss(x0, x0_fake, x0_real, gradient_mask=None):
    """L_G = 0.5·MSE(x0, sg(x0 − grad))；对 x0 求导即得 grad（DMD 技巧）。
    grad 在 no_grad 下算好（x0_fake/x0_real detached）；只有这里对 x0 可微。"""
    grad = dmd_kl_grad(x0, x0_fake, x0_real).detach()
    target = (x0 - grad).detach()
    if gradient_mask is not None:
        return 0.5 * F.mse_loss(x0.double()[gradient_mask], target.double()[gradient_mask])
    return 0.5 * F.mse_loss(x0.double(), target.double())


def critic_eps_loss(eps_pred, eps_true, mask=None):
    """critic 去噪 loss：ε-MSE（X-Nemo 原生）。eps_*: [B,F,C,H,W]。"""
    se = (eps_pred.float() - eps_true.float()) ** 2
    if mask is not None:
        return (se * mask).sum() / mask.expand_as(se).sum().clamp_min(1)
    return se.mean()


# --------------------------- GAN（DMD2 / 官方 Self-Forcing gan.py 迁移）---------------------------
# 官方判别器 = fake_score(critic) 主干 + 轻量 cls 头（在深层特征上出 logit）；非饱和 softplus 损失，
# 权重 1e-2，叠加到 DMD 之上。我们把 DiT 深层特征换成 UNet mid-block(瓶颈) 特征，逐帧出 logit。
def gan_g_loss(fake_logit):
    """生成器对抗损失（非饱和）：softplus(−D(fake))，越骗过 D 越小。logit 任意形状。"""
    return F.softplus(-fake_logit.float()).mean()


def gan_d_loss(real_logit, fake_logit):
    """判别器损失（非饱和）：softplus(−D(real)) + softplus(D(fake))。"""
    return F.softplus(-real_logit.float()).mean() + F.softplus(fake_logit.float()).mean()


# --------------------------- Relativistic GAN + R1/R2（Self-Forcing paper Eq.5-7）---------------------------
# paper「for all experiments」用的稳定配方：相对判别器 + 有限差分 R1/R2（λ=30, σ=0.05）。
# 小 batch 下用正则替代大 batch 的稳定作用（我们 eff-batch 远小于 paper 的 768）。
def gan_g_loss_rel(fake_logit, real_logit):
    """相对 G：LG = −log σ(D(fake)−D(real)) = softplus(D(real)−D(fake))。generator 要 fake 相对 real 更真。"""
    return F.softplus((real_logit.float() - fake_logit.float())).mean()


def gan_d_loss_rel(real_logit, fake_logit):
    """相对 D：LD = −log σ(D(real)−D(fake)) = softplus(D(fake)−D(real))。D 要 real 相对 fake 更真。"""
    return F.softplus((fake_logit.float() - real_logit.float())).mean()


def gan_reg_r1r2(logit_real, logit_real_pert, logit_fake, logit_fake_pert):
    """paper Eq.5：Lreg = 0.5·(‖D(x)−D(x+σε)‖² + ‖D(x̂)−D(x̂+σε̂)‖²)。有限差分近似 R1+R2 梯度惩罚，
    鼓励判别器对真/假 latent 的小扰动稳定 → 平滑决策面、抗 mode collapse。σ 已在扰动里加，这里只算差方。"""
    r1 = ((logit_real.float() - logit_real_pert.float()) ** 2).mean()
    r2 = ((logit_fake.float() - logit_fake_pert.float()) ** 2).mean()
    return 0.5 * (r1 + r2)


# --------------------------- 自测（dummy，验证数学）---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    acp = torch.cumprod(1 - betas, 0)
    B, Fr, C, H, W = 2, 4, 4, 8, 8
    x0 = torch.randn(B, Fr, C, H, W)
    t = torch.randint(20, 980, (B,))

    # ① eps↔x0 往返一致性
    eps = torch.randn_like(x0)
    a = acp[t].view(B, 1, 1, 1, 1)
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
    x0_rec = eps_to_x0(x_t, eps, t, acp)
    eps_rec = x0_to_eps(x_t, x0, t, acp)
    print(f"[①] eps→x0 误差={F.mse_loss(x0_rec, x0):.2e}  x0→eps 误差={F.mse_loss(eps_rec, eps):.2e}  (应~0)")

    # ② DMD loss 对 x0 的梯度应等于 grad（DMD 技巧核心不变量）
    x0g = x0.clone().requires_grad_(True)
    x0_fake = torch.randn_like(x0); x0_real = torch.randn_like(x0)
    grad_expected = dmd_kl_grad(x0g.detach(), x0_fake, x0_real)
    loss = dmd_generator_loss(x0g, x0_fake, x0_real)
    loss.backward()
    err = (x0g.grad.double() - grad_expected.double() / x0g.numel()).abs().max()
    # d/dx0 [0.5*mean((x0-target)^2)] = (x0-target)/N = grad/N
    print(f"[②] dmd loss 梯度 vs grad/N 最大误差={err:.2e}  (应~0 → 验证 DMD 技巧正确)")

    # ③ CFG real x0
    x0_realc = real_score_cfg_x0(x_t, t, eps, eps * 0.9, acp, guidance_scale=2.5)
    print(f"[③] real_score_cfg_x0 形状={tuple(x0_realc.shape)}  ok")
    print("DMD ε-空间数学自测完成")
