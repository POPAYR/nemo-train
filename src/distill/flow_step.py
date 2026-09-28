"""
flow-space DMD 一步:generator_loss / critic_loss(rectified-flow 版 dmd_step.py)
=================================================================================
与 dmd_step 同构,DDPM ε → flow v:
  - rollout:flow_rollout(少步 Euler,σ 网格),全长 L,随机 grad_window 帧带梯度。
  - DMD 打分:在窗内 x0 上 flow 加噪 z_σ=(1-σ)x0+σε(σ 随机 ∈[σ_low,σ_high]);
    teacher/critic 出 v→x0;DMD grad=(x0_fake−x0_real)/norm(x0 空间,不变)。
  - critic 去噪 loss:**v-MSE**(target v=ε−x0),让 critic 成为 student 分布的 score。
  - GAN(可选,DMD2 锐化):disc_head 挂 critic mid-block,逐帧 logit,同 ε 版。
DMD 打分窗 ≤24 帧(teacher PE max 32,分布内、覆盖 block 边界)。
"""
import torch
import torch.nn.functional as F
from .flow_rollout import flow_rollout
from .flow_math import flow_add_noise, v_target
from .dmd_loss import (dmd_generator_loss, critic_eps_loss, gan_g_loss, gan_d_loss,
                       gan_g_loss_rel, gan_d_loss_rel, gan_reg_r1r2)

DMD_WINDOW = 24
# DMD 打分的 σ 采样区间。⭐ 上限 0.70 是**实测定的**(scripts/val/diag_dmd_direction.py):
# 对本 teacher,DMD 目标 (x0-grad) 在 σ≲0.72 处比 x0 更锐(Δ>0,锐化),σ≳0.75 处更糊(Δ<0)——
# σ→1 时 teacher 的 x0 估计塌成模糊条件均值(std 0.58 vs 数据 1.05),DMD 差分信号被噪声主导。
# 用 uniform[0.02,0.98] 时约 28% 样本落在有害区,净效应≈0;shift=5(官方值)几乎全落有害区 → 快速变糊。
# 官方 timestep_shift=5.0 不可照搬:Wan 本身用 shift=5 训练,σ 参数化不同;我们的 teacher 是
# shift=1 + logit-normal(集中 0.5)训的,中 σ 才是它的好区。
SIGMA_LOW, SIGMA_HIGH = 0.02, 0.70
SCORE_SHIFT = 1.0                    # 1.0=不变换(实测优于官方 5.0)


def _add_flow_noise(x0, sig_low=SIGMA_LOW, sig_high=SIGMA_HIGH, shift=SCORE_SHIFT):
    """在 x0 上采随机 σ 做 flow 加噪 → (z, sigma, noise, v_tgt)。
    σ ~ U(0,1) →(可选 shift 映射 σ'=s·σ/(1+(s-1)·σ))→ 线性映到 [sig_low, sig_high]。
    用线性映射而非 clamp:clamp 会在上限处堆出一个质量尖峰。"""
    B = x0.shape[0]
    sigma = torch.rand(B, device=x0.device)
    if shift > 1.0:
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    sigma = sig_low + sigma * (sig_high - sig_low)
    n = torch.randn_like(x0)
    z = flow_add_noise(x0, n, sigma).to(x0.dtype)
    return z, sigma, n, v_target(x0, n)


def generator_loss(M, noise, clip_emb, motion, sigma_list,
                   block_size=8, grad_window=DMD_WINDOW, guidance_scale=0.0, gan_g_weight=0.0,
                   real_latent=None, relativistic=False, score_shift=SCORE_SHIFT, sigma_high=SIGMA_HIGH,
                   cfg_sigma_cut=1.0, reg_target=None, reg_weight=0.0):
    # 1) flow rollout(全长 L;随机连续 grad_window 帧窗保留梯度)
    x0, win = flow_rollout(M, noise, clip_emb, motion, sigma_list,
                           block_size=block_size, grad_window=grad_window)
    k, Wn = win
    x0_win = x0[:, :, k:k + Wn]
    mot_win = motion[:, k:k + Wn]
    # 2) DMD KL grad(全 no_grad;teacher/critic 只看 Wn≤24 帧)
    with torch.no_grad():
        z, sigma, _, _ = _add_flow_noise(x0_win.detach(), sig_high=sigma_high, shift=score_shift)
        _, x0_fake = M.forward_net_flow(M.critic, z, sigma, clip_emb, mot_win)
        _, x0_real = M.forward_net_flow(M.teacher, z, sigma, clip_emb, mot_win)
        # ★ 分段 CFG(1.x-Distill arXiv:2604.04018 式4):高噪段**完全关闭** CFG,低噪段保留。
        #   官方:s_real = s_∅ + w(s_c − s_∅)  if t ≤ α  ;  = s_c  if t > α   (α=0.94, w=7.0, shift=3.0)
        #   动机:结构在高噪段形成,强引导把结构钉死在主模态 → mode collapse。
        #   ⚠️ α 不照抄 0.94:他们 shift=3.0 我们 shift=1.0,且论文中 t 是 shift 前/后有歧义;
        #      照抄只命中我们采样区间约 4%,近乎空操作。实测阈值见 scripts/val/diag_cfg_vs_sigma.py:
        #      σ≳0.7 起 CFG 把目标 std 推离数据(+17%→σ=0.98 时 +79%,而数据 std≈1.05)。
        #   _add_flow_noise 是**每样本一个 σ**([B]),故这里按样本掩码,不是逐元素。
        if guidance_scale > 0 and cfg_sigma_cut < 1.0:
            keep = (sigma <= cfg_sigma_cut)                       # [B] bool:True=开 CFG
            if not bool(keep.any()):
                guidance_scale = 0.0                             # 本 batch 全在高噪段 → 跳过 uncond 前向
        else:
            keep = None
        if guidance_scale > 0:
            # ★ CFG 负分支 = 原 XNeMo pipeline 的**三重置空**,缺一不可(见 pipeline_pose2vid_motenc_long.py):
            #   (1) reference bank 不注入 —— r_teacher.clear() 后 bank_fea=[],attn1 退化为纯自注意力
            #       (mutual_self_attention.py:159-168),即"无参考图空间特征"
            #   (2) CLIP embedding 置零 —— 无外观提示
            #   (3) motion 用**参考帧(首帧)自己的 pose** ("保持参考图不动"),不是零 motion
            #       (test_ar_model.py:191 neg_motion_hidden_states = motion_encoder(ref_pose_image))
            # 曾只做 (2)(3) 而漏掉 (1),等于只有运动引导没有外观引导,与渲染端语义不一致。
            mot_neg = motion[:, 0:1].expand_as(mot_win).contiguous()
            M.r_teacher.clear()
            _, x0_unc = M.forward_net_flow(M.teacher, z, sigma, torch.zeros_like(clip_emb), mot_neg)
            M.r_teacher.update(M.writer, dtype=M.dt)      # 立刻恢复 bank,勿影响后续 cond 前向
            _w = guidance_scale
            if keep is not None:                                  # 高噪样本置 0 → 该样本退化为 s_c
                _w = guidance_scale * keep.to(x0_real.dtype).view(-1, 1, 1, 1, 1)
            x0_real = x0_real + _w * (x0_real - x0_unc)
            log_cfg_frac = float(keep.float().mean()) if keep is not None else 1.0
    # 3) DMD loss(整窗)
    loss = dmd_generator_loss(x0_win, x0_fake, x0_real)
    log = {"x0_std": x0.detach().float().std().item()}
    if cfg_sigma_cut < 1.0:
        log["cfg_frac"] = locals().get("log_cfg_frac", 0.0)   # 本 step 有多少比例样本开了 CFG
    # 3b) ★ ODE 回归锚(DMD v1 的 L_reg,arXiv:2311.18828):把学生对**同一初始噪声**的输出
    #     拴在 teacher ODE 轨迹终点上,锚住大尺度结构、限制生成器漂移。
    #     DMD2 为省数据集去掉了它 → 不稳定 → 用 TTUR+GAN 补;我们既无 reg 也无 GAN,
    #     正落在文献说不稳定的配置里。而 ode_init 已经做过这个回归,只是当成前置阶段、
    #     DMD 一开始就把锚松掉了 —— 官方是**全程并行**的一项。
    #     ⚠️ 与已否决的"MSE 到 GT"不是一回事:target 是 teacher 的 ODE 解,不是真实视频。
    #     ⚠️ 官方用 LPIPS(像素空间);我们用 latent MSE 近似 —— 带梯度过 VAE 太贵
    #        (每帧 72ms × 24帧 + 反传)。锚住大尺度结构的作用共通,保真度有偏差。
    if reg_weight > 0 and reg_target is not None:
        l_reg = F.mse_loss(x0_win.float(), reg_target[:, :, k:k + Wn].float())
        loss = loss + reg_weight * l_reg
        log["reg"] = l_reg.detach().item()
    # 4) GAN-G(DMD2 少步锐化):同一 flow 加噪,logit 经 critic 主干回传 generator
    if gan_g_weight > 0:
        zg, sg, _, _ = _add_flow_noise(x0_win, sig_high=sigma_high, shift=score_shift)       # 不 detach → 梯度回 generator
        _, _, logit_fake = M.forward_net_flow(M.critic, zg, sg, clip_emb, mot_win, capture_disc=True)
        if relativistic and real_latent is not None:
            with torch.no_grad():
                real_win = real_latent[:, :, k:k + Wn]
                nr = torch.randn_like(real_win)
                zr = flow_add_noise(real_win, nr, sg).to(M.dt)
                _, _, logit_real = M.forward_net_flow(M.critic, zr, sg, clip_emb, mot_win, capture_disc=True)
            g_gan = gan_g_loss_rel(logit_fake, logit_real)
        else:
            g_gan = gan_g_loss(logit_fake)
        loss = loss + gan_g_weight * g_gan
        log["gan_g"] = g_gan.detach().item()
    return loss, log


def critic_loss(M, noise, clip_emb, motion, sigma_list, block_size=8, grad_window=DMD_WINDOW,
                real_latent=None, gan_d_weight=0.0, relativistic=False, r1r2_weight=0.0, r1r2_sigma=0.05,
                score_shift=SCORE_SHIFT, sigma_high=SIGMA_HIGH):
    # 1) rollout 出 student 当前样本(no_grad)
    with torch.no_grad():
        x0, _ = flow_rollout(M, noise, clip_emb, motion, sigma_list,
                             block_size=block_size, grad_window=None)
        F_ = x0.shape[2]
        k = int(torch.randint(0, F_ - grad_window + 1, (1,)).item())
        x0_win = x0[:, :, k:k + grad_window]
        mot_win = motion[:, k:k + grad_window]
        z, sigma, n, v_tgt = _add_flow_noise(x0_win, sig_high=sigma_high, shift=score_shift)
    # 2) critic 预 v,v-MSE 去噪 loss;GAN 时同一前向抓 fake logit
    #    reentrant grad-ckpt 需至少一个输入 requires_grad → 让 z 需梯度(其 grad 弃用)
    z = z.detach().requires_grad_(True)
    do_gan = gan_d_weight > 0 and real_latent is not None
    if do_gan:
        v_pred, _, logit_fake = M.forward_net_flow(M.critic, z, sigma, clip_emb, mot_win, capture_disc=True)
    else:
        v_pred, _ = M.forward_net_flow(M.critic, z, sigma, clip_emb, mot_win)
    loss = critic_eps_loss(v_pred, v_tgt)            # 复用 MSE(此处是 v-MSE)
    log = {"critic_x0_std": x0.float().std().item()}
    # 3) GAN-D(DMD2):real 用同一 σ 加噪、同一随机窗
    if do_gan:
        real_win = real_latent[:, :, k:k + grad_window]
        with torch.no_grad():
            zr = flow_add_noise(real_win, torch.randn_like(real_win), sigma).to(M.dt)
        _, _, logit_real = M.forward_net_flow(M.critic, zr, sigma, clip_emb, mot_win, capture_disc=True)
        d_gan = gan_d_loss_rel(logit_real, logit_fake) if relativistic else gan_d_loss(logit_real, logit_fake)
        loss = loss + gan_d_weight * d_gan
        log.update(gan_d=d_gan.detach().item(),
                   d_real=logit_real.detach().float().mean().item(),
                   d_fake=logit_fake.detach().float().mean().item())
        if r1r2_weight > 0:
            zr_p = (zr + r1r2_sigma * torch.randn_like(zr)).to(M.dt)
            z_p = (z + r1r2_sigma * torch.randn_like(z)).to(M.dt)
            _, _, logit_real_p = M.forward_net_flow(M.critic, zr_p, sigma, clip_emb, mot_win, capture_disc=True)
            _, _, logit_fake_p = M.forward_net_flow(M.critic, z_p, sigma, clip_emb, mot_win, capture_disc=True)
            reg = gan_reg_r1r2(logit_real, logit_real_p, logit_fake, logit_fake_p)
            loss = loss + r1r2_weight * reg
            log["gan_reg"] = reg.detach().item()
    return loss, log
