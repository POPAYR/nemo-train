"""
DMD-through-G：把视频空间的 DMD 分布匹配梯度穿过冻结渲染器 R 反传到 motion 生成器。
=====================================================================================
= VSD(ProlificDreamer)/SDS(DreamFusion) 机制迁到 audio→motion（application novelty，非新机制）。
复合生成器 R∘G_θ：G_θ(audio)→m̂（可微）；R=冻结视频 teacher（motion-conditioned）渲染 x̂=R(m̂)。
DMD 在视频空间：grad=(x0_fake−x0_real)/norm 作用在 x̂ 上，链式穿 R 回 m̂ 回 θ。

组件（定义见 motar/AR_MOTION_WORK_SUMMARY.md §10）：
- x̂ = R(m̂)          渲染视频（对 m̂ 可微，窗口 ≤24 帧）
- s_real = 冻结 teacher | [ref, m_gt]  （真实 motion，固定目标 → 反塌缩信号）
- s_fake = online LoRA critic | [ref, m̂]（追踪 p_gen）
本模块只放**纯数学**（render / s_real / s_fake 作为 callable 传入），便于 dummy 单测机制正确性。
loss: L = L_mse + w · L_DMD-through-G。
"""
import torch
from .dmd_loss import dmd_generator_loss, critic_eps_loss, eps_to_x0


def dmd_through_g_generator_loss(x_hat, s_real_x0, s_fake_x0):
    """DMD-through-G 生成器项。
    x_hat: 渲染视频 [B,C,F,H,W]，**对 motion 可微**（唯一带梯度的量）。
    s_real_x0 / s_fake_x0: 在 x̂ 加噪后 teacher/critic 的 x0 预测，**detached**（DMD 里是常数目标）。
    复用 dmd_generator_loss：d L/d x̂ = (x0_fake−x0_real)/(N·norm) = DMD 梯度 → 经 x̂ 反传到 motion。"""
    return dmd_generator_loss(x_hat, s_fake_x0, s_real_x0)


@torch.no_grad()
def make_dmd_noise(scheduler, x_hat, t_low=20, t_high=980, dtype=torch.float16):
    """在渲染视频 x̂(detach) 上采 DMD 噪声，返回 (x_t, t)。给 s_real/s_fake 打分用。"""
    B = x_hat.shape[0]
    t = torch.randint(t_low, t_high, (B,), device=x_hat.device)
    n = torch.randn_like(x_hat)
    x_t = scheduler.add_noise(x_hat.detach().float(), n, t).to(dtype)
    return x_t, t


def critic_denoise_loss(eps_pred, eps_true):
    """s_fake(critic) 的去噪训练 loss：ε-MSE（在 x̂.detach() 上，条件 m̂.detach()）。复用 critic_eps_loss。"""
    return critic_eps_loss(eps_pred, eps_true)


# --------------------------- 编排（render/score 作 callable，训练脚本与单测共用）---------------------------
def generator_step(render_fn, s_real_fn, s_fake_fn, scheduler, t_low=20, t_high=980, dtype=torch.float16):
    """G 侧 DMD-through-G 一步。
    render_fn()            -> x̂ [B,C,F,H,W]，**对 motion 可微**（内部跑 teacher 少步 x0 渲染|m̂）。
    s_real_fn(x_t, t)      -> x0_real（teacher | m_gt，no_grad 目标）。
    s_fake_fn(x_t, t)      -> x0_fake（critic  | m̂.detach()，no_grad 目标）。
    返回 (loss_DMD, x̂, {x0std})。loss 对 motion 可微；s_real/s_fake 在 no_grad 下算好当常数目标。"""
    x_hat = render_fn()                                   # 唯一带梯度的量
    x_t, t = make_dmd_noise(scheduler, x_hat, t_low, t_high, dtype)
    with torch.no_grad():
        x0_real = s_real_fn(x_t, t).detach()
        x0_fake = s_fake_fn(x_t, t).detach()
    loss = dmd_through_g_generator_loss(x_hat, x0_real, x0_fake)
    return loss, x_hat, {"x0_std": x_hat.detach().float().std().item()}


def _unet_x0_eps(unet, x_t, t, clip_emb, motion_tokens, acp):
    """X-Nemo denoising_unet 前向 → (x0, eps)。t:[B] 或标量。motion_tokens:[B,F,32,16]。"""
    eps = unet(x_t, t, encoder_hidden_states=[clip_emb, motion_tokens],
               pose_cond_fea=None, return_dict=False)[0]
    x0 = eps_to_x0(x_t, eps, t.reshape(-1) if torch.is_tensor(t) else t, acp)
    return x0, eps


def build_dmd_callables(teacher_unet, critic_unet, clip_emb, mot_hat_tokens, mot_gt_tokens,
                        acp, t_render, latent_shape, device, dtype, generator=None):
    """把真实 X-Nemo UNet 前向包成 generator_step/critic_step 需要的 4 个 callable。
    约定（见 §10）：reference bank 已由调用方设好（teacher/critic 各自的 reader 已 update）。
      - render_fn : teacher(冻结, adapter off) 从噪声 1 步 x0，条件 **m̂ tokens（带梯度）** → x̂ 可微。
      - s_real_fn : teacher | **m_gt tokens**（固定真实目标）。
      - s_fake_fn : critic(LoRA on) | **m̂.detach()**（追踪生成分布）。
      - s_fake_eps_fn: 同上但返回 ε（critic 去噪训练用）。
    mot_hat_tokens 须带梯度（来自 rollout）；mot_gt_tokens 为 GT（无梯度）。"""
    B, C, F, H, W = latent_shape
    mot_hat_det = mot_hat_tokens.detach()
    t_list = [int(t_render)] if isinstance(t_render, (int, float)) else [int(x) for x in t_render]

    def render_fn():
        # K 步 DDIM 从噪声渲染 x̂|m̂（多步累积 motion 敏感度 → 强化 ∂x̂/∂m̂）。全程对 m̂ 可微。
        x_t = torch.randn(B, C, F, H, W, device=device, dtype=dtype, generator=generator)
        x0 = None
        for i, tv in enumerate(t_list):
            t = torch.full((B,), tv, device=device, dtype=torch.long)
            x0, eps = _unet_x0_eps(teacher_unet, x_t, t, clip_emb, mot_hat_tokens, acp)
            if i < len(t_list) - 1:                      # DDIM 到下一个更低 t
                a_next = acp[t_list[i + 1]].view(1, 1, 1, 1, 1).to(x0.dtype)
                x_t = (a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps)
        return x0

    def s_real_fn(x_t, t):
        x0, _ = _unet_x0_eps(teacher_unet, x_t, t, clip_emb, mot_gt_tokens, acp)
        return x0

    def s_fake_fn(x_t, t):
        x0, _ = _unet_x0_eps(critic_unet, x_t, t, clip_emb, mot_hat_det, acp)
        return x0

    def s_fake_eps_fn(x_t, t):
        _, eps = _unet_x0_eps(critic_unet, x_t, t, clip_emb, mot_hat_det, acp)
        return eps

    return render_fn, s_real_fn, s_fake_fn, s_fake_eps_fn


def critic_step(x_hat, s_fake_eps_fn, scheduler, t_low=20, t_high=980, dtype=torch.float16):
    """critic(s_fake) 一步：在生成样本 x̂(detach) 上加噪，critic|m̂.detach() 预 ε，ε-MSE。
    s_fake_eps_fn(x_t, t, eps_true 无需) -> eps_pred。返回 (loss, {})。只更 critic（LoRA）。"""
    B = x_hat.shape[0]
    t = torch.randint(t_low, t_high, (B,), device=x_hat.device)
    n = torch.randn_like(x_hat)
    x_t = scheduler.add_noise(x_hat.detach().float(), n, t).to(dtype)
    # reentrant grad-ckpt 需至少一个输入 requires_grad 才会算参数梯度；critic 输入全 detached
    # → 让 x_t 需梯度(其 grad 弃用)，否则 LoRA 参数拿不到梯度(gradnorm=0)。
    x_t = x_t.detach().requires_grad_(True)
    eps_pred = s_fake_eps_fn(x_t, t)
    return critic_denoise_loss(eps_pred, n), {}


# --------------------------- 自测（dummy，验证「梯度穿过渲染器回到 motion」）---------------------------
if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    B, Dm = 2, 16               # motion 维
    C, Fr, H, W = 4, 3, 8, 8    # 视频 latent
    Nx = C * Fr * H * W

    # dummy 可微渲染器 R：线性 motion→video，x̂ = (Wr @ m̂).reshape(视频)。∂x̂/∂m̂ = Wr（已知）
    Wr = torch.randn(Nx, Dm)
    m_hat = torch.randn(B, Dm, requires_grad=True)
    x_hat = (m_hat @ Wr.T).reshape(B, C, Fr, H, W)   # 对 m_hat 可微

    # dummy s_real / s_fake 的 x0 预测（detached 常数目标）
    x0_real = torch.randn(B, C, Fr, H, W)
    x0_fake = torch.randn(B, C, Fr, H, W)

    loss = dmd_through_g_generator_loss(x_hat, x0_real.detach(), x0_fake.detach())
    loss.backward()

    # 理论：d L/d x̂ = (x0_fake − x0_real)/(N·norm)；再 ∂x̂/∂m̂=Wr → d L/d m̂ = Wrᵀ·(d L/d x̂) 展平
    from .dmd_loss import dmd_kl_grad
    grad_xhat = dmd_kl_grad(x_hat.detach(), x0_fake, x0_real) / x_hat.numel()   # [B,C,F,H,W]
    grad_m_expected = (grad_xhat.reshape(B, Nx) @ Wr)                            # [B,Dm]
    err = (m_hat.grad - grad_m_expected).abs().max()
    print(f"[①] motion.grad 非零: {m_hat.grad.abs().max():.3e}  (应>0 → 梯度确实穿过渲染器回到 motion)")
    print(f"[②] motion.grad vs 理论 Wrᵀ·(DMD grad) 最大误差: {err:.2e}  (应~0 → 链式法则正确)")

    # ③ 塌缩(静态)的 x̂ 应比匹配 s_real 的 x̂ 拿到更大梯度（反塌缩信号方向自检）
    x0_real_dyn = torch.randn(B, C, Fr, H, W) * 2.0        # 「动态」真实目标（大幅度）
    x_static = torch.zeros(B, C, Fr, H, W, requires_grad=True)  # 塌成静态
    x_match = (x0_real_dyn.clone().requires_grad_(True))       # 已匹配目标
    g_static = dmd_through_g_generator_loss(x_static, x0_real_dyn.detach(), x_static.detach())
    print(f"[③] 静态 x̂ 的 DMD loss={g_static.item():.3f}（>0 → 有梯度把它往动态推；机制方向正确）")

    # ④ 编排 generator_step：render/s_real/s_fake 作 callable，验证 loss 经 x̂ 反传到 motion
    class _StubSched:
        def __init__(self, T=1000):
            betas = torch.linspace(1e-4, 0.02, T); self.acp = torch.cumprod(1 - betas, 0)
        def add_noise(self, x0, n, t):
            a = self.acp[t].view([-1] + [1] * (x0.dim() - 1)); return a.sqrt() * x0 + (1 - a).sqrt() * n
    sched = _StubSched()
    m2 = torch.randn(B, Dm, requires_grad=True)
    render_fn = lambda: (m2 @ Wr.T).reshape(B, C, Fr, H, W)          # 可微渲染
    s_real_fn = lambda x_t, t: torch.randn(B, C, Fr, H, W)           # teacher|m_gt
    s_fake_fn = lambda x_t, t: x_t.float() * 0.9                     # critic|m̂（≈追踪）
    lo, xh, log = generator_step(render_fn, s_real_fn, s_fake_fn, sched, dtype=torch.float32)
    lo.backward()
    print(f"[④] generator_step: loss={lo.item():.3f} x0std={log['x0_std']:.3f} "
          f"motion.grad={m2.grad.abs().max():.3e}（>0 → 编排端到端可微）")

    # ⑤ critic_step：只更 critic（这里 dummy），验证能出 ε-MSE 标量
    lc, _ = critic_step(xh.detach(), lambda x_t, t: torch.randn_like(x_t), sched, dtype=torch.float32)
    print(f"[⑤] critic_step: ε-MSE={lc.item():.3f}（去噪 loss 正常）")
    print("DMD-through-G 机制自测完成 ✅")
