"""
flow-matching(rectified-flow)空间的 DMD 蒸馏核心数学
=====================================================
teacher 已从 ε-pred 换成 rectified-flow v-prediction(objective=rectified_flow_v):
  前向加噪:  z_σ = (1-σ)·x0 + σ·ε           σ∈[0,1],σ=1 纯噪声、σ=0 干净数据
  预测目标:  v = ε - x0                       (rectified flow 直线速度场)
  反演 x0:   x0 = z_σ - σ·v                    (由 z_σ 与 v 两式消 ε)
  反演 ε:    ε  = z_σ + (1-σ)·v
DMD 在 x0 空间做分布匹配(见 dmd_loss.dmd_kl_grad),与参数化无关 —— 只要 teacher/critic
都用同一套 flow 加噪 + v→x0,DMD grad=(x0_fake−x0_real)/norm 依旧成立。

时间嵌入约定(对齐 flow_teacher_ft.py / flow_render.py):
  网络输入的 timestep t_emb = σ·(num_train_timesteps-1) ≈ σ·999,σ 越大越接近纯噪声。
  少步采样用 diffusers FlowMatchEulerDiscreteScheduler(shift=1.0) 的 σ 网格。
"""
import torch
import torch.nn.functional as F


def flow_add_noise(x0, noise, sigma):
    """flow 前向:z_σ = (1-σ)·x0 + σ·ε。sigma: [B] 或标量,broadcast 到 x0 形状。"""
    s = _bcast(sigma, x0)
    return (1.0 - s) * x0 + s * noise


def v_to_x0(z, v, sigma):
    """由带噪 z_σ 与速度 v 反演干净 x0 = z_σ − σ·v。fp32 内算,结果回 z.dtype。"""
    s = _bcast(sigma, z).float()
    x0 = z.float() - s * v.float()
    return x0.to(z.dtype)


def x0_to_v(z, x0, sigma):
    """逆向:v = (z_σ − x0)/σ。仅调试/单测用(σ→0 不稳,勿用于训练)。"""
    s = _bcast(sigma, z).float().clamp_min(1e-6)
    v = (z.float() - x0.float()) / s
    return v.to(z.dtype)


def v_target(x0, noise):
    """rectified-flow 训练目标 v = ε − x0(与 σ 无关)。"""
    return noise - x0


def sigma_to_t(sigma, num_train_timesteps=1000):
    """σ∈[0,1] → 网络 timestep 输入 σ·T(对齐 flow_teacher_ft:t_emb=σ·1000,σ=1→t=1000)。"""
    return sigma * num_train_timesteps


def _bcast(sigma, x):
    """把 sigma(标量/[B]/[B,1..])广播成能和 x[B,...] 逐元素运算的形状。"""
    if not torch.is_tensor(sigma):
        return torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
    s = sigma.to(x.device)
    while s.dim() < x.dim():
        s = s.view(list(s.shape) + [1])
    return s.to(x.dtype)


def make_flow_scheduler(num_train_timesteps=1000, shift=1.0):
    """返回一个 FlowMatchEulerDiscreteScheduler(与 flow_render.py 同配置),
    用于取少步 σ 网格 + Euler step。少步 rollout 与推理共用同一 scheduler 保证一致。"""
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=num_train_timesteps, shift=shift)


def flow_step_list(num_steps, num_train_timesteps=1000, shift=1.0, device="cpu"):
    """少步采样的 (timesteps, sigmas):
      timesteps: [N] 网络输入的 t(高→低,≈999→0),喂 UNet。
      sigmas:    [N+1] 对应 σ 网格(高→低,末尾 0),sigmas[i]→sigmas[i+1] 是第 i 步。
    与 FlowMatchEulerDiscreteScheduler.set_timesteps(N) 完全一致。"""
    sched = make_flow_scheduler(num_train_timesteps, shift)
    sched.set_timesteps(num_steps, device=device)
    timesteps = sched.timesteps.to(device)                       # [N]
    sigmas = sched.sigmas.to(device)                             # [N+1](含末尾 0)
    return timesteps, sigmas


# --------------------------- 自测(dummy,验证 flow 数学往返)---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, Fr, H, W = 2, 4, 3, 8, 8
    x0 = torch.randn(B, C, Fr, H, W)
    eps = torch.randn_like(x0)
    sigma = torch.rand(B).clamp(0.02, 0.98)

    # ① v↔x0 往返
    z = flow_add_noise(x0, eps, sigma)
    v = v_target(x0, eps)
    x0_rec = v_to_x0(z, v, sigma)
    v_rec = x0_to_v(z, x0, sigma)
    print(f"[①] v→x0 误差={F.mse_loss(x0_rec, x0):.2e}  x0→v 误差={F.mse_loss(v_rec, v):.2e}  (应~0)")

    # ② z_σ 与 v 的线性关系:ε = z + (1-σ)v
    s = _bcast(sigma, z)
    eps_rec = z + (1 - s) * v
    print(f"[②] ε=z+(1-σ)v 误差={F.mse_loss(eps_rec, eps):.2e}  (应~0)")

    # ③ 少步 σ 网格
    ts, sg = flow_step_list(4)
    print(f"[③] N=4  timesteps={ts.tolist()}  sigmas={[round(x,3) for x in sg.tolist()]}")
    assert sg[0] > 0.99 and abs(sg[-1]) < 1e-6, "σ 应从~1 到 0"
    print("flow 数学自测完成 ✅")
