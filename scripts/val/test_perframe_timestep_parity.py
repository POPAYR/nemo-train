"""逐帧 timestep 改造的等价性测试。

改了 unet_3d.py(构造 [B*F] 的 time embedding)和 resnet.py(把 temb 广播成 [B,D,F,1,1])。
必须保证:**常数 σ 下逐帧路径与原标量路径逐位等价**,否则 teacher/渲染行为会被悄悄改变。
再验证:不同帧给不同 σ 时输出确实随之改变(说明逐帧真的生效,不是被广播吃掉)。

用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/test_perframe_timestep_parity.py
"""
import sys
import torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from src.models.unet_3d import UNet3DConditionModel
from src.models.temporal_causal import set_temporal_rope

DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"
CKPT = "/media/ps/ssd5/ayr/x-nemo-inference/output/flow_stage2_cfgdrop/CUM1500.pt"
import argparse
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_ap = argparse.ArgumentParser(); _ap.add_argument("--fp32", action="store_true")
_a = _ap.parse_args()
dev = torch.device("cuda:0"); dt = torch.float32 if _a.fp32 else torch.bfloat16
print(f"[dtype] {dt}")
cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)

denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
_k = torch.load(CKPT, map_location="cpu")
denu.load_state_dict(_k["denoising_unet"], strict=False); denu.eval()
if bool(_k.get("rope", False)):
    set_temporal_rope(denu, True, mode="bidir")

B, F_ = 1, 16
torch.manual_seed(0)
z = torch.randn(B, 4, F_, 64, 64, device=dev, dtype=dt)
clip = torch.randn(B, 1, 768, device=dev, dtype=dt)
mo = torch.randn(B, F_, 32, 16, device=dev, dtype=dt)


@torch.no_grad()
def run(t):
    return denu(z, t, encoder_hidden_states=[clip, mo], pose_cond_fea=None,
                return_dict=False)[0].float()


T = 500.0
v_scalar = run(torch.tensor([T], device=dev, dtype=dt).expand(B))
v_frames = run(torch.full((B, F_), T, device=dev, dtype=dt))
d = (v_scalar - v_frames).abs()
rel = (d.mean() / v_scalar.std()).item()
print(f"[常数σ 等价性] max|Δ|={d.max():.3e}  mean|Δ|={d.mean():.3e}  "
      f"相对={rel:.2e}  (v.std={v_scalar.std():.4f})")
# 噪声地板:同一路径跑两次的差(捕捉 kernel 非确定性)
floor = (run(torch.tensor([T], device=dev, dtype=dt).expand(B)) - v_scalar).abs().mean().item()
print(f"  噪声地板(同路径重跑)mean|Δ|={floor:.3e}")
tol = 1e-5 if dt == torch.float32 else 5e-2   # fp32 GEMM 重排序地板 ~2e-6
print("  → " + ("✅ 等价(差异在数值精度内)" if rel < tol else "❌ 不等价,改造有问题"))

# 逐帧真的生效吗:前半帧 σ=1.0,后半帧 σ=0.25。
# ⚠️ teacher 是**双向** temporal attention,改后半帧 σ 会经注意力传到前半帧,
#    故前半帧 Δ≠0 是物理正确的;判据只能看"后半帧变化远大于前半帧"。
t_mix = torch.full((B, F_), 1000.0, device=dev, dtype=dt)
t_mix[:, F_ // 2:] = 250.0
v_mix = run(t_mix)
v_all_hi = run(torch.tensor([1000.0], device=dev, dtype=dt).expand(B))
front = (v_mix[:, :, :F_ // 2] - v_all_hi[:, :, :F_ // 2]).abs().mean()
back = (v_mix[:, :, F_ // 2:] - v_all_hi[:, :, F_ // 2:]).abs().mean()
print(f"[逐帧生效性] 与全σ=1.0相比:前半帧Δ={front:.4f}  后半帧Δ={back:.4f}  比值={back/front:.2f}×")
print("  → " + ("✅ 后半帧变化显著更大,逐帧生效" if back > 2 * front
                else "❌ 逐帧未生效"))
print("DONE")
