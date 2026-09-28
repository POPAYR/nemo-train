"""
Phase 1 验收：因果 temporal 模块的逐元素对拍（DECODER_DISTILL_PLAN.md §Phase 1）
==============================================================================
两个不变量（用整个去噪 UNet 的输出比，含 reference attention + motion + 全分辨率 temporal）:

  P1  causal-train(block=F, 退化为单块=全双向)  ==  原始双向 UNet
      → 验证 causal 代码路径在退化情形不破坏原模型（仅 SDPA vs AttnProcessor 数值差）。

  P2  causal-train(block=K, block-causal mask 整窗前向)  ==  stream(逐 block KV-cache)
      → 验证 KV-cache 流式与掩码训练态逐帧一致（这是 train==推理的核心正确性不变量）。

用 fp32 跑以排除 fp16 噪声、隔离算法正确性。
用法: CUDA_VISIBLE_DEVICES=5 python scripts/val/test_causal_parity.py
"""
import sys, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl

import os
_DT = {"fp32": torch.float32, "fp64": torch.float64, "fp16": torch.float16}[os.environ.get("DTYPE", "fp32")]
dev = torch.device("cuda:0"); dt = _DT
torch.manual_seed(0)
cfg = OmegaConf.load("/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml")
ic = OmegaConf.load(cfg.inference_config)

refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt).eval()
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)

F_, K = 16, 8
clip = torch.randn(1, 1, 768, device=dev, dtype=dt)
refl = torch.randn(1, 4, 64, 64, device=dev, dtype=dt)
t = torch.tensor(500, device=dev)
lat = torch.randn(1, 4, F_, 64, 64, device=dev, dtype=dt)
mot = torch.randn(1, F_, 32, 16, device=dev, dtype=dt)

w = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
r = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
w.clear(); refu(refl, torch.zeros_like(t), encoder_hidden_states=clip, return_dict=False); r.update(w)

ctrl = TemporalCausalControl(denu, block_size=K, window=0)
print(f"[setup] 找到 {len(ctrl)} 个时序自注意力模块  F={F_}  K={K}  dtype={dt}")


@torch.no_grad()
def run_full(lat_in, mot_in):
    return denu(lat_in, t, encoder_hidden_states=[clip, mot_in], pose_cond_fea=None, return_dict=False)[0]


def report(name, a, b):
    diff = (a - b).abs()
    denom = b.abs().mean().clamp_min(1e-8)
    print(f"  {name:38s} max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}  "
          f"rel(mean)={(diff.mean()/denom).item():.3e}")
    return diff.max().item()


with torch.no_grad():
    # 基准：原始双向
    ctrl.disable()
    out_orig = run_full(lat, mot)

    # P1: causal-train, block=F (单块 → 退化为全双向)
    ctrl.set_block_size(F_); ctrl.set_mode("train")
    out_p1 = run_full(lat, mot)

    # P2-train: causal-train, block=K (真 block-causal)
    ctrl.set_block_size(K); ctrl.set_mode("train")
    out_train = run_full(lat, mot)

    # P2-stream: 逐 block KV-cache
    ctrl.set_mode("stream"); ctrl.reset_cache()
    outs = []
    for s in range(0, F_, K):
        ctrl.set_offset(s)
        outs.append(run_full(lat[:, :, s:s+K], mot[:, s:s+K]))
    out_stream = torch.cat(outs, dim=2)
    ctrl.disable()

print("\n[P1] causal-train(block=F) vs 原始双向  (退化为全双向, 应≈0)")
d1 = report("P1", out_p1, out_orig)
print("\n[P2] 整 UNet end-to-end: causal-train(block=K) vs stream(KV-cache)")
print("     (fp32 下经 42 模块+深 UNet 误差放大, 看 rel; 逻辑精确性见下方 fp64 孤立单元测试)")
d2 = report("P2", out_stream, out_train)
rel2 = ((out_stream - out_train).abs().mean() / out_train.abs().mean().clamp_min(1e-8)).item()

print("\n[健全性] block-causal 与双向应当显著不同 (证明因果确实裁掉了未来):")
report("bidir vs block-causal", out_orig, out_train)


def unit_logic_exactness():
    """fp64 孤立单元测试: 单个时序模块 train(mask)==stream(cache) 是否机器精度一致。
    隔离整 UNet 的误差放大, 直接证明 cache/mask/PE 逻辑正确。"""
    from src.models.motion_module import VersatileAttention
    torch.manual_seed(0)
    qd, heads = 320, 8
    m = VersatileAttention(attention_mode="Temporal", cross_attention_dim=None,
                           query_dim=qd, heads=heads, dim_head=qd // heads,
                           temporal_position_encoding=True,
                           temporal_position_encoding_max_len=32).to(dev, torch.float64).eval()
    bd, Fu, Ku = 64, 40, 8   # 40 帧 > PE max_len(32) → 同时验证 PE 自动扩展 + 窗口逐出
    hu = torch.randn(bd, Fu, qd, device=dev, dtype=torch.float64)
    worst = 0.0
    for win in (0, 24):
        m._causal_block_size = Ku; m._causal_window = win
        with torch.no_grad():
            m._causal_mode = "train"; m._frame_offset = 0
            ot = m._causal_temporal_attn(hu, Fu, "train")
            m._causal_mode = "stream"; m._kv_cache = None
            os_ = []
            for s in range(0, Fu, Ku):
                m._frame_offset = s
                os_.append(m._causal_temporal_attn(hu[:, s:s+Ku], Ku, "stream"))
            os_ = torch.cat(os_, dim=1)
        dd = (ot - os_).abs().max().item()
        worst = max(worst, dd)
        print(f"  fp64 单元 window={win:2d}:  max|Δ|={dd:.3e}")
    return worst

print("\n[P3] fp64 孤立单元 (logic exactness):")
d3 = unit_logic_exactness()

ok1 = d1 < 5e-3
ok2 = rel2 < 2e-3          # end-to-end fp32: 相对误差 (放大后) 阈值
ok3 = d3 < 1e-10           # 逻辑精确性: fp64 机器精度
print(f"\n结果: P1 {'PASS' if ok1 else 'FAIL'} (max|Δ|={d1:.2e})   "
      f"P2 {'PASS' if ok2 else 'FAIL'} (rel={rel2:.2e})   "
      f"P3 {'PASS' if ok3 else 'FAIL'} (fp64 max|Δ|={d3:.2e})")
print("Phase 1 因果对拍" + (" 全部 PASS ✅" if (ok1 and ok2 and ok3) else " 有 FAIL ❌"))
