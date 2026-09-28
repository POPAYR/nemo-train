"""RoPE 正确性 + train/stream 一致性验证(避免 train-infer 错配)。
① RoPE 相对性:rot(q,i)·rot(k,j) 只依赖 (i-j) —— 绝对偏移不改变注意力分数。
② train == stream:整段 block-causal 前向  vs  逐 block KV-cache 流式前向,逐帧输出应数值一致。
   这是官方 Self-Forcing 的核心不变量(训练用 block_mask、推理用 cache,两者必须等价)。
③ 局部窗口下同样成立(window>0)。
用法: CUDA_VISIBLE_DEVICES=0 python scripts/val/test_rope_parity.py
"""
import sys, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from src.models.unet_3d import UNet3DConditionModel
from src.models.motion_module import apply_rope_1d
from src.models.temporal_causal import TemporalCausalControl, collect_temporal_self_attns

dev = torch.device("cuda:0"); dt = torch.float32   # 用 fp32 做数值对拍
torch.manual_seed(0)

# ---------- ① RoPE 相对性 ----------
print("① RoPE 相对性(分数只依赖 i-j)")
B, H, dh = 2, 8, 40
q = torch.randn(B, H, 1, dh, device=dev)
k = torch.randn(B, H, 1, dh, device=dev)
for shift in [0, 7, 100, 1000]:
    i, j = 10 + shift, 4 + shift
    qi = apply_rope_1d(q, torch.tensor([i], device=dev))
    kj = apply_rope_1d(k, torch.tensor([j], device=dev))
    s = (qi * kj).sum(-1).mean().item()
    print(f"   位置对 ({i:>5d},{j:>5d})  相对距离={i-j}  分数={s:+.6f}")
print("   ↑ 四行分数应完全相同(同为相对距离 6)\n")

# ---------- ②③ train vs stream 逐帧对拍 ----------
cfg = OmegaConf.load("/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml")
ic = OmegaConf.load(cfg.inference_config)
unet = UNet3DConditionModel.from_pretrained_2d(
    cfg.pretrained_base_model_path, "", subfolder="unet",
    unet_additional_kwargs=ic.unet_additional_kwargs).to(dev, dt).eval()
attns = collect_temporal_self_attns(unet)
print(f"② train vs stream 逐帧对拍(temporal_self attn 层数={len(attns)})")

BL, F_ = 8, 32
for use_rope in [True, False]:
    for window in [0, 16]:
        # 直接对单层 attention 做对拍(隔离 temporal 注意力本身)
        m = attns[0]
        m._rope = use_rope
        m._causal_block_size = BL
        m._causal_window = window
        x = torch.randn(3, F_, m.to_q.in_features, device=dev, dtype=dt)

        m._causal_mode = "train"; m._frame_offset = 0; m._kv_cache = None
        with torch.no_grad():
            out_train = m._causal_temporal_attn(x, F_, "train")

        m._causal_mode = "stream"; m._kv_cache = None
        outs = []
        with torch.no_grad():
            for cur in range(0, F_, BL):
                m._frame_offset = cur
                m._causal_commit = True
                outs.append(m._causal_temporal_attn(x[:, cur:cur + BL], BL, "stream"))
        out_stream = torch.cat(outs, dim=1)

        err = (out_train - out_stream).abs().max().item()
        rel = err / out_train.abs().max().item()
        tag = "RoPE " if use_rope else "加性PE"
        ok = "✓ 一致" if rel < 1e-4 else "✗ 错配!"
        print(f"   {tag} window={window:<3d} 最大绝对误差={err:.3e}  相对={rel:.3e}  {ok}")

print("\n③ 外推检验:stream 在远超训练长度的 offset 上是否仍与「相对等价的短序列」一致")
m = attns[0]; m._rope = True; m._causal_block_size = BL; m._causal_window = 16
x = torch.randn(3, BL, m.to_q.in_features, device=dev, dtype=dt)
ref = None
for off in [0, 32, 500, 5000]:
    m._causal_mode = "stream"; m._kv_cache = None; m._frame_offset = off
    m._causal_commit = True
    with torch.no_grad():
        o = m._causal_temporal_attn(x, BL, "stream")
    if ref is None:
        ref = o; print(f"   offset={off:<6d} (基准)")
    else:
        e = (o - ref).abs().max().item()
        print(f"   offset={off:<6d} 与基准最大误差={e:.3e}  {'✓ 完全外推' if e < 1e-4 else '✗ 随位置漂移'}")
print("   ↑ RoPE 下「无历史的首块」在任意 offset 应给出相同输出(相对位置全同)")
print("\nDONE")
