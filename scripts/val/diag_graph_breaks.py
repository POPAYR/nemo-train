"""
Fullgraph 重构第 0 步：用 torch._dynamo.explain 把去噪 UNet 的 graph break 全量列出
================================================================================
目标：找出到底哪些地方 break（einops? reference-attention bank? assert? diffusers 包装?），
按数量/位置排序，决定先改哪个。改前基线，改后复测 break 数下降。

用法：CUDA_VISIBLE_DEVICES=0 python scripts/val/diag_graph_breaks.py
"""
import os, sys
import torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from omegaconf import OmegaConf
from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder


def main():
    device = torch.device("cuda:0"); dtype = torch.float16
    config = OmegaConf.load(os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    print("[load] building decoder ...")
    ref_unet, den_unet, _ = build_decoder(config, device, dtype)

    H = W = 64; F = 8
    writer = ReferenceAttentionControl(ref_unet, do_classifier_free_guidance=False,
                                       mode="write", batch_size=1, fusion_blocks="full")
    reader = ReferenceAttentionControl(den_unet, do_classifier_free_guidance=False,
                                       mode="read", batch_size=1, fusion_blocks="full")
    clip_emb = torch.randn(1, 1, 768, device=device, dtype=dtype)
    ref_latent = torch.randn(1, 4, H, W, device=device, dtype=dtype)
    t = torch.tensor(500, device=device)
    writer.clear()
    ref_unet(ref_latent, torch.zeros_like(t), encoder_hidden_states=clip_emb, return_dict=False)
    reader.update(writer)

    lat = torch.randn(1, 4, F, H, W, device=device, dtype=dtype)
    mot = torch.randn(1, F, 32, 16, device=device, dtype=dtype)

    print("[explain] 运行 torch._dynamo.explain（可能几分钟）...")
    explanation = torch._dynamo.explain(den_unet)(
        lat, t, encoder_hidden_states=[clip_emb, mot], pose_cond_fea=None, return_dict=False)

    # torch 2.1: explanation 有 .graph_count/.graph_break_count/.break_reasons
    print("\n" + "=" * 70)
    print(f"graph_count       = {getattr(explanation, 'graph_count', '?')}")
    print(f"graph_break_count = {getattr(explanation, 'graph_break_count', '?')}")
    print(f"op_count          = {getattr(explanation, 'op_count', '?')}")
    print("=" * 70)

    reasons = getattr(explanation, "break_reasons", None)
    if reasons:
        # 归并相同原因
        from collections import Counter
        agg = Counter()
        for r in reasons:
            rsn = getattr(r, "reason", str(r))
            # user_stack 取最后一帧（我们的代码）
            stk = getattr(r, "user_stack", None)
            loc = ""
            if stk:
                for fr in reversed(stk):
                    fn = getattr(fr, "filename", "")
                    if "x-nemo-inference/src" in fn:
                        loc = f"{os.path.basename(fn)}:{getattr(fr,'lineno','?')}"; break
                if not loc:
                    fr = stk[-1]; loc = f"{os.path.basename(getattr(fr,'filename','?'))}:{getattr(fr,'lineno','?')}"
            agg[(rsn[:80], loc)] += 1
        print("\n[break 原因 × 位置  (count 降序)]")
        for (rsn, loc), c in agg.most_common(30):
            print(f"  {c:3d}×  {loc:32s}  {rsn}")
    else:
        print("\n[explain] 无 break_reasons 字段，打印原始 explanation：")
        print(str(explanation)[:4000])


if __name__ == "__main__":
    main()
