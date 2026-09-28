"""
compare_samplers_diagnose.py
============================
对比三种 motion 来源，并验证 "std_ratio=0.5" 的归因。

对比对象：
    (A) DDIM-4   : generate(use_ddim=True,  num_sampling_steps=4)   ← 与训练 4 步 DDIM 同构
    (B) DPM-20   : generate(use_ddim=False, num_sampling_steps=20)  ← 旧推理采样器，多步
    (C) GT       : pose_embed 里的真值 motion

输出三组证据：
    1. 全局指标：每种来源 vs GT 的 std_ratio、cos_sim、MSE
    2. 【判定性】按帧 std 时间曲线：递减 => 自回归收缩；持平 => 全局缩放
    3. 归因消融（只对 DDIM-4 这条做，逐个隔离变量）：
         a. cfg_schedule = linear   vs   constant
         b. cfg 全开      vs   cfg=1(完全无引导)
         c. 100% 自回归   vs   teacher-forced(每帧喂 GT 历史)
       —— 若 teacher-forced 把 std_ratio 拉回 ~1.0，则坐实"自回归收缩 + p_sf 不足"

依赖：直接复用 verify_ar_xnemo_sweep.py 里的加载/特征逻辑。
请确保本脚本与 verify_ar_xnemo_sweep.py 在同一目录（或在 PYTHONPATH 中）。
"""

import argparse
import json
import os
import sys
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 复用 verify 脚本里的现成逻辑（编码器加载、特征提取、AR 加载、样本收集）
# 不重复造轮子，保证与正式推理链路完全一致
# ---------------------------------------------------------------------------
try:
    from scripts.test_ar_model import (
        load_encoders, load_ar_model, extract_features,
        compute_ref_motion_emb, load_motion_gt, collect_samples,
        load_xnemo_pipeline,
    )
except Exception as e:
    print("[FATAL] 无法从 verify_ar_xnemo_sweep 导入，请确认本脚本与其同目录。")
    print("        原始错误:", e)
    raise

from omegaconf import OmegaConf
from PIL import Image


# ===========================================================================
#  统计工具
# ===========================================================================
def per_frame_std(motion: torch.Tensor) -> np.ndarray:
    """每一帧在 feature 维上的 std，再... 不对，我们要的是“某段时间窗内、
    跨帧的动态范围”。这里用滑窗：对每个 t，取局部窗口内跨帧的 std 均值。
    这样能看出 std 随时间是否衰减（自回归收缩的特征）。
    motion: [T, D] -> 返回 [n_seg] 每段一个标量
    """
    raise NotImplementedError  # 占位，真正实现见 segmented_std


def segmented_std(motion: torch.Tensor, seg: int = 30) -> List[Dict]:
    """把序列切成长度 seg 的段，每段算“跨帧 std 的特征维均值”。
    返回 [{start, end, std}]。这是判定自回归收缩的核心量。
    """
    T = motion.shape[0]
    out = []
    for i in range(0, T, seg):
        chunk = motion[i:i + seg].float()           # [seg, D]
        if chunk.shape[0] < 2:
            continue
        s = chunk.std(dim=0).mean().item()           # 跨帧 std，再对 D 求均值
        out.append({"start": i, "end": min(i + seg, T), "std": s})
    return out


def global_metrics(pred: torch.Tensor, gt: torch.Tensor) -> Dict:
    T = min(pred.shape[0], gt.shape[0])
    p, g = pred[:T].float(), gt[:T].float()
    gt_std = g.std(dim=0).mean().item()
    pr_std = p.std(dim=0).mean().item()
    return {
        "frames": T,
        "pred_std": pr_std,
        "gt_std": gt_std,
        "std_ratio": pr_std / (gt_std + 1e-8),
        "cos_sim": F.cosine_similarity(p, g, dim=-1).mean().item(),
        "mse": F.mse_loss(p, g).item(),
    }


def slope_of_std(segs: List[Dict]) -> float:
    """对分段 std 做线性拟合，返回斜率（每段的变化）。
    斜率显著为负 => 自回归收缩；接近 0 => 全局缩放。"""
    if len(segs) < 2:
        return 0.0
    x = np.arange(len(segs), dtype=np.float64)
    y = np.array([s["std"] for s in segs], dtype=np.float64)
    # 归一化到首段，便于跨样本比较
    y = y / (y[0] + 1e-8)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


# ===========================================================================
#  生成封装：一个统一入口，控制所有变量
# ===========================================================================
@torch.no_grad()
def gen_motion(ar_model, feats, device, dtype, *,
               use_ddim: bool, num_steps: int,
               cfg_audio: float, cfg_text: float, cfg_schedule: str,
               seed_frame: Optional[torch.Tensor],
               teacher_forced_gt: Optional[torch.Tensor] = None) -> torch.Tensor:
    """统一生成接口。

    teacher_forced_gt 不为 None 时，走 teacher-forcing 诊断路径：
        每帧不把模型自己的输出当历史，而是喂 GT 历史。
        —— 这是验证“自回归收缩”的关键开关：
           若 TF 下 std_ratio 回到 ~1.0，则收缩来自自回归，而非采样器/denorm。
    """
    T = feats["total_frames"]
    text_emb = feats["text_emb"].to(device, dtype=dtype)
    audio_emb = feats["audio_emb"].to(device, dtype=dtype)
    local_audio = feats["local_audio_emb"].to(device, dtype=dtype)

    if teacher_forced_gt is None:
        # 正常自回归路径：直接用 generate（与正式推理完全一致）
        with torch.cuda.amp.autocast(dtype=dtype):
            out = ar_model.generate(
                seq_len=T, text_emb=text_emb,
                audio_emb=audio_emb, local_audio_feat=local_audio,
                cfg_audio=cfg_audio, cfg_text=cfg_text, cfg_schedule=cfg_schedule,
                use_tqdm=False, first_frame=seed_frame,
                num_sampling_steps=num_steps, use_ddim=use_ddim,
                denorm_output=True,
            )
        return out.squeeze(0)

    # ---- teacher-forced 诊断路径 ----
    # 复刻 generate 的内部循环，但每帧 query 用 GT 历史而非自输出。
    # 注意：这里需要访问 ar_model 的内部组件，若你的 MotionTransformer
    #       字段名不同，请按需改 fusion_net/audio_proj/motion_proj/layers/norm/
    #       normalize/denormalize/diffloss 的命名。
    return _generate_teacher_forced(
        ar_model, feats, device, dtype,
        use_ddim=use_ddim, num_steps=num_steps,
        cfg_audio=cfg_audio, cfg_text=cfg_text, cfg_schedule=cfg_schedule,
        seed_frame=seed_frame, gt_motion=teacher_forced_gt,
    )


@torch.no_grad()
def _generate_teacher_forced(ar_model, feats, device, dtype, *,
                             use_ddim, num_steps, cfg_audio, cfg_text,
                             cfg_schedule, seed_frame, gt_motion):
    """与 generate() 同构，唯一区别：cur = GT[t-1]（归一化后），而非 cur = sample。
    这样隔离掉“误差累积”，单看“单帧生成本身”的方差是否正常。

    若你的 generate 内部结构与此不完全一致，本函数需要同步修改。
    我已尽量贴合你贴出的 generate 实现。
    """
    from model.armodel import SelfAttnKVCache, CrossAttnKVCache  # 按你的实际模块路径

    m = ar_model
    T = feats["total_frames"]
    text_emb = feats["text_emb"].to(device, dtype=dtype)
    audio_emb = feats["audio_emb"].to(device, dtype=dtype)
    local_audio = feats["local_audio_emb"].to(device, dtype=dtype)
    bsz = text_emb.shape[0]

    # GT 归一化到模型内部空间（generate 里 first_frame 也是先 normalize 的）
    gt_norm = m.normalize(gt_motion.to(device=device, dtype=dtype))  # [T, D] -> 同空间
    if gt_norm.dim() == 2:
        gt_norm = gt_norm.unsqueeze(0)  # [1, T, D]

    if num_steps is not None:
        old_steps = m.diffloss.num_sampling_steps
        m.diffloss.num_sampling_steps = int(num_steps)

    try:
        with torch.cuda.amp.autocast(dtype=dtype):
            fusion_latents = m.fusion_net(text_emb, audio_emb)
            zero_fusion = torch.zeros_like(fusion_latents)
            local_audio_feat = m.audio_proj(local_audio)
            use_cfg = (cfg_audio != 1.0) or (cfg_text != 1.0)

            if use_cfg:
                fusion_in = torch.cat([fusion_latents, fusion_latents, zero_fusion], dim=0)
            else:
                fusion_in = fusion_latents

            self_caches = [SelfAttnKVCache() for _ in m.layers]
            fusion_caches = [CrossAttnKVCache() for _ in m.layers]

            # 起始帧
            if seed_frame is None:
                cur = torch.zeros(bsz, 1, m.motion_dim, device=device, dtype=dtype)
            else:
                ff = m.normalize(seed_frame.to(device=device, dtype=dtype))
                if ff.dim() == 2:
                    ff = ff.unsqueeze(1)
                cur = ff[:, -1:].contiguous()

            samples = []
            for t in range(T):
                local_t = local_audio_feat[:, t:t + 1]
                if use_cfg:
                    x_in = torch.cat([cur, cur, cur], dim=0)
                    zero_local = torch.zeros_like(local_t)
                    local_in = torch.cat([local_t, zero_local, zero_local], dim=0)
                else:
                    x_in = cur
                    local_in = local_t

                h = m.motion_proj(x_in)
                for layer, sc, fc in zip(m.layers, self_caches, fusion_caches):
                    h = layer.forward_cached(h, fusion_in, local_in, sc, fc, m.max_len)
                last_feat = m.norm(h[:, -1])

                if cfg_schedule == "linear":
                    prog = (t + 1) / T
                    ca = 1.0 + (cfg_audio - 1.0) * prog
                    ct = 1.0 + (cfg_text - 1.0) * prog
                else:
                    ca, ct = cfg_audio, cfg_text

                if use_ddim:
                    sample = m.diffloss.sample_for_infer(
                        last_feat, bsz=bsz, num_steps=num_steps,
                        cfg_audio=ca, cfg_text=ct, use_cfg=use_cfg)
                else:
                    sample = m.diffloss.sample_dual_cfg(
                        last_feat, bsz=bsz, cfg_audio=ca, cfg_text=ct, use_cfg=use_cfg)

                samples.append(sample.unsqueeze(1))

                # ★ 唯一区别：下一帧 query 用 GT 历史，而非自输出
                if t < T - 1:
                    cur = gt_norm[:, t:t + 1].contiguous()

            out = torch.cat(samples, dim=1)
        return m.denormalize(out).squeeze(0)
    finally:
        if num_steps is not None:
            m.diffloss.num_sampling_steps = old_steps


# ===========================================================================
#  主流程
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ar_ckpt", type=str, required=True)
    ap.add_argument("--test_dir", type=str, required=True)
    ap.add_argument("--ref_image", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="output/compare_samplers")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--seg", type=int, default=30, help="按帧 std 曲线的分段长度")
    ap.add_argument("--ar_cfg_audio", type=float, default=4.0)
    ap.add_argument("--ar_cfg_text", type=float, default=2.0)
    ap.add_argument("--seed_mode", type=str, default="ref", choices=["ref", "zero"])
    ap.add_argument("--ddim_steps", type=int, default=4)
    ap.add_argument("--dpm_steps", type=int, default=20)
    ap.add_argument("--need_pipe_for_seed", action="store_true",
                    help="seed_mode=ref 时需要 X-Nemo 的 motion_encoder 来算参考帧 seed")
    args = ap.parse_args()

    device = torch.device(args.device)
    ar_dtype = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)
    config = OmegaConf.load(args.config)

    print("[Init] 加载编码器 …")
    tokenizer, text_encoder, audio_processor, audio_encoder = load_encoders(config, device)

    # seed=ref 需要 motion_encoder（在 X-Nemo pipeline 里）。为省显存，只在需要时加载。
    pipe = None
    if args.seed_mode == "ref":
        print("[Init] 加载 X-Nemo pipeline（仅为 ref seed 的 motion_encoder）…")
        weight_dtype = torch.float16
        pipe = load_xnemo_pipeline(config, device, weight_dtype)

    samples = collect_samples(args.test_dir, args.num_samples)
    print(f"[Data] {len(samples)} 个样本\n")

    # 预加载两个步数的 AR 模型（DDIM 和 DPM 都用同一权重，仅步数/采样器不同）
    # 注意：num_sampling_steps 在 generate 内部会被临时覆盖，所以这里实例化一次即可。
    print("[Init] 加载 AR 模型 …")
    ar_model = load_ar_model(config, args.ar_ckpt, device, ar_dtype, args.ddim_steps)

    report = []

    for s in samples:
        name = s["name"]
        print("=" * 64)
        print(f"Sample: {name}")
        print("=" * 64)

        ref_path = args.ref_image or s["ref_frame"]
        if ref_path is None:
            print("  [skip] 无参考图"); continue
        W = H = 512
        ref_pil = Image.open(ref_path).convert("RGB").resize((W, H))

        # GT motion
        if not (s["pose_path"] and os.path.exists(s["pose_path"])):
            print("  [skip] 无 GT pose_embed"); continue
        gt_motion = load_motion_gt(s["pose_path"]).to(device, dtype=ar_dtype)

        # 特征（所有路径共享）
        feats = extract_features(s, tokenizer, text_encoder,
                                 audio_processor, audio_encoder, device)

        # ref seed（所有路径共享，保证可比）
        seed_frame = None
        if args.seed_mode == "ref" and pipe is not None:
            seed_frame = compute_ref_motion_emb(pipe, ref_pil, device, ar_dtype)

        # ---- 对齐 GT 帧数 ----
        T = min(feats["total_frames"], gt_motion.shape[0])
        gt_trim = gt_motion[:T]

        entry = {"name": name, "frames": int(T), "runs": {}}

        # =================================================================
        # PART 1 + 2：三种来源的全局指标 + 按帧 std 曲线
        # =================================================================
        configs = {
            "DDIM4_linear_cfgON": dict(
                use_ddim=True, num_steps=args.ddim_steps,
                cfg_audio=args.ar_cfg_audio, cfg_text=args.ar_cfg_text,
                cfg_schedule="linear", teacher_forced_gt=None),
            "DPM20_linear_cfgON": dict(
                use_ddim=False, num_steps=args.dpm_steps,
                cfg_audio=args.ar_cfg_audio, cfg_text=args.ar_cfg_text,
                cfg_schedule="linear", teacher_forced_gt=None),
        }

        # =================================================================
        # PART 3：归因消融（只在 DDIM-4 上做，逐个隔离变量）
        # =================================================================
        # a. cfg_schedule: linear -> constant（验证 schedule 是否压早期帧）
        configs["DDIM4_CONST_cfgON"] = dict(
            use_ddim=True, num_steps=args.ddim_steps,
            cfg_audio=args.ar_cfg_audio, cfg_text=args.ar_cfg_text,
            cfg_schedule="constant", teacher_forced_gt=None)
        # b. cfg 全关（cfg=1）：隔离 CFG 本身对方差的影响
        configs["DDIM4_linear_cfgOFF"] = dict(
            use_ddim=True, num_steps=args.ddim_steps,
            cfg_audio=1.0, cfg_text=1.0,
            cfg_schedule="linear", teacher_forced_gt=None)
        # c. teacher-forced（喂 GT 历史）：隔离“自回归误差累积”这一项
        #    —— 若此项 std_ratio 回到 ~1.0，则坐实自回归收缩
        configs["DDIM4_CONST_TEACHERFORCED"] = dict(
            use_ddim=True, num_steps=args.ddim_steps,
            cfg_audio=args.ar_cfg_audio, cfg_text=args.ar_cfg_text,
            cfg_schedule="constant", teacher_forced_gt=gt_trim)

        for tag, kw in configs.items():
            print(f"\n  >>> {tag}")
            try:
                motion = gen_motion(
                    ar_model, feats, device, ar_dtype,
                    use_ddim=kw["use_ddim"], num_steps=kw["num_steps"],
                    cfg_audio=kw["cfg_audio"], cfg_text=kw["cfg_text"],
                    cfg_schedule=kw["cfg_schedule"],
                    seed_frame=seed_frame,
                    teacher_forced_gt=kw["teacher_forced_gt"],
                )
            except Exception as e:
                print(f"      [error] 生成失败: {e}")
                entry["runs"][tag] = {"error": str(e)}
                continue

            motion = motion[:T].float().cpu()
            gm = global_metrics(motion, gt_trim.float().cpu())
            segs = segmented_std(motion, args.seg)
            gt_segs = segmented_std(gt_trim.float().cpu(), args.seg)
            slope = slope_of_std(segs)

            entry["runs"][tag] = {
                "global": gm,
                "std_slope_norm": slope,          # 负=收缩，~0=持平
                "seg_std": segs,
                "gt_seg_std": gt_segs,
            }

            # 控制台速览
            print(f"      std_ratio={gm['std_ratio']:.3f}  cos={gm['cos_sim']:.3f}  "
                  f"mse={gm['mse']:.4f}  std_slope={slope:+.4f}")
            # 打印分段 std（首/中/尾），直观看收缩
            if len(segs) >= 3:
                f0, fm, fl = segs[0], segs[len(segs)//2], segs[-1]
                g0 = gt_segs[0]["std"] if gt_segs else 1.0
                print(f"      seg std: head={f0['std']/g0:.3f}  "
                      f"mid={fm['std']/g0:.3f}  tail={fl['std']/g0:.3f}  (相对GT首段)")

        report.append(entry)

    # =================================================================
    #  落盘 + 汇总判断
    # =================================================================
    report_path = os.path.join(args.output_dir, "compare_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] 详细报告 -> {report_path}")

    # ---- 自动归因结论 ----
    print("\n" + "=" * 64)
    print("[归因判断]（跨样本平均）")
    print("=" * 64)

    def avg(tag, field_path):
        vals = []
        for e in report:
            r = e["runs"].get(tag, {})
            cur = r
            ok = True
            for k in field_path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False; break
            if ok and isinstance(cur, (int, float)):
                vals.append(cur)
        return float(np.mean(vals)) if vals else None

    rows = [
        ("DDIM4_linear_cfgON",        "DDIM-4 (正式推理设定)"),
        ("DPM20_linear_cfgON",        "DPM-20 (旧多步采样器)"),
        ("DDIM4_CONST_cfgON",         "DDIM-4 + 常数cfg"),
        ("DDIM4_linear_cfgOFF",       "DDIM-4 + 无cfg"),
        ("DDIM4_CONST_TEACHERFORCED", "DDIM-4 + GT历史(TF)"),
    ]
    print(f"  {'配置':<28}{'std_ratio':>10}{'cos':>8}{'std_slope':>11}")
    print("  " + "-" * 56)
    for tag, label in rows:
        sr = avg(tag, ["global", "std_ratio"])
        cs = avg(tag, ["global", "cos_sim"])
        sl = avg(tag, ["std_slope_norm"])
        if sr is None:
            print(f"  {label:<28}{'—':>10}")
            continue
        print(f"  {label:<28}{sr:>10.3f}{cs:>8.3f}{sl:>+11.4f}")

    # ---- 给出文字结论 ----
    sr_ddim = avg("DDIM4_linear_cfgON", ["global", "std_ratio"])
    sr_tf = avg("DDIM4_CONST_TEACHERFORCED", ["global", "std_ratio"])
    sr_const = avg("DDIM4_CONST_cfgON", ["global", "std_ratio"])
    sl_ddim = avg("DDIM4_linear_cfgON", ["std_slope_norm"])

    print("\n  ---- 自动结论 ----")
    if sr_ddim is not None and sr_tf is not None:
        if sr_tf - sr_ddim > 0.25:
            print("  ✓ teacher-forced 把 std_ratio 显著拉回 => 主因是【自回归方差收缩】")
            print("    （训练 p_sf 上限 0.5，测试 100% 自历史，分布错配）")
            print("    建议：提高 curriculum.self_forcing.p_max（治本，需重训一段）")
        else:
            print("  ✗ teacher-forced 未能拉回 std_ratio => 主因【不在自回归】，")
            print("    转查 denorm/stats 或采样器轨迹（看 DDIM vs DPM 差异）")
    if sl_ddim is not None:
        if sl_ddim < -0.02:
            print(f"  ✓ DDIM-4 的 std 随时间递减 (slope={sl_ddim:+.4f}) => 支持自回归收缩")
        else:
            print(f"  ~ DDIM-4 的 std 时间曲线基本持平 (slope={sl_ddim:+.4f}) => 更像全局缩放")
    if sr_const is not None and sr_ddim is not None and sr_const - sr_ddim > 0.1:
        print("  ✓ 常数cfg 比 linear 明显改善 => cfg_schedule=linear 在放大早期收缩")

    print(f"\n[Done] 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()