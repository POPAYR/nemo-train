"""ODE init 第 1 步:用**双向 teacher** 离线生成 ODE 轨迹对(对齐 Self-Forcing 官方
scripts/generate_ode_pairs.py + CausVid Sec 4.3)。

官方做法:teacher 跑完整 ODE(48 步),整条轨迹只保留 5 个点 `[0,12,24,36,-1]`,
对应 denoising_step_list=[1000,750,500,250] 四个噪声档 + 最终干净解。
学生随后做**回归**:target = 轨迹终点(teacher 的解,**不是 GT**),输入 = 某个噪声档的轨迹点。

我们的对应关系
--------------
学生 4 步的 σ 网格取 **[1.0, 0.75, 0.5, 0.25]**(= 官方 denoising_step_list/1000,shift=1.0)。
⚠️ 不能用 diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps(4)` —— 它给的是
[1.0, 0.667, 0.334, 0.001],末档与终点重合,第 4 步空转,"4 步"实为 3 步。
ODE 用均匀网格 linspace(1,0,N+1),N=12 时索引 [0,3,6,9] 恰为学生的 4 个 σ,索引 12 为 target。
这就是"ode init 少步"的落点 —— 12 步而非 35 步,生成成本降到 34%。

CFG 用 XNeMo 三重置空(bank 不注入 + CLIP 置零 + 参考帧 motion),与
scripts/val/render_cfg_compare.py --cfg_mode full 及 src/distill/flow_step.py 完全一致。
⚠️ 必须 2× batch 且 uncond 在**前半**(uc_mask 约定);单 batch 开 do_cfg 会误切帧维。

用法(6 卡分片):
  for i in 0 1 2 3 4 5; do CUDA_VISIBLE_DEVICES=$i python scripts/train/gen_ode_pairs.py \
      --shard $i --nshards 6 --clips 200 --out output/ode_pairs & done
"""
import os, sys, argparse, time
import torch
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(_REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
sys.path.append(third_party("motar"))
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers.video_processor import VideoProcessor
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection
from data.dataset import MotarDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import set_temporal_rope

DEC_CFG = XP("REPO", "configs/test_ar_model.yaml")
TRAIN_CFG = XP("REPO", "configs/train_ar.yaml")
# ★ σ 网格不用 diffusers 默认:它是 linspace(1.0, 1/1000, N) 再补 0,4 步时末档 σ=0.001
#   与终点重合 → 第 4 步空转,"4 步"实为 3 步。官方 denoising_step_list=[1000,750,500,250]
#   即 σ=[1.0,0.75,0.5,0.25],最后一步 0.25→0 是实打实的。这里用均匀 linspace(1,0,N+1),
#   N=12 时索引 [0,3,6,9] 恰为 σ=[1.0,0.75,0.5,0.25],索引 12 为 σ=0(target)。
KEEP_IDX = [0, 3, 6, 9, 12]

def _sigma_grid(steps, shift):
    """σ 网格。shift=1 → 均匀 linspace(1,0,N+1);shift>1 → SD3/Flux 重参数化
    σ = s·t/(1+(s-1)t),把步子往高噪段挤(结构在高噪段成形)。

    ★ 关键性质:shift 是对**同一个 t 网格**逐点映射,所以 KEEP_IDX 不用动 ——
      idx [0,3,6,9] 永远对应 t=[1,.75,.5,.25],只是映射后的 σ 变了:
        shift=1 → [1.0, 0.75, 0.5,  0.25]
        shift=3 → [1.0, 0.90, 0.75, 0.50]
      学生的 σ 档随之改变,故 STUDENT_SIGMAS 必须由本函数派生,不能写死。"""
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift * t / (1.0 + (shift - 1.0) * t) if shift != 1.0 else t

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="output/flow_stage2_cfgdrop/CUM1500.pt")
ap.add_argument("--out", default="output/ode_pairs")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--clips", type=int, default=200, help="本分片生成多少条")
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--cfg", type=float, default=2.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data_list", default=None,
                help="覆盖数据清单(ode_init 不用 pad mask,应传 ≥64 帧清单)")
ap.add_argument("--sources", type=int, nargs="+", default=None,
                help="只用 train_ar.yaml 的第几个数据源(0=hallo3, 1=MEAD);缺省=全部")
ap.add_argument("--resume", type=int, default=1, help="1=跳过本分片已生成的条数,断点续跑")
ap.add_argument("--use_ema", action="store_true", help="用 teacher ckpt 的 ema 键")
ap.add_argument("--shift", type=float, default=1.0,
                help="SD3/Flux σ 重参数化 σ=s·t/(1+(s-1)t)。1.0=均匀(现状)。"
                     "★ 改这个会同时改学生的 4 个 σ 档(见下方 STUDENT_SIGMAS 的计算)")
a = ap.parse_args()
assert a.steps % 4 == 0, "步数须被 4 整除,否则 σ 网格不含学生的 4 步网格"
SIGMAS = _sigma_grid(a.steps, a.shift)                      # [steps+1]
STUDENT_SIGMAS = [round(float(SIGMAS[i]), 6) for i in KEEP_IDX[:4]]   # 学生 4 步各次前向的 σ
print(f"[grid] steps={a.steps} shift={a.shift}  student_sigmas={STUDENT_SIGMAS}", flush=True)
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda:0"); dt = torch.bfloat16
torch.manual_seed(a.seed + a.shard * 9973)

cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                map_location="cpu"), strict=True)
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
_tk = torch.load(a.teacher, map_location="cpu")
# ★ --use_ema:teacher 的最佳权重在 ema 键里(瞬时权重在 eff_bsz=4 下会游走,见 src/utils/ema.py)
_tsd = dict(_tk["denoising_unet"])
if a.use_ema:
    assert "ema" in _tk, f"{a.teacher} 没有 ema 键"
    _tsd.update(_tk["ema"]); print(f"[teacher] 使用 EMA 权重({len(_tk['ema'])} 个张量)", flush=True)
denu.load_state_dict(_tsd, strict=False); denu.eval()
if bool(_tk.get("rope", False)):
    set_temporal_rope(denu, True, mode="bidir")            # teacher 是双向的
print(f"[teacher] {a.teacher} step={_tk.get('step')} rope={_tk.get('rope')}", flush=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()

# CFG:writer/reader 都开 do_cfg,denoising 走 2× batch,uncond 在前半
rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=True, mode="write",
                                    batch_size=1, fusion_blocks="full")
rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=True, mode="read",
                                    batch_size=1, fusion_blocks="full")

dcfg = OmegaConf.load(TRAIN_CFG).data
vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
_srcs = list(dcfg.sources) if a.sources is None else [dcfg.sources[i] for i in a.sources]
# ★ --data_list:ODE 轨迹喂给 ode_init,而 ode_init 的 MSE **不用 pad mask**,
#   <64 帧样本经 pad_short 末帧 repeat 会变成静止帧当真值 → 传 ≥64 帧清单
if a.data_list:
    _srcs = [OmegaConf.merge(s, {'data_name_path': a.data_list}) for s in _srcs]
print(f"[src] 使用数据源 {a.sources if a.sources is not None else '全部'}: "
      f"{[str(s.pose_dir).split('/')[-2] for s in _srcs]}", flush=True)
def _pose_dir(src):
    """★ 与 flow_stage2_temporal.py 一致:按 source 推导 pose_embed_real。
    数据重做后 hallo3/MEAD 都只产出 pose_embed_real(常量 bbox 那版已不再生成),
    直接用 cfg 里的 pose_embed 会 100% 取样失败(症状:连续 20 次 Missing data)。"""
    _pd = str(src.pose_dir)
    _pr = _pd.replace("pose_embed", "pose_embed_real")
    if os.path.isdir(_pr):
        return _pr
    assert os.path.isdir(_pd), f"pose 目录都不存在: {_pd} / {_pr}"
    return _pd

ds = ConcatDataset([MotarDataset(
        pose_dir=_pose_dir(s), audio_dir=s.audio_dir, caption_dir=s.caption_dir,
        data_name_path=s.data_name_path, tokenizer_path=dcfg.tokenizer_path,
        data_stats_path=dcfg.data_stats_path, context_length=a.L, fps=dcfg.fps, sr=dcfg.sr,
        text_max_len=dcfg.get("text_max_len", 128), random_crop=True, pad_short=False,
        load_video=True, latent_dir=s.latent_dir, video_dir=s.video_dir, video_processor=vproc)
    for s in _srcs])
g_dl = torch.Generator(); g_dl.manual_seed(a.seed + a.shard * 9973)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
                drop_last=True, generator=g_dl)
stats = torch.load(dcfg.data_stats_path, map_location="cpu")
m_mean = stats["mean"].reshape(-1).to(dev, dt); m_std = stats["std"].reshape(-1).to(dev, dt)
print(f"[data] {len(ds)} clips;本分片目标 {a.clips} 条 → {a.out}", flush=True)


@torch.no_grad()
def ode_traj(clip_emb, motion, ref_latent, F_):
    """N 步均匀 Euler flow ODE(CFG 三重置空),返回保留点 [5,4,F,H,W]。"""
    clip_cat = torch.cat([torch.zeros_like(clip_emb), clip_emb], 0)     # uncond 在前半
    mo_neg = motion[:, 0:1].expand_as(motion).contiguous()
    mo_cat = torch.cat([mo_neg, motion], 0)
    rwriter.clear()
    # reference UNet 跑两遍:uncond 用零 CLIP → 产生「无外观提示」的 bank;
    # 且 do_cfg=True 让前半 batch 在 attn1 里完全不注入 bank(纯自注意力)
    refu(ref_latent.repeat(2, 1, 1, 1), torch.zeros((), device=dev).long(),
         encoder_hidden_states=clip_cat, return_dict=False)
    rreader.update(rwriter, dtype=dt)
    # σ 网格(均匀或 shift)+ 手写 Euler(z_{i+1} = z_i + (σ_{i+1}-σ_i)·v),
    # 不依赖 diffusers 的 sigma_min 怪癖
    sigmas = SIGMAS.to(dev)
    z = torch.randn(1, 4, F_, 64, 64, device=dev, dtype=dt)
    traj = [z.clone()]
    for i in range(a.steps):
        s_cur, s_nxt = sigmas[i], sigmas[i + 1]
        t_emb = (s_cur * 1000.0).expand(2).to(dt)
        v2 = denu(torch.cat([z, z], 0), t_emb,
                  encoder_hidden_states=[clip_cat, mo_cat],
                  pose_cond_fea=None, return_dict=False)[0]
        vu, vc = v2.float().chunk(2)
        v = vu + a.cfg * (vc - vu)
        z = (z.float() + (s_nxt - s_cur) * v).to(dt)
        traj.append(z.clone())
    assert len(traj) == a.steps + 1
    return torch.cat([traj[i] for i in KEEP_IDX], 0)                    # [5,4,F,H,W]


# ★ --resume:断点续生成。文件名是 s{shard}_{n}.pt,n 是分片内计数器,
#   dataloader 的取样顺序由 seed+shard 固定 ⇒ 已有 k 个文件就等于前 k 个样本已完成,
#   跳过它们、让 n 从 k 继续即可,不会重复也不会漏。
_done = 0
if a.resume:
    import glob as _glob
    _done = len(_glob.glob(f"{a.out}/s{a.shard:02d}_*.pt"))
    if _done:
        print(f"[resume] 分片 {a.shard} 已有 {_done} 条,跳过后从第 {_done} 条继续", flush=True)

n = _done; _skipped = 0; t0 = time.time()
for b in dl:
    if n >= a.clips:
        break
    if _skipped < _done:          # 跳过已完成的样本(只走 dataloader,不做推理)
        _skipped += 1
        continue
    F_ = b["video_tensor"].shape[1]
    if F_ != a.L:
        continue
    motion = (b["motion_tensor"].to(dev, dt) * (m_std + 1e-6) + m_mean).reshape(1, F_, 32, 16)
    ref_latent = b["ref_latent"].to(dev, dt)
    with torch.no_grad():                       # 否则 clip_emb 带 requires_grad,存进文件会毒化 collate
        clip_emb = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
    traj = ode_traj(clip_emb, motion, ref_latent, F_)
    torch.save({"traj": traj.to(torch.bfloat16).cpu(),          # [5,4,F,64,64] 前4=噪声档,末=target
                "clip_emb": clip_emb.cpu(), "ref_latent": ref_latent.cpu(),
                "motion": motion.cpu(), "mask": b.get("mask", torch.ones(1, F_)).cpu(),
                "keep_idx": KEEP_IDX, "steps": a.steps, "cfg": a.cfg,
                "student_sigmas": STUDENT_SIGMAS, "shift": a.shift},
               f"{a.out}/s{a.shard:02d}_{n:05d}.pt")
    n += 1
    if n % 10 == 0:
        el = time.time() - t0
        print(f"[shard {a.shard}] {n}/{a.clips}  {el/n:.1f}s/clip  剩余 {(a.clips-n)*el/n/60:.0f}min",
              flush=True)
print(f"[shard {a.shard}] DONE {n} clips in {(time.time()-t0)/60:.1f}min", flush=True)
