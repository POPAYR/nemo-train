"""
Flow-matching 微调 teacher（长序列 64 帧 + rectified-flow 目标）
================================================================
目标：把双向 X-Nemo teacher 从 ε-pred 微调成 rectified-flow 速度预测，并在 L=64 上训（扩时序 PE 32→64），
得到「拉直 ODE、少步稳定」的 flow-teacher，供后续蒸馏因果学生。

rectified flow:  z_t=(1-t)·x0 + t·ε,  t∈[0,1];  目标速度 v=ε-x0;  loss=‖v_θ(z_t, t)-v‖²。
时间嵌入：连续 t → 传 t*999（UNet Timesteps 正弦嵌入吃连续值）。
双向 teacher：不加 TemporalCausalControl（保持 native 双向时序注意力）。因果化留给后续蒸馏。
数据/参考设置复用 ode_init_decoder.py。

用法:  CUDA_VISIBLE_DEVICES=N python scripts/train/flow_teacher_ft.py --smoke
       CUDA_VISIBLE_DEVICES=N python scripts/train/flow_teacher_ft.py --L 64 --batch 1 --accum 8 &
"""
import os, sys, argparse, glob, time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
sys.path.append("/media/ps/ssd5/ayr/motar")
# ★ 强制非 reentrant grad-checkpoint（DDP 兼容；UNet3D 用 torch.utils.checkpoint.checkpoint 默认 reentrant=True，
#   与 DDP 冲突 "mark ready only once"）。模块限定调用→patch 该函数即可。
import torch.utils.checkpoint as _ckptmod
_real_ckpt = _ckptmod.checkpoint
def _ckpt_nonreentrant(fn, *a, use_reentrant=None, **k):
    return _real_ckpt(fn, *a, use_reentrant=False, **k)
_ckptmod.checkpoint = _ckpt_nonreentrant
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from diffusers.video_processor import VideoProcessor
from transformers import CLIPVisionModelWithProjection
from data.dataset import MotarDataset
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
DEC_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml"


def build(dev, dt, fresh_temporal=False):
    cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
    refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
    denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
    if not fresh_temporal:
        denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    # fresh_temporal=True 时保持构造时的初始化:proj_out 为 zero_module → temporal block 恒等,
    # 起点即「纯逐帧空间模型」,空间/motion cross-attn 全部保留。
    imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
    return refu, denu, imgenc


def build_loader(args, dev, dt, rank=0, world=1):
    dcfg = OmegaConf.load(TRAIN_CFG).data
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
    def make(src):
        return MotarDataset(
            pose_dir=src.pose_dir, audio_dir=src.audio_dir, caption_dir=src.caption_dir,
            data_name_path=src.data_name_path, tokenizer_path=dcfg.tokenizer_path,
            data_stats_path=dcfg.data_stats_path,
            context_length=args.L, fps=dcfg.fps, sr=dcfg.sr,
            text_max_len=dcfg.get("text_max_len", 128),
            random_crop=True, pad_short=True, load_video=True,
            latent_dir=src.latent_dir, video_dir=src.video_dir, video_processor=vproc)
    ds = [make(s) for s in dcfg.sources]
    train_ds = ConcatDataset(ds) if len(ds) > 1 else ds[0]
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    if rank == 0:
        print(f"[data] {len(dcfg.sources)} sources → {len(train_ds)} clips  L={args.L}  world={world}")
    dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                    num_workers=6, pin_memory=True, drop_last=True, prefetch_factor=3, persistent_workers=True)
    stats = torch.load(dcfg.data_stats_path, map_location="cpu")
    mean = stats["mean"].reshape(-1).to(dev, dt); std = stats["std"].reshape(-1).to(dev, dt)
    return dl, mean, std, sampler


def sample_t(B, dev, mode="lognorm"):
    """t∈(0,1)。lognorm=logit-normal(SD3,集中中段);uniform=U(0,1)。"""
    # ★ 2026-09-13 起**写死 uniform**,lognorm 分支已删除。
    # 原因见 docs/decisions/2026-09-12_sigma-distribution.md:
    # lognorm(=sigmoid(randn)) 配 shift=3 时 σ<0.25 只占 **1.4%** 训练样本,
    # 细节精修段严重欠训 → 帧间跳变;而 uniform 配同样的 shift=3 得到
    # σ<.25/.25-.5/.5-.75/.75-.9/.9-1 = 10.0/15.0/25.0/25.0/25.1%,
    # **σ≥0.75 仍是 50%**(4 步学生的需求不受损),是纯增益。
    # 曾经的注释把 uniform 的占比写成了 lognorm 的,导致这个问题长期未被发现。
    assert mode == "uniform", f"σ 分布已写死 uniform,收到 {mode!r}"
    return torch.rand(B, device=dev).clamp(1e-3, 1 - 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-5)
    # ★ EMA:见 src/utils/ema.py —— 无 EMA 时权重在小 batch 下游走,单点指标不可信
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = 关闭 EMA")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--L", type=int, default=64, help="★长序列微调(扩时序PE)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1, help="L=64 显存重,micro-batch=1")
    ap.add_argument("--train_scope", choices=["temporal", "temporal_cross", "all"], default="temporal",
                    help="ε→v 是全局输出语义变化;temporal 收不动就换 all")
    ap.add_argument("--t_mode", choices=["uniform"], default="uniform",
                    help="σ 分布已写死 uniform;传 lognorm 会被 argparse 直接拒绝")
    ap.add_argument("--from_ckpt", default=None, help="从已有 flow ckpt 续训")
    ap.add_argument("--out", default="/media/ps/ssd5/ayr/x-nemo-inference/output/flow_teacher")
    ap.add_argument("--fresh_temporal", action="store_true",
                    help="★temporal 模块从零开始训(不加载 temporal_module_path,也不从 --from_ckpt 载 temporal)。"
                         "proj_out 是 zero_module 初始化 → 起点=纯逐帧空间模型(画质好但帧间抖),"
                         "用 RoPE 从零学平滑,避免『先忘掉加性PE』的包袱。对应 XNeMo 原 stage2 的做法。")
    ap.add_argument("--rope", action="store_true",
                    help="★用 RoPE 替代加性正弦绝对PE(相对位置,train==stream严格一致,任意长度可外推)。"
                         "改架构,temporal 模块需重新适配;验证见 scripts/val/test_rope_parity.py")
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--keep_last", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.accum, args.batch, args.L = 30, 10000, 5, 1, 1, 32

    # ---- DDP 初始化（torchrun 设 LOCAL_RANK；单卡则退化）----
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"]); rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(local_rank); dev = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, rank, world = 0, 0, 1; dev = torch.device("cuda:0")
    is_main = (rank == 0)
    dt = torch.bfloat16
    if is_main: os.makedirs(args.out, exist_ok=True)

    refu, denu, imgenc = build(dev, dt, fresh_temporal=args.fresh_temporal)
    if args.rope:
        from src.models.temporal_causal import set_temporal_rope
        _n = set_temporal_rope(denu, True, mode="bidir")
        if is_main: print(f"[rope] 已启用 RoPE(替代加性绝对PE),temporal_self attn 层数={_n},mode=bidir", flush=True)
    if args.from_ckpt:
        _sd = torch.load(args.from_ckpt, map_location="cpu")["denoising_unet"]
        if args.fresh_temporal:      # 只取空间/cross-attn,temporal 保持 fresh(zero proj_out)
            _sd = {k: v for k, v in _sd.items() if "temporal_modules" not in k}
        denu.load_state_dict(_sd, strict=False)
        if is_main:
            print(f"[resume] {args.from_ckpt}" + ("  (已过滤 temporal_modules → fresh)" if args.fresh_temporal else ""))

    denu.requires_grad_(False)
    if args.train_scope == "all":
        denu.requires_grad_(True)
    elif args.train_scope == "temporal_cross":   # temporal + cross-attn(attn2, 条件于clip身份+motion)
        for nm, p in denu.named_parameters():
            if "temporal_modules" in nm or "attn2" in nm:
                p.requires_grad_(True)
    else:  # temporal
        for nm, p in denu.named_parameters():
            if "temporal_modules" in nm:
                p.requires_grad_(True)
    try: denu.enable_gradient_checkpointing()
    except Exception as e: print("[warn] grad ckpt:", e)
    denu.train()

    # 双向 teacher：reference 读写在 raw denu 上设好（hooks 在子模块，DDP 包裹后仍生效）
    rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write", batch_size=args.batch, fusion_blocks="full")
    rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read", batch_size=args.batch, fusion_blocks="full")

    raw = denu
    if ddp:
        # 非reentrant ckpt + find_unused + broadcast_buffers=False(reference bank 存为 buffer,禁止 DDP 同步它,
        # 否则触发 "parameter used outside forward")
        denu = DDP(denu, device_ids=[local_rank], gradient_as_bucket_view=True,
                   find_unused_parameters=True, broadcast_buffers=False)
    train_params = [p for p in raw.parameters() if p.requires_grad]
    if is_main:
        eff = args.batch * args.accum * world
        print(f"[scope] {args.train_scope}  trainable={sum(p.numel() for p in train_params)/1e6:.1f}M  "
              f"双向teacher无causal  world={world} batch={args.batch} accum={args.accum} → 有效bsz={eff}")

    dl, m_mean, m_std, sampler = build_loader(args, dev, dt, rank, world)
    def denorm(x): return x * (m_std + 1e-6) + m_mean
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    from src.utils.ema import EMA, ema_weights
    ema = EMA(raw, args.ema_decay, args.ema_device) if args.ema_decay > 0 else None
    print(f"[ema] {'on decay=%.4f' % args.ema_decay if ema else 'off'}", flush=True)

    step = 0; t_last = time.time(); loss_acc = 0.0; checked = False; epoch = 0
    if sampler is not None: sampler.set_epoch(epoch)
    data_iter = iter(dl)
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        for a in range(args.accum):
            try: b = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None: sampler.set_epoch(epoch)
                data_iter = iter(dl); b = next(data_iter)
            x0 = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]
            B, T = x0.shape[0], x0.shape[2]
            motion = denorm(b["motion_tensor"].to(dev, dt)).reshape(B, T, 32, 16)
            ref_latent = b["ref_latent"].to(dev, dt)
            mask = b.get("mask", None)
            if mask is not None:
                mask = mask.to(dev, dt).view(B, 1, T, 1, 1)

            with torch.no_grad():
                clip_emb = imgenc(b["ref_img"].to(dev, dt)).image_embeds.unsqueeze(1)
                rwriter.clear()
                refu(ref_latent, torch.zeros((), device=dev).long(), encoder_hidden_states=clip_emb, return_dict=False)
                rreader.update(rwriter, dtype=dt)

            # ---- rectified flow（约定同 diffusers FlowMatchEulerDiscreteScheduler, shift=1）----
            sig = sample_t(B, dev, args.t_mode)                    # [B] sigma∈(0,1)
            sb = sig.view(B, 1, 1, 1, 1)
            noise = torch.randn_like(x0)
            z_t = ((1 - sb) * x0.float() + sb * noise.float()).to(dt)   # scale_noise
            v_target = (noise.float() - x0.float())                     # 速度 ε-x0
            t_emb = (sig * 1000.0).to(dt)                               # timestep=sigma*1000
            # static_graph 不兼容 no_sync：每次 backward 都 allreduce(冗余但正确),static_graph 支持梯度累积
            v_pred = denu(z_t, t_emb, encoder_hidden_states=[clip_emb, motion], pose_cond_fea=None, return_dict=False)[0]
            se = (v_pred.float() - v_target) ** 2
            loss = (se * mask).sum() / mask.expand_as(se).sum().clamp_min(1) if mask is not None else se.mean()
            if not checked and is_main:
                print(f"[check] x0.std={x0.float().std():.3f} v_target.std={v_target.std():.3f} "
                      f"v_pred.std={v_pred.float().std():.3f} sigma=[{sig.min():.3f},{sig.max():.3f}]", flush=True)
                checked = True
            (loss / args.accum).backward()
            micro += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if ema is not None:
            ema.update(raw)
        step += 1; loss_acc += micro
        if step % args.log_every == 0 and is_main:
            dt_s = (time.time() - t_last) / args.log_every; t_last = time.time()
            print(f"step {step:6d}  flow_mse={loss_acc/args.log_every:.5f}  {dt_s:.2f}s/it  "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            loss_acc = 0.0
        elif step % args.log_every == 0:
            loss_acc = 0.0
        if (step % args.save_every == 0 or step == args.max_steps) and is_main:
            torch.save({**({"ema": ema.state_dict()} if ema is not None else {}), "denoising_unet": raw.state_dict(), "step": step, "args": vars(args),
                        "objective": "rectified_flow_v", "rope": args.rope},
                       f"{args.out}/flow_step_{step}.pt")
            print(f"[save] {args.out}/flow_step_{step}.pt", flush=True)
            cks = sorted(glob.glob(f"{args.out}/flow_step_*.pt"),
                         key=lambda p: int(os.path.basename(p)[len('flow_step_'):-3]))
            for old in cks[:-args.keep_last]:
                try: os.remove(old)
                except OSError: pass
    if is_main: print("[done]")
    if ddp: dist.destroy_process_group()


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
