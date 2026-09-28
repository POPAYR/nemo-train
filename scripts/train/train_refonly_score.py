"""
训 ref-only 视频 score model —— 当 DMD-through-G 的 s_real（方案②，见 motar/AR_MOTION_WORK_SUMMARY.md §10.4）。
=====================================================================================
问题：s_real=teacher|m̂ 对任何 m̂ 渲的 x̂ 都认为合理 → DMD 零信号(l_dmd≈0)。
解法：s_real 换成"真实说话视频给定身份的分布"——在真实视频 latent 上、**motion 喂 null**、
     ε-MSE 训一个 ref-only score。denoise 的 target 是真实动态视频 → 学到"真实脸是动态的"先验。
     塌 motion 渲的静态 x̂ 会被它判低密度 → DMD 顶 std。
复用 DMD2Models：它的 critic(可训、从 teacher init、双向)正好当这个 ref-only score 训。
输入(每步)：真实视频 latent [B,C,24,H,W] + ref_latent + ref_img(→clip)；motion=零。
loss：ε-MSE。存 critic 当 s_real ckpt。
"""
import os, sys, time, argparse, datetime
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from src.distill.models import DMD2Models
from data.dataset import MotarDataset

TRAIN_CFG = "/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml"
VAE_SCALE = 0.18215


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="refonly_score")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = 关闭 EMA")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--max_steps", type=int, default=15000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.save_every, args.log_every, args.batch = 8, 6, 1, 1

    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        lrk = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lrk); dev = torch.device(f"cuda:{lrk}")
    else:
        rank, world, dev = 0, 1, torch.device("cuda:0")
    is_main = (rank == 0); dt = torch.bfloat16

    outdir = f"/media/ps/ssd5/ayr/motar/refonly_score_output/{args.exp}"
    ckptdir = os.path.join(outdir, "ckpt")
    if is_main: os.makedirs(ckptdir, exist_ok=True)

    # DMD2Models：用它的 critic(可训、从 teacher init) 当 ref-only score；teacher/generator 用不到
    M = DMD2Models(dev, dt=dt, gen_ckpt=None, block_size=8)
    critic = M.critic                      # 可训双向 UNet = ref-only score
    critic.train().requires_grad_(True)

    # motion_encoder：算 X-Nemo 推理用的 neg_motion(中性 motion)当 null（比 zeros 精确、in-distribution）
    from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
    cfg_full = OmegaConf.load(TRAIN_CFG)
    motion_encoder = MotEncoder().to(dev, dt).eval().requires_grad_(False)
    motion_encoder.load_state_dict(torch.load(cfg_full.motion_encoder_path, map_location="cpu"), strict=True)
    params = [p for p in critic.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    from src.utils.ema import EMA
    ema = EMA(critic, args.ema_decay, args.ema_device) if args.ema_decay > 0 else None
    print(f"[ema] {'on decay=%.4f' % args.ema_decay if ema else 'off'}", flush=True)
    if is_main:
        print(f"[refonly] critic trainable={sum(p.numel() for p in params)/1e6:.0f}M  "
              f"frames={args.frames} batch={args.batch} lr={args.lr}")

    # 数据（load_video=True 给 video_tensor + ref_latent + ref_img；pad_short 容短样本）
    dcfg = OmegaConf.load(TRAIN_CFG).data
    from diffusers.video_processor import VideoProcessor
    vproc = VideoProcessor(do_resize=True, vae_scale_factor=8)
    def make(s):
        return MotarDataset(pose_dir=s.pose_dir, audio_dir=s.audio_dir, caption_dir=s.caption_dir,
                            data_name_path=s.data_name_path, tokenizer_path=dcfg.tokenizer_path,
                            data_stats_path=dcfg.data_stats_path, context_length=args.frames,
                            fps=dcfg.fps, sr=dcfg.sr, text_max_len=128, random_crop=True,
                            pad_short=True, load_video=True, latent_dir=s.latent_dir,
                            video_dir=s.video_dir, video_processor=vproc)
    ds = [make(s) for s in dcfg.sources]
    dataset = ConcatDataset(ds) if len(ds) > 1 else ds[0]
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    dl = DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                    num_workers=6, pin_memory=True, drop_last=True, prefetch_factor=3, persistent_workers=True)

    def allreduce(ps):
        if world <= 1: return
        for p in ps:
            if p.grad is not None:
                dist.all_reduce(p.grad); p.grad /= world

    it = iter(dl); step = 0; t0 = time.time(); acc = 0.0
    def nb():
        nonlocal it
        try: return next(it)
        except StopIteration: it = iter(dl); return next(it)

    if is_main: print(f"[train] start → {args.max_steps}")
    while step < args.max_steps:
        b = nb()
        vlat = b["video_tensor"].to(dev, dt).permute(0, 2, 1, 3, 4).contiguous() * VAE_SCALE  # [B,C,F,H,W]
        ref_lat = b["ref_latent"].to(dev, dt) * VAE_SCALE
        clip = M.clip_embed(b["ref_img"].to(dev, dt))
        B, C, F_, H, W = vlat.shape
        M.set_reference(ref_lat, clip, B)
        # neg_motion = motion_encoder(参考帧, bbox[:2]=0) = X-Nemo 推理的中性 motion（比 zeros 精确）
        rmc = b["ref_mot_cond"].to(dev, dt)                           # [B,3,224,224]
        bbox = torch.ones((B, 3), device=dev, dtype=dt); bbox[:, :2] = 0
        with torch.no_grad():
            neg = motion_encoder(rmc, bbox)                          # [B,32,16]
        null_mot = neg.unsqueeze(1).expand(B, F_, *neg.shape[1:]).to(dt)   # [B,F,32,16] 所有帧同一中性 motion
        t = torch.randint(20, 980, (B,), device=dev)
        noise = torch.randn_like(vlat)
        x_t = M.scheduler.add_noise(vlat, noise, t).to(dt)
        eps_pred, _ = M.forward_net(critic, x_t, t, clip, null_mot)
        loss = ((eps_pred.float() - noise.float()) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce(params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if ema is not None:
            ema.update(critic)
        step += 1; acc += loss.item()

        if step % args.log_every == 0 and is_main:
            dts = (time.time() - t0) / args.log_every; t0 = time.time()
            print(f"[{datetime.datetime.now():%H:%M:%S}] step {step}/{args.max_steps} "
                  f"eps_mse={acc/args.log_every:.4f} {dts:.1f}s/it mem={torch.cuda.max_memory_allocated()/1e9:.0f}GB")
            acc = 0.0
        if (step % args.save_every == 0 or step == args.max_steps) and is_main:
            path = os.path.join(ckptdir, f"refonly_step_{step}.pt")
            torch.save({**({"critic_ema": ema.state_dict()} if ema is not None else {}), "critic": critic.state_dict(), "step": step, "args": vars(args)}, path)
            print(f"[save] step {step} → {path}")


if __name__ == "__main__":
    main()
