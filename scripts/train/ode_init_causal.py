"""ODE init 第 2 步:因果学生对 teacher ODE 轨迹做回归(对齐 Self-Forcing 官方
model/ode_regression.py:generator_loss + CausVid Sec 4.3)。

官方逻辑(逐行对应)
------------------
  target_latent = ode_latent[:, -1]                      # 轨迹终点 = teacher 的解,**不是 GT**
  index = _get_timestep(..., uniform_timestep=False)     # 每个 block 独立抽噪声档,块内同档
  noisy_input = gather(ode_latent, index)                # 取该档的轨迹点
  _, pred = generator(noisy_input, cond, timestep)       # 因果 mask 下**一次**前向
  loss = mse(pred[timestep!=0], target[timestep!=0])

我们的对应
----------
* 噪声档 = gen_ode_pairs.py 存的 student_sigmas(随 --shift 变;shift1=[1,.75,.5,.25], shift3=[1,.9,.75,.5]),轨迹 idx [0,1,2,3],
  target = idx 4(σ=0)。
* 模型出 v,按 x0 = z − σ·v 转成 x0 再回归(与官方在 x0 空间比对一致)。
* **逐帧 timestep**:块间 σ 不同必须靠它;unet_3d.py/resnet.py 已改造并通过等价性测试
  (scripts/val/test_perframe_timestep_parity.py)。
* 只训 temporal_modules,image backbone 冻结 —— 因果化改的是时间维注意力,
  放开空间主干会重蹈 train_scope=all 的空间/时间协同适配陷阱。
* 不再做 CFG:引导已经烘焙进轨迹里了(生成时 cfg=2.5),学生学的是**已引导**的映射。

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/train/ode_init_causal.py --smoke
  torchrun --nproc_per_node=2 scripts/train/ode_init_causal.py --max_steps 4000 --batch 1 --accum 8
"""
import os, sys, glob, time, argparse, random
import torch
import torch.distributed as dist
import torch.nn.functional as F
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(_REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
sys.path.append(third_party("motar"))
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.temporal_causal import TemporalCausalControl

DEC_CFG = XP("REPO", "configs/test_ar_model.yaml")


class ODEPairs(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.pt")))
        assert self.files, f"{root} 里没有 ODE 轨迹"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu")
        # ★ 必须 detach:gen_ode_pairs 早期版本在 no_grad 外算 clip_emb,requires_grad=True
        #   被一起存进了文件;带梯度的张量会让 DataLoader 的共享内存 collate(out=)直接报错。
        return {"traj": d["traj"].detach(), "clip_emb": d["clip_emb"][0].detach(),
                "ref_latent": d["ref_latent"][0].detach(), "motion": d["motion"][0].detach(),
                "sigmas": torch.tensor(d["student_sigmas"])}


# ★ 非重入梯度检查点:reentrant checkpoint + DDP(find_unused_parameters=True) 是脆弱组合,
#   计算图一被扰动(例如训练循环里插入验证前向)就报「参数被标记 ready 两次」。
#   flow_stage1_image.py / flow_stage2_temporal.py 早就打了这个补丁,这里当初漏了。
import torch.utils.checkpoint as _ckptmod
_orig_ckpt = _ckptmod.checkpoint
def _ckpt_nonreentrant(fn, *a, use_reentrant=None, **k):
    return _orig_ckpt(fn, *a, use_reentrant=False, **k)
_ckptmod.checkpoint = _ckpt_nonreentrant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="output/ode_pairs")
    ap.add_argument("--init", default="output/flow_stage2_cfgdrop/CUM1500.pt")
    ap.add_argument("--out", default="output/ode_init_causal")
    ap.add_argument("--block", type=int, default=8, help="num_frame_per_block")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    # ★ EMA:见 src/utils/ema.py —— 无 EMA 时权重在小 batch 下游走,单点指标不可信
    ap.add_argument("--ema_decay", type=float, default=0.999, help="0 = 关闭 EMA")
    ap.add_argument("--ema_device", default="auto", choices=["auto", "cpu"])
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--val_every", type=int, default=0,
                    help="每 N 步在固定验证子集上算 x0-MSE(便宜)。0=关闭。见 CLAUDE.md §6.4")
    ap.add_argument("--val_sample_every", type=int, default=0,
                    help="每 N 步额外做因果 renoise rollout 采样,算 PSNR/SSIM/LPIPS/FID 并存图。0=不采样")
    ap.add_argument("--val_clips", type=int, default=8)

    ap.add_argument("--keep_last", type=int, default=3, help="只保留最近 N 个 ckpt(每个 3.4GB)")
    ap.add_argument("--refresh_every", type=int, default=100,
                    help="每 N 步重扫轨迹目录(边生成边训时吃到新数据;0=关闭)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        lrk = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lrk)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        lrk, rank, world = 0, 0, 1
    is_main = rank == 0
    dev = torch.device(f"cuda:{lrk}"); dt = torch.bfloat16
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(DEC_CFG); ic = OmegaConf.load(cfg.inference_config)
    refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path,
                                                subfolder="unet").to(dev, dt).eval()
    refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"),
                                    map_location="cpu"), strict=True)
    refu.requires_grad_(False)
    denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt)
    denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
    denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
    _ik = torch.load(args.init, map_location="cpu")
    denu.load_state_dict(_ik["denoising_unet"], strict=False)
    assert bool(_ik.get("rope", False)), "init ckpt 必须是 RoPE 版(因果与双向共用同一位置编码)"

    # ★ 因果化:mode="train" = 并行 block-causal mask(块内双向、块间因果)
    ctrl = TemporalCausalControl(denu, block_size=args.block, window=0)
    ctrl.set_rope(True)
    ctrl.set_mode("train")

    denu.requires_grad_(False)
    for nm, p in denu.named_parameters():
        if "temporal_modules" in nm:
            p.requires_grad_(True)
    try: denu.enable_gradient_checkpointing()
    except Exception as e: print("[warn] grad ckpt:", e)
    train_params = [p for p in denu.parameters() if p.requires_grad]
    if is_main:
        print(f"[ode-init] init={args.init} block={args.block} 因果attn={len(ctrl)}层  "
              f"可训 temporal={sum(p.numel() for p in train_params)/1e6:.1f}M", flush=True)

    rwriter = ReferenceAttentionControl(refu, do_classifier_free_guidance=False, mode="write",
                                        batch_size=args.batch, fusion_blocks="full")
    rreader = ReferenceAttentionControl(denu, do_classifier_free_guidance=False, mode="read",
                                        batch_size=args.batch, fusion_blocks="full")

    def build_dl():
        ds = ODEPairs(args.pairs)
        sampler = None
        if world > 1:
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        return DataLoader(ds, batch_size=args.batch, sampler=sampler, shuffle=(sampler is None),
                          num_workers=4, pin_memory=True, drop_last=True), len(ds)

    dl, nds = build_dl()
    if is_main:
        print(f"[data] {nds} 条 ODE 轨迹", flush=True)

    if distributed:
        denu = torch.nn.parallel.DistributedDataParallel(denu, device_ids=[lrk], output_device=lrk,
                broadcast_buffers=False, find_unused_parameters=True, gradient_as_bucket_view=True)
        denu_core = denu.module
    else:
        denu_core = denu
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    from src.utils.ema import EMA, ema_weights
    ema = EMA(denu_core, args.ema_decay, args.ema_device) if args.ema_decay > 0 else None
    print(f"[ema] {'on decay=%.4f' % args.ema_decay if ema else 'off'}", flush=True)

    step = 0; it = iter(dl); loss_acc = 0.0; t_last = time.time(); checked = False
    max_steps = 3 if args.smoke else args.max_steps
    # ---- 训练内验证(CLAUDE.md §6.4) + TensorBoard
    _tb = None; _val = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        _tb = SummaryWriter(os.path.join(args.out, "logs", "tb"))
    if args.val_every > 0:
        from src.utils.inproc_val import Validator
        # ★ student_sigmas 必须在构造时就从轨迹文件读:step-0 基线验证在训练循环**之前**跑,
        #   那时还没有 batch。gen_ode_pairs 按 --shift 派生并写进每个轨迹文件,不能写死。
        import glob as _g
        _ss = torch.load(sorted(_g.glob(f"{args.pairs}/*.pt"))[0],
                         map_location="cpu")["student_sigmas"]
        if is_main: print(f"[val] student_sigmas ← 轨迹文件: {_ss}", flush=True)
        _val = Validator(out_dir=args.out, kind="causal", n_clips=args.val_clips,
                         student_sigmas=list(_ss), block=args.block, ctrl=ctrl,
                         is_main=is_main, tb=_tb, clip_len=args.block * 8)

    # ★ 训练前先跑一次 step 0 基线,否则看不到第一段的跃升
    if args.val_every > 0 and is_main:
        with ema_weights(ema, denu_core):
            _val.run(0, denu_core, refu, None, dev, dt, rr=rreader, rw=rwriter,
                     do_sample=args.val_sample_every > 0)

    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        micro = 0.0
        for _ in range(args.accum):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            traj = b["traj"].to(dev, dt)                       # [B,5,4,F,H,W]
            B, _, C, F_, H, W = traj.shape
            target = traj[:, -1]                              # [B,4,F,H,W] σ=0 的 teacher 解
            sig_list = b["sigmas"][0].to(dev)                 # [4] = [1.0,.75,.5,.25]
            motion = b["motion"].to(dev, dt)
            nblk = (F_ + args.block - 1) // args.block
            # ★ 官方 uniform_timestep=False:每 block 独立抽档,块内同档
            idx_blk = torch.randint(0, len(sig_list), (B, nblk), device=dev)
            idx = idx_blk.repeat_interleave(args.block, dim=1)[:, :F_]        # [B,F]
            z = torch.gather(traj, 1, idx.view(B, 1, 1, F_, 1, 1)
                             .expand(B, 1, C, F_, H, W)).squeeze(1)          # [B,4,F,H,W]
            sig = sig_list[idx]                                              # [B,F]
            with torch.no_grad():
                rwriter.clear()
                refu(b["ref_latent"].to(dev, dt), torch.zeros((), device=dev).long(),
                     encoder_hidden_states=b["clip_emb"].to(dev, dt), return_dict=False)
                rreader.update(rwriter, dtype=dt)
            t_emb = (sig * 1000.0).to(dt)                                    # [B,F] 逐帧
            v_pred = denu(z, t_emb, encoder_hidden_states=[b["clip_emb"].to(dev, dt), motion],
                          pose_cond_fea=None, return_dict=False)[0]
            x0_pred = z.float() - sig.view(B, 1, F_, 1, 1) * v_pred.float()   # x0 = z − σ·v
            loss = F.mse_loss(x0_pred, target.float())
            if not checked and is_main:
                print(f"[check] F={F_} nblk={nblk} 档分布={torch.bincount(idx_blk.flatten(), minlength=4).tolist()} "
                      f"z.std={z.float().std():.3f} target.std={target.float().std():.3f} "
                      f"x0_pred.std={x0_pred.std():.3f}", flush=True); checked = True
            (loss / args.accum).backward(); micro += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if ema is not None:
            ema.update(denu_core)
        step += 1; loss_acc += micro
        # ★ 边生成边训:周期性重扫目录,吸收新落盘的轨迹(ODEPairs 只在构造时 glob)
        if args.refresh_every and step % args.refresh_every == 0:
            dl, n_new = build_dl(); it = iter(dl)
            if is_main and n_new != nds:
                print(f"[data] 轨迹 {nds} → {n_new} 条", flush=True)
            nds = n_new
        if step % args.log_every == 0 and is_main:
            dts = (time.time() - t_last) / args.log_every; t_last = time.time()
            print(f"step {step:6d}  ode_x0_mse={loss_acc/args.log_every:.5f}  {dts:.2f}s/it  "
                  f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            if _tb is not None: _tb.add_scalar("train/ode_x0_mse", loss_acc/args.log_every, step)
            loss_acc = 0.0
        _do_s = args.val_sample_every > 0 and step % args.val_sample_every == 0
        # ★ 采样档独立判定:val_sample_every 不必是 val_every 的倍数
        #   (曾因嵌套判定导致 --val_sample_every 250 只在 500/1000/... 触发)
        if args.val_every > 0 and (step % args.val_every == 0 or _do_s):
            if is_main:
                with ema_weights(ema, denu_core):
                    _val.run(step, denu_core, refu, None, dev, dt, rr=rreader, rw=rwriter,
                             do_sample=_do_s)
            if dist.is_initialized(): dist.barrier()
        if (step % args.save_every == 0 or step == max_steps) and is_main:
            torch.save({**({"ema": ema.state_dict()} if ema is not None else {}), "denoising_unet": denu_core.state_dict(), "step": step, "args": vars(args),
                        "objective": "ode_regression", "stage": "causal_init", "rope": True,
                        "block": args.block, "student_sigmas": sig_list.tolist()},
                       f"{args.out}/odeinit_step_{step}.pt")
            print(f"[save] {args.out}/odeinit_step_{step}.pt", flush=True)
            # ★ 每个 ckpt 3.4GB,盘只剩 ~180G,必须限量(flow_stage2_temporal.py 有,这里当初漏了)
            if args.keep_last > 0:
                cks = sorted(glob.glob(f"{args.out}/odeinit_step_*.pt"),
                             key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
                for old in cks[:-args.keep_last]:
                    os.remove(old); print(f"[prune] {old}", flush=True)
    if is_main:
        print("DONE", flush=True)


if __name__ == "__main__":
    main()
