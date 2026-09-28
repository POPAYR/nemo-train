"""
渲染对比：teacher-causal vs ODE-init-causal —— 看 ODE-init 是否修好 cold-start。
两者都用因果流式(block=8)、同 ref/motion/init_noise，唯一差别=temporal 权重(teacher vs ODE-init ckpt)。
ODE-init 保持多步 → 用多步 DDIM 渲染。

用法: CUDA_VISIBLE_DEVICES=N python scripts/val/render_ode_ckpt.py --ckpt output/ode_init/ode_step_2000.pt
"""
import sys, os, argparse, torch
sys.path.append("/media/ps/ssd5/ayr/x-nemo-inference")
from omegaconf import OmegaConf
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKLTemporalDecoder, AutoencoderTiny
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
from src.pipelines.causal_streaming_pipeline import CausalStreamingPipeline
from src.utils.util import save_videos_grid

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, default=None, help="ODE-init ckpt (含 denoising_unet)")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--frames", type=int, default=32)
ap.add_argument("--block", type=int, default=8)
ap.add_argument("--sample", type=str, default="M003_video_down_angry_level_1_001")
ap.add_argument("--decoder", type=str, default="svd", choices=["svd", "taesd"], help="svd=高质量(慢), taesd=快(少细节)")
ap.add_argument("--out", type=str, default="/media/ps/ssd5/ayr/x-nemo-inference/output/ode_render")
args = ap.parse_args()

dev = torch.device("cuda:0"); dt = torch.float16
cfg = OmegaConf.load("/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml")
ic = OmegaConf.load(cfg.inference_config)
vae = AutoencoderKLTemporalDecoder.from_pretrained(cfg.vae_path).to(dev, dt).eval()
refu = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet").to(dev, dt).eval()
denu = UNet3DConditionModel.from_pretrained_2d(cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=ic.unet_additional_kwargs).to(device=dev, dtype=dt).eval()
denu.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)
refu.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
denu.load_state_dict(torch.load(cfg.temporal_module_path, map_location="cpu"), strict=False)
motenc = MotEncoder().to(dev, dt).eval()
motenc.load_state_dict(torch.load(cfg.denoising_unet_path.replace("denoising_unet", "motion_encoder"), map_location="cpu"), strict=True)
imgenc = CLIPVisionModelWithProjection.from_pretrained(cfg.image_encoder_path).to(dev, dt).eval()
sched = DDIMScheduler(**OmegaConf.to_container(ic.noise_scheduler_kwargs))
taesd = AutoencoderTiny.from_pretrained("/media/ps/ssd5/ayr/pretrained/taesd", torch_dtype=dt).to(dev).eval()
pipe = CausalStreamingPipeline(vae=vae, image_encoder=imgenc, reference_unet=refu,
                               denoising_unet=denu, motion_encoder=motenc, scheduler=sched).to(dev, dt)

ROOT = "/media/ps/ssd5/ayr/MEAD_frames_512_25fps"; n = args.sample; T = args.frames
ref_pil = Image.open(f"{ROOT}/face_frames/{n}/000000.jpg").convert("RGB").resize((512, 512))
mot = torch.load(f"{ROOT}/pose_embed/{n}.pt", map_location="cpu").float()[:, :T].reshape(1, T, -1).to(dev, dt)
g = torch.Generator(device=dev); g.manual_seed(1234)
init_noise = torch.randn((1, 4, T, 64, 64), generator=g, device=dev, dtype=dt)
os.makedirs(args.out, exist_ok=True)


def run(tag):
    return pipe.stream(ref_pil, mot, block_size=args.block, window=0, num_inference_steps=args.steps,
                       guidance_scale=1.0, decoder=args.decoder, taesd=taesd, init_noise=init_noise, verbose=False)


print("[render] teacher-causal ...")
vid_teacher = run("teacher")
if args.ckpt:
    print(f"[render] ODE-init-causal ({args.ckpt}) ...")
    sd = torch.load(args.ckpt, map_location="cpu")["denoising_unet"]
    denu.load_state_dict(sd, strict=True)
    vid_ode = run("odeinit")
    both = torch.cat([vid_teacher, vid_ode], dim=4)   # [teacher | odeinit]
    save_videos_grid(both, f"{args.out}/{n}_{args.decoder}_teacher_vs_odeinit.mp4", n_rows=1, fps=25)
    # 抽 cold-start 帧(4) 和 warm 帧(24) 做并排 png
    def grab(v, f): return (v[0, :, f].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    for f in [4, 24]:
        if f < T:
            row = Image.new("RGB", (512 * 2, 512))
            row.paste(Image.fromarray(grab(vid_teacher, f)), (0, 0))
            row.paste(Image.fromarray(grab(vid_ode, f)), (512, 0))
            row.save(f"{args.out}/{n}_frame{f}_{args.decoder}_teacher_vs_odeinit.png")
            print(f"  frame{f} (左teacher|右odeinit) -> {args.out}/{n}_frame{f}_{args.decoder}_teacher_vs_odeinit.png")
    print(f"[done] 视频 -> {args.out}/{n}_{args.decoder}_teacher_vs_odeinit.mp4")
else:
    save_videos_grid(vid_teacher, f"{args.out}/{n}_teacher_causal.mp4", n_rows=1, fps=25)
    print(f"[done] (无 ckpt) -> {args.out}/{n}_teacher_causal.mp4")
