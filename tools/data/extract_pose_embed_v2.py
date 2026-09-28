"""帧 → motion embed (pose_embed)。重提取后必须重算,因为帧变了。

★ bbox_param 统一用 [0,0,1] —— 与官方推理 pose2vid_xnemo.py:417 一致
  ( ref_mot_bbox_param = ones; [:2] *= 0 )。
  注意本仓库两个旧脚本互相矛盾:pose_extract_dir.py 用 [0,0,1],
  pose_extract_dir_multi.py 用 [0,0,0]。这里以官方推理为准。

★ 逐帧真实 bbox_param 的信息已由 extract_frames_v2.py 存在 <name>_bbox_param.npy
  (相对首帧的 dy,dx,scale)。要改成相对任意参考帧只需换算,无需重提取:
     rel_to_j = ((dy_i-dy_j)/scale_j, (dx_i-dx_j)/scale_j, scale_i/scale_j)
  本脚本暂不启用,保持与官方推理同口径。
"""
import os, sys, glob, time, argparse
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
import numpy as np, torch, cv2
import mediapipe as mp
from PIL import Image
sys.path.append(third_party("motar"))
from model.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder
from diffusers.image_processor import VaeImageProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default=XP("XN_HALLO3", "face_frames"))
ap.add_argument("--out",    default=XP("XN_HALLO3", "pose_embed_real"))
ap.add_argument("--list",   default=XP("XN_HALLO3", "train_data.txt"))
ap.add_argument("--shard",  default="0/1")
ap.add_argument("--encoder", default=XP("XN_PRETRAINED", "xnemo_ckpt/xnemo_motion_encoder.pth"))
ap.add_argument("--detector", default=XP("REPO", "blaze_face_short_range.tflite"))
ap.add_argument("--chunk", type=int, default=64,
                help="每次送进 encoder 的帧数;encoder 逐帧独立,分块不改变结果")
ap.add_argument("--save_facemot", type=int, default=0,
                help="1=存 bbox 之前的 face_mot_feat(方案3);0=存最终 motion latent")
ap.add_argument("--real_bbox", type=int, default=0,
                help="1=喂逐帧真实 bbox_param(验证 bbox_proj 通路);0=官方口径常量 [0,0,1]")
ap.add_argument("--resume", type=int, default=1)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda:0"); dt = torch.bfloat16

enc = MotEncoder().to(dev, dt).eval()
sd = torch.load(a.encoder, map_location="cpu")
enc.load_state_dict(sd.get("state_dict", sd), strict=False)
proc = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=True)
det = mp.tasks.vision.FaceDetector.create_from_options(mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=a.detector),
    running_mode=mp.tasks.vision.RunningMode.IMAGE))

def face_crop(bgr):
    """裁人脸区域给 motion encoder(224)。检测失败则中心裁。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = rgb.shape[:2]
    if not r.detections:
        s = min(h, w); return Image.fromarray(rgb[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s])
    b = r.detections[0].bounding_box
    cx, cy = b.origin_x + b.width/2, b.origin_y + b.height/2
    s = max(b.width, b.height) * 1.4
    l, t = int(max(0, cx-s/2)), int(max(0, cy-s/2))
    rr, bb = int(min(w, cx+s/2)), int(min(h, cy+s/2))
    return Image.fromarray(rgb[t:bb, l:rr])

names = [l.strip() for l in open(a.list) if l.strip()]
i, n = map(int, a.shard.split("/")); names = names[i::n]
print(f"[shard {i}/{n}] {len(names)} 条", flush=True)
t0 = time.time(); done = 0
for nm in names:
    fd = os.path.join(a.frames, nm); op = os.path.join(a.out, f"{nm}.pt")
    fs = sorted(glob.glob(f"{fd}/*.jpg")) or sorted(glob.glob(f"{fd}/*.png"))
    if not fs: continue
    # ★ 与 encode_latents.py 同一坑:重提取后帧数不变、内容变了,resume 必须比 mtime
    if a.resume and os.path.exists(op) and os.path.getmtime(op) > os.path.getmtime(fd):
        try:
            if torch.load(op, map_location="cpu").shape[1] == len(fs): done += 1; continue
        except Exception: pass
    if a.real_bbox:
        try:
            _bbp = np.load(os.path.join(a.frames, f"{nm}_bbox_param.npy")).astype("float32")
            if len(_bbp) < len(fs): _bbp = np.concatenate([_bbp, np.repeat(_bbp[-1:], len(fs)-len(_bbp), 0)])
        except Exception as e:
            print(f"[err] {nm}: 缺 bbox_param {e}", flush=True); continue
    try:
        with torch.no_grad():
            ts = [proc.preprocess(face_crop(cv2.imread(f)), height=224, width=224).transpose(0,1) for f in fs]
            xs = torch.cat(ts, dim=1).unsqueeze(0)                          # [1,c,T,224,224] (CPU)
            # ★ 分块送进 encoder。encoder 内部是 rearrange("b c f h w -> (b f) c h w")
            #   —— 时间维直接展平进 batch、**逐帧独立编码,无任何时序依赖**,故分块结果逐位一致。
            #   整段一次性送(T 可达 512)会吃掉约 8GB 显存/进程,40 进程正好打满 4×80GB,
            #   进程全在等显存、吞吐崩到 9 clip/min。分块后显存与 T 无关。
            outs = []
            for j in range(0, xs.shape[2], a.chunk):
                xj = xs[:, :, j:j+a.chunk].to(dev, dt)
                if a.real_bbox:
                    # ★ 真实逐帧 bbox_param(dy,dx,scale),相对首帧。用于验证 bbox_proj 通路是否活着:
                    #   常量 [0,0,1] 会让 bbox_proj 输出恒定 = 该端口等于不存在,
                    #   而训练数据里背景确实随裁剪窗口平移(70% 样本 >0.1 人脸宽)。
                    bpj = torch.from_numpy(_bbp[j:j+xj.shape[2]]).unsqueeze(0).to(dev, dt)
                else:
                    bpj = torch.ones((1, xj.shape[2], 3), device=dev, dtype=dt); bpj[:,:,:2] = 0  # ★ [0,0,1]
                if a.save_facemot:
                    # ★ 方案3:存 bbox_proj **之前**的 face_mot_feat(512维/帧,与 bbox 无关)。
                    #   最终 motion latent = final_proj(cat[face_mot_feat, bbox_proj(bbox)]) + pe,
                    #   放到训练时现算 → bbox_param 完全自由(可 drop、可换策略),
                    #   不必再为改一个选择重提 1.7h 的数据。
                    #   encode_facemot 与 forward 里的 face_mot_feat 等价(out_drop 为 None)。
                    outs.append(enc.encode_facemot(xj).to(torch.float32).cpu())
                else:
                    outs.append(enc(xj, bpj).to(torch.float32).cpu())
            emb = torch.cat(outs, dim=1)
        torch.save(emb, op); done += 1
    except Exception as e:
        print(f"[err] {nm}: {type(e).__name__} {e}", flush=True); continue
    if done % 200 == 0:
        el = time.time()-t0
        print(f"[prog] {done}/{len(names)} {el/60:.1f}min {done/max(el,1):.2f} clip/s "
              f"剩余 {(len(names)-done)/max(done/max(el,1),1e-9)/3600:.1f}h", flush=True)
print(f"[done] {done}/{len(names)} 条, {(time.time()-t0)/3600:.2f}h", flush=True)
