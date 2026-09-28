"""帧提取 v2 —— 对齐 XNeMo 官方 pose2vid_xnemo.py 的做法。

与旧版 tool/extract_frames.py 的三处差异(都是旧版缺的):
  ① 裁剪中心做**高斯时序平滑**(σ=5,官方 pose2vid_xnemo.py:430-433 就是这么做的)
     旧版逐帧用原始检测中心 → 实测 24.7% 的背景轨迹是会被这个滤波消除的抖动
  ② **不做 int() 量化**,用 warpAffine 亚像素裁剪
     旧版 cx=int((x1+x2)/2) 把亚像素抖动量化成 0/±1 像素的跳变
  ③ 逐帧计算并保存 **bbox_param**(相对首帧的位置偏移+尺度比,与官方 get_bbox_param 同式)
     旧版从不保存,pose_extract 里写死 [0,0,1] → motion encoder 的 bbox_proj 通路是死的

★ 两种中心策略,用 --mode 选:
    smooth  官方做法:平滑跟随人脸(背景仍会平滑移动,但无抖动)
    fixed   首帧中心固定(背景完全静止,脸在画面里移动) —— 需要配合真实 bbox_param 使用
"""
import os, sys, argparse, cv2, numpy as np, torch
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
from scipy.ndimage import gaussian_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("--videos", nargs="+", default=[])
ap.add_argument("--list", default=None, help="clip 名清单(每行一个),与 --src 配合")
ap.add_argument("--src", default=XP("XN_HALLO3_RAW"))
ap.add_argument("--shard", default="0/1", help="i/n 分片")
ap.add_argument("--resume", type=int, default=1, help="1=跳过已有 .done 标记的 clip")
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--mode", choices=["smooth", "fixed"], default="smooth")
ap.add_argument("--sigma", type=float, default=5.0)
ap.add_argument("--max_frames", type=int, default=256)
ap.add_argument("--fps", type=float, default=25.0, help="目标帧率,按时间轴重采样")
ap.add_argument("--min_face_ratio", type=float, default=1.1,
                help="第二张脸边长/主脸边长阈值。**默认 >1 = 提取阶段不裁决**,\n                      只把 ratio 存进 _mfratio.npy,由 recheck_clean.py 按阈值筛")
ap.add_argument("--multi_min_frames", type=int, default=3,
                help="裁剪后仍检测到多人的帧数达到此值则排除整条(放宽以容忍偶发误检)")
a = ap.parse_args()

import insightface
# ★ 只加载 detection:buffalo_l 默认还会载 landmark_2d_106 / landmark_3d_68 / genderage /
#   recognition 共 5 个模型并逐帧全跑,而本脚本只用 bbox。实测 30.8 → 10.9 ms/帧(2.8×)。
#   bbox 来自同一个检测模型,输出与全量模式逐位一致。
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"],
                                   allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(640, 640))

def resize_shortside(f, s):
    h, w = f.shape[:2]; r = s/min(h, w)
    return cv2.resize(f, (int(round(w*r)), int(round(h*r))), interpolation=cv2.INTER_AREA)

if a.list:
    names=[l.strip() for l in open(a.list) if l.strip()]
    i,n=map(int, a.shard.split("/")); names=names[i::n]
    a.videos=[os.path.join(a.src, f"{x}.mp4") for x in names]
    print(f"[shard {i}/{n}] {len(a.videos)} 条", flush=True)

import shutil, time
_t0=time.time(); _n=0
for vp in a.videos:
    _nm=os.path.splitext(os.path.basename(vp))[0]
    _od=os.path.join(a.out, _nm)
    # ★ resume 判据必须用**只有当前版本才产出**的文件(_srcfps.npy,25fps 重采样版才写)。
    #   用 _bbox_param.npy 会把上一轮(未重采样)的旧帧误判为已完成 → 重提取瞬间"完成"却什么都没做。
    if a.resume and (os.path.exists(os.path.join(a.out, f"{_nm}_skip.npy"))
                 or (os.path.exists(os.path.join(a.out, f"{_nm}_srcfps.npy"))
                     and os.path.isdir(_od) and len(os.listdir(_od))>0)):
        _n+=1; continue
    if not os.path.exists(vp):
        print(f"[miss] {_nm}", flush=True); continue
    # ★ 原地覆盖前必须清空旧目录:新视频若帧数更少,残留旧帧会混进训练
    if os.path.isdir(_od): shutil.rmtree(_od)
    name = _nm
    cap = cv2.VideoCapture(vp)
    # ★ 帧率重采样到 25fps(a.fps)。数据集实测 78.5% 是 24fps、20.4% 是 25fps,还有 30/50/60/120fps。
    #   旧版逐帧全读、不重采样 → 24fps 素材被当 25fps 训练 = 画面快 4.2%,而音频按真实时长切,
    #   口型误差在 clip 内**线性累积**(74 帧的中位 clip 末尾差 ~3 帧);50fps 更是慢放一半。
    #   做法:按目标时间轴 j/25 秒取最近邻原始帧(等价 ffmpeg -r 25),音画时间轴严格对齐。
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    if not (fps_src and 1.0 < fps_src < 1000.0): fps_src = float(a.fps)   # 元数据坏则按目标帧率处理
    # ★ 流式重采样:第 j 个输出帧取原始第 round(j*fps_src/fps) 帧(即时间 j/fps 秒的最近邻)。
    #   不预读全部原始帧 —— 1080p×512 帧≈3.2GB/进程,32 进程会打爆内存;这里只留 resize 后的。
    #   上采样(24→25)时同一原始帧会被取用两次,是有意为之:保证音画时间轴严格对齐。
    frames = []; j = 0; k = 0
    while len(frames) < a.max_frames:
        ok, f = cap.read()
        if not ok: break
        small = None
        while j < a.max_frames and int(round(j * fps_src / a.fps)) == k:
            if small is None: small = resize_shortside(f, a.size)
            frames.append(small); j += 1
        k += 1
    cap.release()
    if len(frames) < 2: print(f"[skip] {name} 重采样后不足 2 帧"); continue
    n_out = len(frames)
    H, W = frames[0].shape[:2]

    # ---- 第一遍:逐帧检测人脸中心与尺寸(浮点,不量化)
    cen, lens, others = [], [], []
    for f in frames:
        fs = app.get(f)
        if not fs:
            cen.append(cen[-1] if cen else [W/2, H/2]); lens.append(lens[-1] if lens else a.size*0.5)
            others.append(0); continue
        ar = [(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]) for x in fs]
        mi = int(np.argmax(ar)); b = fs[mi].bbox
        cen.append([(b[0]+b[2])/2, (b[1]+b[3])/2]); lens.append(max(b[2]-b[0], b[3]-b[1]))
        others.append(len(fs))              # 该帧原图上检测到几张脸(仅用于挑候选帧)
    cen = np.asarray(cen, np.float32); lens = np.asarray(lens, np.float32)

    # ---- ① 高斯时序平滑(官方 σ=5)
    cen_s = gaussian_filter1d(cen, sigma=a.sigma, axis=0) if a.mode == "smooth" else \
            np.repeat(cen[:1], len(cen), axis=0)          # fixed:全程用首帧中心

    # ---- ★ 多人同框:**在裁剪后的 512 帧上判定**(用户定调)。
    #   裁剪只围绕最大的那张脸,原图里的其他人可能整个被裁掉、也可能留在画面里 ——
    #   只有留下来的才是干扰。直接在裁剪结果上检测,比用原图 bbox 做几何推算准确。
    #   ★ 只对"原图就检测到 ≥2 张脸"的帧做二次检测:绝大多数帧是单人,
    #     全量二次检测会让检测开销翻倍(检测是本脚本的主要成本)。
    half = a.size/2
    def _crop(i):
        cx = float(np.clip(cen_s[i][0], half, W-half)); cy = float(np.clip(cen_s[i][1], half, H-half))
        M = np.float32([[1, 0, half-cx], [0, 1, half-cy]])
        return cv2.warpAffine(frames[i], M, (a.size, a.size), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)
    # ★ 还要看第二张脸**多大**:背景里路过的人脸只有几十像素、糊成一团,
    #   对训练几乎没影响,不该和"两个人对着镜头说话"同等对待。
    #   用相对主脸的边长比(而非绝对像素):裁剪虽围绕主脸,主脸尺寸仍随景别变化。
    # ★ 不提前跳出、且把每帧的 ratio **完整存下来**:
    #   阈值只能靠分布定,存了原始度量才能事后重筛而不必重跑提取。
    #   默认阈值取**偏宽松**的值 —— 事后收紧只需从保留样本里再剔(有 ratio 就够),
    #   而放宽却要把误删的重新提取一遍,两个方向的代价不对称。
    mf_ratios = []
    for i_ in [k for k, c in enumerate(others) if c >= 2]:
        fs2 = app.get(_crop(i_))
        if len(fs2) < 2: continue
        sz = sorted((max(x.bbox[2]-x.bbox[0], x.bbox[3]-x.bbox[1]) for x in fs2), reverse=True)
        mf_ratios.append(sz[1] / max(sz[0], 1e-6))          # 第二大脸 / 最大脸 的边长比
    mf_ratios = np.asarray(mf_ratios, np.float32)
    np.save(os.path.join(a.out, f"{name}_mfratio.npy"), mf_ratios)
    # ★ 提取阶段**只测量、不裁决**:多人判据放到 recheck_clean.py 里按 _mfratio.npy 筛。
    #   这样调阈值只要重跑几分钟的清洗,而不必重跑几小时的提取
    #   —— 一旦在这里排除,帧就被删了,想放宽阈值就只能重新提取。
    multi_n = int((mf_ratios >= a.min_face_ratio).sum())
    if a.min_face_ratio <= 1.0 and multi_n >= a.multi_min_frames:
        np.save(os.path.join(a.out, f"{name}_skip.npy"),
                np.float32([1, multi_n, len(frames), multi_n/max(len(frames),1)]))
        print(f"[multi] {name} 裁剪后仍多人 {multi_n} 帧,排除", flush=True)
        _n += 1; continue

    # ---- ③ bbox_param:相对**首帧**的位置偏移+尺度比(与官方 get_bbox_param 同式)
    ref_c, ref_l = cen[0], lens[0]
    bbox_param = np.stack([(cen[:,1]-ref_c[1])/ref_l,      # dy (官方顺序是 [y,x])
                           (cen[:,0]-ref_c[0])/ref_l,      # dx
                           lens/ref_l], axis=1).astype(np.float32)

    # ---- ② 亚像素裁剪(warpAffine,不 int 量化)
    od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True)
    for i in range(len(frames)):
        cv2.imwrite(os.path.join(od, f"{i:06d}.jpg"), _crop(i), [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(os.path.join(a.out, f"{name}_bbox_param.npy"), bbox_param)
    np.save(os.path.join(a.out, f"{name}_srcfps.npy"),
            np.float32([fps_src, n_out, multi_n]))      # multi_n 留档,可事后按不同 N 重筛
    _n+=1
    if _n % 200 == 0:
        el=time.time()-_t0
        print(f"[prog] {_n}/{len(a.videos)}  {el/60:.1f}min  {_n/max(el,1):.2f} clip/s  "
              f"剩余 {(len(a.videos)-_n)/max(_n/max(el,1),1e-9)/3600:.1f}h", flush=True)
print(f"[done] shard 完成 {_n}/{len(a.videos)} 条, 用时 {(time.time()-_t0)/3600:.2f}h", flush=True)
