"""分区域时序稳定性:把整帧拆成「中心=人脸」和「外围=背景」,分别测时间方向的波动。
动机:整帧平均的指标(运动量/抖动比/空间高频)分不出「背景在飘」和「人脸在动」,
而观感上背景飘是明显缺陷、人脸动是应该的。用户实测反馈 baseline 背景静而我们背景飘。
用法: python scripts/val/metric_region.py <dir_with_mp4s>
"""
import sys, glob, os, subprocess
import numpy as np

def read(path, W=512, H=512):
    r = subprocess.run(["ffmpeg","-v","error","-i",path,"-f","rawvideo","-pix_fmt","rgb24","-"],
                       capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8); n = a.size//(W*H*3)
    return a[:n*W*H*3].reshape(n,H,W,3).astype(np.float32).mean(3)   # [T,H,W] 灰度

def regions(H=512, W=512, r=0.32):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H/2, W/2
    face = ((yy-cy)**2/(r*H)**2 + (xx-cx)**2/(r*W)**2) < 1.0     # 中心椭圆≈人脸
    return face, ~face

d = sys.argv[1]
face_m, bg_m = regions()
print(f"{'文件':>46} | {'背景锐度':>9} {'人脸锐度':>9} | {'背景时序std':>11} {'人脸时序std':>11} | {'背景帧差':>9}")
print("-"*104)
for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
    g = read(p)
    if g.shape[0] < 3: continue
    tstd = g.std(axis=0)
    d1 = np.abs(np.diff(g, axis=0)).mean(axis=0)
    # 空间锐度:逐帧梯度幅值,再按区域平均(边缘用 valid 区避免越界)
    gx = np.abs(np.diff(g, axis=2)); gy = np.abs(np.diff(g, axis=1))
    sharp = np.zeros_like(g[0]); sharp[:, :-1] += gx.mean(axis=0); sharp[:-1, :] += gy.mean(axis=0)
    print(f"{os.path.basename(p):>46} | {sharp[bg_m].mean():>9.3f} {sharp[face_m].mean():>9.3f} | "
          f"{tstd[bg_m].mean():>11.3f} {tstd[face_m].mean():>11.3f} | {d1[bg_m].mean():>9.3f}")
