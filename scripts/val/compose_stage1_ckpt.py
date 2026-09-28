"""把 stage1 的空间主干合进 stage2 teacher(保留其 temporal),产出可渲染的完整 ckpt。

为什么需要:stage1 只有空间主干(1232 键,无 temporal),单独无法渲染视频,
而 σ 分辨 v-MSE 与最终画质只是弱相关(实测 v-MSE 变 0.3% ↔ FVD 变 5.7%),
只靠它选 stage1 的点信号太弱。

做法:取 CUM2000 的完整 state_dict(1232 空间 + 546 temporal),
把其中 1232 个空间键换成新 stage1 的。
⚠️ 这是**故意的错配组合**——temporal 是在旧空间主干上训的。但对所有候选点错配方式相同,
   所以只能做**相对比较**(哪个 stage1 更好),不能当作最终 teacher 的绝对水平。
"""
import argparse, torch, os
ap = argparse.ArgumentParser()
ap.add_argument("--stage1", required=True)
ap.add_argument("--temporal_from", default="output/flow_stage2_cfgdrop/CUM2000.pt")
ap.add_argument("--out", required=True)
a = ap.parse_args()
base = torch.load(a.temporal_from, map_location="cpu")
sp = torch.load(a.stage1, map_location="cpu")["denoising_unet"]
sd = base["denoising_unet"]
n = 0
for k in sd:
    if "temporal_modules" not in k:
        assert k in sp, f"stage1 缺键 {k}"
        sd[k] = sp[k]; n += 1
assert n == 1232, n
torch.save({"denoising_unet": sd, "rope": base.get("rope", True),
            "cfg_drop": base.get("cfg_drop"), "objective": "rectified_flow_v", "stage": 2,
            "composed_from": {"spatial": a.stage1, "temporal": a.temporal_from}},
           a.out)
print(f"[compose] 空间{n} ← {os.path.basename(a.stage1)} | temporal ← {os.path.basename(a.temporal_from)}"
      f" → {a.out}", flush=True)
