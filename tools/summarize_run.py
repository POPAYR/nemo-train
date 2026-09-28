#!/usr/bin/env python3
"""把 output/runs/<实验>/ 汇总成一份 Markdown(SUMMARY.md),供实验机 → 开发机(Claude)传递结果。

设计目标:几十行、纯文本、可直接贴进对话;包含判断实验所需的全部数字,不需要原始日志。
内容:元信息 / 决策记录(RUN.log 末尾) / 每个训练阶段的 val 曲线与进度 / 评测表(metrics.log, regions.log) / 报错。
用法: python tools/summarize_run.py output/runs/<实验>
"""
import glob
import os
import re
import sys

run = sys.argv[1].rstrip("/")
name = os.path.basename(run)
out = [f"# 实验 {name}\n"]


def tail(p, n):
    try:
        return open(p, errors="ignore").read().splitlines()[-n:]
    except OSError:
        return []


meta = os.path.join(run, "meta.txt")
if os.path.exists(meta):
    out += ["## 元信息", "```", *open(meta).read().strip().splitlines(), "```"]

runlog = os.path.join(run, "RUN.log")
if os.path.exists(runlog):
    lines = [l for l in tail(runlog, 400) if not re.search(r"Warning|deprecat|it/s\]|\d+%\|", l)]
    out += ["## 决策记录(RUN.log 末尾)", "```", *lines[-40:], "```"]

for lg in sorted(glob.glob(os.path.join(run, "logs", "*.log"))):
    L = open(lg, errors="ignore").read().splitlines()
    phase = os.path.basename(lg)[:-4]
    vals = [l for l in L if l.startswith("[val] step")]
    steps = [l for l in L if l.startswith("step ")]
    done = any(l.startswith("[done]") for l in L)
    errs = [i for i, l in enumerate(L) if "Traceback" in l or "out of memory" in l]
    out += [f"## 阶段 {phase}  —  {'✓ 已完成' if done else ('✗ 出错' if errs else '… 进行中')}"]
    if steps:
        out.append(f"- 最新:`{steps[-1][:150]}`")
    for l in L:
        if l.startswith(("[lr]", "[pe]", "[init]")):
            out.append(f"- `{l[:160]}`")
    if vals:
        out += ["", "| step | vmse | 降幅 | 采样指标(cfg2,EMA;训练内 val 的 FVD 不看) |", "|---|---|---|---|"]
        prev = None
        for v in vals:
            st = re.search(r"step (\d+)", v).group(1)
            vm = float(re.search(r"vmse_mean=([0-9.]+)", v).group(1))
            d = f"{100 * (vm - prev) / prev:+.3f}%" if prev else ""
            prev = vm
            smp = v.split("|", 1)[1].split("(")[0].strip() if "|" in v else ""
            out.append(f"| {st} | {vm:.5f} | {d} | {smp} |")
    for i in errs[:2]:
        out += ["", "报错:", "```", *L[max(0, i - 3): i + 12], "```"]

for kind in ("metrics.log", "regions.log"):
    for f in sorted(glob.glob(os.path.join(run, "**", kind), recursive=True)):
        L = open(f, errors="ignore").read().splitlines()
        k = next((i for i, l in enumerate(L) if l.startswith("来源")), None)
        if k is None:
            continue
        end = next((j for j in range(k + 1, len(L)) if not L[j].strip()), len(L))
        out += [f"## 评测 {os.path.relpath(f, run)}", "```", *L[k:end + 2], "```"]

keys = glob.glob(os.path.join(run, "**", "_key", "key.json"), recursive=True)
if keys:
    out += ["## 盲测", *[f"- 答案:`{os.path.relpath(k, run)}`(视频在实验机同目录,先看视频再看答案)" for k in keys]]

print("\n".join(out))
