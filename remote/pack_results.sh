#!/bin/bash
# 用法: bash remote/pack_results.sh <实验名>   —— 随时可手动运行(实验中途也行),打出当前进度。
# 只收小文件:日志(裁剪)、决策记录、指标、eval json、帧条 png、盲测答案;不收 ckpt / 视频。
source "$(dirname "$0")/env.sh" || exit 1
EXP=$1; RUN=$XN_OUTPUT/runs/$EXP; [ -d "$RUN" ] || { echo "✗ 没有 $RUN"; exit 1; }
TS=$(date +%m%d_%H%M); B=$XN_OUTPUT/bundles/${EXP}_$TS; mkdir -p "$B"
# 1) 原样收的小文件
( cd "$RUN" && find . -type f \( -name "*.txt" -o -name "*.log" -o -name "*.json" -o -name "*.watch" -o -name "*.md" \) \
    -not -path "*/_fid_png/*" -size -20M -print0 | xargs -0 -I{} cp --parents {} "$B/" )
# 帧条缩略图:每个训练阶段只取**最后一次**采样 val,缩一半转 JPEG(原 PNG 每张约 3MB)
$XN_PY - "$RUN" "$B" <<'PY'
import glob, os, sys
from PIL import Image
run, dst = sys.argv[1], sys.argv[2]
for ph in sorted(glob.glob(f"{run}/*/samples")):
    steps = sorted(glob.glob(f"{ph}/step_*"))
    if not steps: continue
    last = steps[-1]; od = os.path.join(dst, os.path.relpath(last, run)); os.makedirs(od, exist_ok=True)
    for f in glob.glob(f"{last}/*_strip.png"):
        im = Image.open(f).convert("RGB"); im = im.resize((im.width // 2, im.height // 2))
        im.save(os.path.join(od, os.path.basename(f)[:-4] + ".jpg"), quality=85)
PY
# 2) 训练日志裁剪:只留 step/val/save/lr/报错行 + 末尾 80 行(原始日志含大量进度条)
for lg in "$B"/logs/*.log; do [ -f "$lg" ] || continue
  { grep -aE "^step |^\[val\]|\[save\]|\[done\]|\[lr\]|\[init\]|\[pe\]|\[ema\]|Traceback|Error|out of memory" "$lg"; echo "---- tail ----"; tail -n 80 "$lg"; } > "$lg.trim" && mv "$lg.trim" "$lg"; done
# 3) 摘要
$XN_PY "$XN_REPO/tools/summarize_run.py" "$RUN" > "$B/SUMMARY.md" 2>&1
tar -czf "$B.tar.gz" -C "$XN_OUTPUT/bundles" "$(basename "$B")"
echo "结果包:$B.tar.gz ($(du -h "$B.tar.gz" | cut -f1))"; echo "==================== SUMMARY.md ===================="; cat "$B/SUMMARY.md"
