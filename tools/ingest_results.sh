#!/bin/bash
# 开发机(Claude 所在机器)导入实验机结果包。
# 用法: bash tools/ingest_results.sh <结果包.tar.gz>   → 解到 inbox/<包名>/ 并打印 SUMMARY.md
# 也可以不传包:把实验机打印出的 SUMMARY.md 文本直接贴进对话。
set -e
T=$1; [ -f "$T" ] || { echo "用法: bash tools/ingest_results.sh <结果包.tar.gz>"; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$REPO/inbox"
tar -xzf "$T" -C "$REPO/inbox"
D="$REPO/inbox/$(tar -tzf "$T" | head -1 | cut -d/ -f1)"
echo "已导入 → $D"; echo; cat "$D/SUMMARY.md"
