#!/bin/bash
# 独立 watcher:setsid 常驻,不依赖 Claude 会话。输出写 $L.watch,Claude 侧只 tail 它。
# 用法: watch_s2.sh <训练日志> <标签> <步数偏移> <GPU列表>
L=$1; TAG=$2; OFF=${3:-0}; GPUS=${4:-0}
W=${L%.log}.watch
cd /media/ps/ssd5/ayr/x-nemo-inference
echo "[$(date +%m-%d\ %H:%M)] [$TAG] watcher 启动 pid=$$" >> $W
# 事件流:val / 崩溃 / 结束 → 立即写
( tail -n0 -F $L 2>/dev/null | grep -E --line-buffered "^\[val\] step|Traceback|out of memory|Killed|No space left|assert|\[done\]|\[save\]" \
  | while IFS= read -r line; do echo "[$(date +%m-%d\ %H:%M)] [$TAG] $line" >> $W; done ) &
# 心跳 + 收敛判据 + 进程存活:每 30 分钟
while true; do
  sleep 1800
  if ! ps -eo args | grep -q "[f]low_stage2_temporal.*--out output/${TAG}\b"; then
    if grep -q "^\[done\]" $L; then
      echo "[$(date +%m-%d\ %H:%M)] ✓ [$TAG] 训练正常结束([done]) | $(grep -E '^step ' $L | tail -1)" >> $W
    else
      echo "[$(date +%m-%d\ %H:%M)] ✗ [$TAG] 训练进程异常消失(无 [done]) | $(grep -E '^step ' $L | tail -1)" >> $W
    fi
    pkill -P $$ 2>/dev/null; exit 0
  fi
  st=$(grep -E "^step " $L | tail -1 | awk '{print $2}')
  it=$(grep -E "^step " $L | tail -1 | grep -oP "[0-9.]+(?=s/it)")
  conv=$(grep -E "^\[val\] step" $L | sed -E 's/.*vmse_mean=([0-9.]+).*/\1/' | tail -4 | /usr/bin/python3 -c "
import sys
v=[float(x) for x in sys.stdin.read().split()]
if len(v)<4: print('vmse 点数不足')
else:
    d=[100*(v[i+1]-v[i])/v[i] for i in range(3)]
    print('vmse降幅 '+' '.join('%.3f%%'%x for x in d)+('  ★可判收敛' if all(abs(x)<0.05 for x in d) else ''))")
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPUS | tr -d ' ' | paste -sd/)
  echo "[$(date +%m-%d\ %H:%M)] [$TAG] step ${st} (真实 $((st+OFF))) | ${it}s/it | 显存 $m | $conv" >> $W
done
