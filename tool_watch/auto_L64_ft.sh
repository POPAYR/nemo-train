#!/bin/bash
# 自主编排(2026-09-24 用户授权"auto research,无需询问"):
#   ① 等 s2_L24c_anneal 结束 → ② 按 EMA vmse 最低选 ckpt → ③ 冒烟 → ④ 启动 L64 微调 + watcher
# 所有决策与结果写入 $W,供任何会话读取。
cd /media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python
A=output/_logs/active/s2_L24c_anneal.log
W=output/_logs/active/auto_L64_ft.watch
OUT=output/s2_L64ft
log(){ echo "[$(date +%m-%d\ %H:%M)] [auto] $*" >> $W; }
log "编排启动 pid=$$,等待退火结束"

# ① 等退火结束(正常 [done],或进程消失)
until grep -q "^\[done\]" $A || ! ps -eo args | grep -qE -- "[-]-out output/s2_L24c_anneal$"; do sleep 60; done
grep -q "^\[done\]" $A || { log "✗ 退火进程异常退出(无 [done]),中止编排"; exit 1; }
log "退火已结束"

# ② 选 ckpt:EMA vmse 最低(采样指标 4 clip 噪声太大,不参与)
BEST=$(grep -E "^\[val\] step" $A | awk '{st=$3; for(i=1;i<=NF;i++) if($i~/^vmse_mean=/){split($i,a,"=");print st, a[2]}}' | \
  while read st v; do [ -f output/s2_L24c_anneal/stage2_step_${st}.pt ] && echo "$st $v"; done | sort -k2 -g | head -1)
BS=${BEST%% *}; BV=${BEST##* }
CK=output/s2_L24c_anneal/stage2_step_${BS}.pt
[ -f "$CK" ] || { log "✗ 找不到可用 ckpt,中止"; exit 1; }
log "选定 $CK (EMA vmse=$BV);候选:$(grep -E '^\[val\] step' $A | awk '{print $3":"$4}' | tr '\n' ' ')"

# ③ 冒烟(CLAUDE.md §6.8):L64 路径 + resume_use_ema + 退火 20 步 + 保存
until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2) -lt 2000 ] && \
      [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7) -lt 2000 ]; do sleep 30; done
S=output/_smoke_L64ft; mkdir -p $S
CUDA_VISIBLE_DEVICES=2 timeout 1200 $PY -u scripts/train/flow_stage2_temporal.py --smoke --rope \
  --stage1_ckpt output/s1_uni/stage1_step_750.pt --resume $CK --resume_use_ema \
  --pose_real --bbox_drop 0 --sources 0 --t_mode uniform --lr 3e-5 --lr_anneal_steps 20 --lr_min 1e-6 \
  --sigma_shift 1.0 --cfg_drop 0.1 --ema_decay 0.999 --val_every 0 --out $S > $S/smoke.log 2>&1
if grep -q "^\[done\]" $S/smoke.log && ! grep -q Traceback $S/smoke.log; then
  log "冒烟通过:$(grep -E '^step' $S/smoke.log | tail -1 | cut -c1-70)"
else
  log "✗ 冒烟失败,见 $S/smoke.log,中止"; exit 1
fi

# ④ 启动 L64 微调:EMA 初始化,eff_bsz 8,lr 3e-5 余弦→1e-6 共 2000 步,单窗 64 val,8 clip 分片
#    先试 batch2×accum2;第一步前 OOM 则退到 batch1×accum4(eff_bsz 不变)
launch(){ # $1=batch $2=accum
  setsid nohup env CUDA_VISIBLE_DEVICES=2,7 $PY -u -m torch.distributed.run --nproc_per_node=2 --master_port 29891 \
    scripts/train/flow_stage2_temporal.py --rope \
    --stage1_ckpt output/s1_uni/stage1_step_750.pt --resume $CK --resume_use_ema \
    --pose_real --bbox_drop 0 --sources 0 --grad_stat --t_mode uniform \
    --lr 3e-5 --lr_anneal_steps 2000 --lr_min 1e-6 --L 64 --batch $1 --accum $2 \
    --sigma_shift 1.0 --cfg_drop 0.1 --ema_decay 0.999 --ema_device auto \
    --val_cfg 2.0 --val_window 0 \
    --save_every 500 --keep_last 12 --log_every 20 --val_every 250 --val_sample_every 500 \
    --val_clips 8 --val_sample_steps 20 \
    --out $OUT > output/_logs/active/s2_L64ft.log 2>&1 < /dev/null &
  sleep 10
  until grep -qE "^step |Traceback|out of memory" output/_logs/active/s2_L64ft.log; do sleep 20; done
  ! grep -qE "Traceback|out of memory" output/_logs/active/s2_L64ft.log
}
if launch 2 2; then
  log "L64 微调已启动(batch2×accum2×2卡)→ $OUT"
else
  log "batch2 启动失败:$(grep -m1 -E 'out of memory|Error' output/_logs/active/s2_L64ft.log | cut -c1-120);退到 batch1×accum4"
  pkill -f -- "--out $OUT\$" 2>/dev/null; sleep 30
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2) -lt 2000 ]; do sleep 10; done
  mv output/_logs/active/s2_L64ft.log output/_logs/active/s2_L64ft.fail_b2.log
  if launch 1 4; then log "L64 微调已启动(batch1×accum4×2卡)→ $OUT"
  else log "✗ batch1 也失败,中止:$(grep -m1 -E 'out of memory|Error' output/_logs/active/s2_L64ft.log | cut -c1-120)"; exit 1; fi
fi
log "首行:$(grep -m1 '^step ' output/_logs/active/s2_L64ft.log | cut -c1-90)"
setsid nohup tool_watch/watch_s2_v2.sh output/_logs/active/s2_L64ft.log s2_L64ft 0 2,7 >/dev/null 2>&1 < /dev/null &
echo "$CK" > output/_logs/active/auto_L64_ft.best_ckpt
log "watcher 已启动;编排 ① ~ ④ 完成"
