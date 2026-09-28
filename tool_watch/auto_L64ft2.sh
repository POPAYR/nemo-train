#!/bin/bash
# 用户 2026-09-28 批准:64 微调恒定 lr 续训 → 退火 → 20 clip 评测。检验 s2_L64ft 是否"被 lr 冻住"而非真收敛。
# 09-28 04:04 首次启动 OOM(GPU6 被他人占 10G);改 batch1×accum4(eff_bsz 仍 8,数学等价)
cd /media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python; FPY=/home/ayr/miniconda3/envs/face/bin/python
W=output/_logs/active/auto_L64ft2.watch; GP=4,6
log(){ echo "[$(date +%m-%d\ %H:%M)] [ft2] $*" >> $W; }
COMMON="--rope --stage1_ckpt output/s1_uni/stage1_step_750.pt --resume_use_ema --pose_real --bbox_drop 0 --sources 0 --grad_stat --t_mode uniform --L 64 --batch 1 --accum 4 --sigma_shift 1.0 --cfg_drop 0.1 --ema_decay 0.999 --ema_device auto --val_cfg 2.0 --val_window 0 --save_every 500 --keep_last 12 --log_every 20 --val_every 250 --val_sample_every 500 --val_clips 8 --val_sample_steps 20"
run(){ # $1=out $2=resume $3..=额外参数;等 [done] 或失败
  local out=$1 ck=$2; shift 2; local lg=output/_logs/active/$(basename $out).log
  if grep -q "^\[done\]" $lg 2>/dev/null; then log "$out 已完成,跳过"; return 0; fi
  if ps -eo args | grep -qE -- "[-]-out $out( |\$)"; then log "$out 已在运行,接续等待"
  else
  setsid nohup env CUDA_VISIBLE_DEVICES=$GP $PY -u -m torch.distributed.run --nproc_per_node=2 --master_port $PORT \
    scripts/train/flow_stage2_temporal.py $COMMON --resume $ck --out $out "$@" > $lg 2>&1 < /dev/null &
  sleep 10
  until grep -qE "^step |Traceback|out of memory" $lg; do sleep 20; done
  grep -qE "Traceback|out of memory" $lg && { log "✗ $out 启动失败:$(grep -m1 -E 'out of memory|Error:' $lg | cut -c1-120)"; return 1; }
  log "$out 已启动:$(grep -m1 '^step ' $lg | cut -c1-100)"
  setsid nohup tool_watch/watch_s2_v2.sh $lg $(basename $out) 0 $GP >/dev/null 2>&1 < /dev/null &
  fi
  until grep -q "^\[done\]" $lg || ! ps -eo args | grep -qE -- "[-]-out $out( |\$)"; do sleep 60; done
  grep -q "^\[done\]" $lg || { log "✗ $out 异常结束(无 [done])"; return 1; }
  log "$out 正常结束;vmse 序列:$(grep -E '^\[val\] step' $lg | awk '{printf "%s:%s ", $3, substr($4,11)}')"
}
log "编排启动 pid=$$"
# 冒烟(CLAUDE.md §6.8):恒定 lr 路径(不退火)+ resume_use_ema
S=output/_smoke_L64ft2; mkdir -p $S
if ! grep -q "^\[done\]" $S/smoke.log 2>/dev/null; then
CUDA_VISIBLE_DEVICES=4 timeout 1200 $PY -u scripts/train/flow_stage2_temporal.py --smoke --rope --stage1_ckpt output/s1_uni/stage1_step_750.pt \
  --resume output/s2_L64ft/stage2_step_2000.pt --resume_use_ema --pose_real --bbox_drop 0 --sources 0 --t_mode uniform \
  --lr 1.5e-5 --sigma_shift 1.0 --cfg_drop 0.1 --ema_decay 0.999 --val_every 0 --out $S > $S/smoke.log 2>&1
grep -q "^\[done\]" $S/smoke.log && ! grep -q Traceback $S/smoke.log || { log "✗ 冒烟失败 $S/smoke.log"; exit 1; }
log "冒烟通过:$(grep -E '^step' $S/smoke.log | tail -1 | cut -c1-80)"
fi
# 阶段 A:恒定 lr 1.5e-5,2000 步
PORT=29901 run output/s2_L64ft2 output/s2_L64ft/stage2_step_2000.pt --lr 1.5e-5 --max_steps 2000 || exit 1
# 阶段 B:余弦退火 1.5e-5 → 1e-6,1000 步
PORT=29903 run output/s2_L64ft2a output/s2_L64ft2/stage2_step_2000.pt --lr 1.5e-5 --lr_anneal_steps 1000 --lr_min 1e-6 || exit 1
# 评测:20 clip 单窗 64
T=output/eval/teacher_cmp; L=$T/L24c_s9000/_lists; A=$T/L64ft2a_s1000; B=$T/L64ft2_s2000; mkdir -p $A $B
freegpu(){ while true; do g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<50000{print $1}' | grep -vxE "$1" | head -1); [ -n "$g" ] && { echo $g; return; }; sleep 60; done; }
G1=$(freegpu "x"); G2=$(freegpu "$G1"); log "评测渲染 GPU $G1,$G2"
R(){ env CUDA_VISIBLE_DEVICES=$5 VAL_CLIPS_FILE=$L/$4.txt $PY -u /media/ps/ssd5/ayr/tool/render_s2_cfg.py --s2_ckpt $2 \
  --stage1_ckpt output/s1_uni/stage1_step_750.pt --use_ema --window 0 --shift 1.0 --cfgs 2.0 --steps 20 --clips 10 --data_seed $6 --out $1 > $1/render_$4.log 2>&1; }
R $A output/s2_L64ft2a/stage2_step_1000.pt 0 a $G1 1234 & R $A output/s2_L64ft2a/stage2_step_1000.pt 0 b $G2 1244 &
R $B output/s2_L64ft2/stage2_step_2000.pt 0 a $G1 1234 & R $B output/s2_L64ft2/stage2_step_2000.pt 0 b $G2 1244 &
wait; log "渲染完成 $(ls $A/*_cfg2.0.mp4 | wc -l) + $(ls $B/*_cfg2.0.mp4 | wc -l) 条"
D="$A $B $T/L64ft_s2000 $T/flow $T/eps"; NM=("64续训+退火" "64续训恒定lr@2000" "64微调@2000(旧)" "64_S单窗" "eps(cfg2.5)")
CUDA_VISIBLE_DEVICES=$G1 $PY /media/ps/ssd5/ayr/tool/video_metrics.py --dirs $D --names "${NM[@]}" --work $T/_metrics_L64ft2 > $A/metrics.log 2>&1
CUDA_VISIBLE_DEVICES=$G1 $PY /media/ps/ssd5/ayr/tool/teacher_compare.py --dirs $D --names "${NM[@]}" > $A/regions.log 2>&1
Bl=output/eval/blind_L64ft2_vs_L64ft
$FPY /media/ps/ssd5/ayr/tool/region_zoom_blind.py --dirs $A $T/L64ft_s2000 --names "64续训+退火" "64微调@2000(旧)" --seed 11 --out $Bl > $Bl.log 2>&1
log "评测完成:$A/metrics.log $A/regions.log;盲测 $Bl"
log "[FT2_DONE]"
