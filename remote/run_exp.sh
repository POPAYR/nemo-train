#!/bin/bash
# 用法: bash remote/run_exp.sh exps/<实验>.sh [额外环境变量,如 GPUS=0,1]
# 后台启动(关终端不影响),产物统一在 output/runs/<实验名>/,结束自动打包结果 → output/bundles/。
source "$(dirname "$0")/env.sh" || exit 1
EXPF=$1; [ -f "$EXPF" ] || { echo "用法: bash remote/run_exp.sh exps/<实验>.sh"; exit 1; }
EXP=$(basename "$EXPF" .sh); RUN=$XN_OUTPUT/runs/$EXP; mkdir -p "$RUN/logs"
{ echo "exp=$EXP"; echo "host=$(hostname)"; echo "start=$(date '+%F %T')"
  echo "git=$(git rev-parse HEAD) $(git diff --quiet || echo '(工作区有未提交改动!)')"
  echo "python=$($XN_PY --version 2>&1)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null; } > "$RUN/meta.txt"
$XN_PY -m pip freeze > "$RUN/pip_freeze.txt" 2>/dev/null
export EXP RUN
setsid nohup bash -c "bash '$EXPF' >> '$RUN/RUN.log' 2>&1; echo \"[exit=\$?] \$(date '+%F %T')\" >> '$RUN/RUN.log'; bash '$XN_REPO/remote/pack_results.sh' '$EXP'" > /dev/null 2>&1 < /dev/null &
echo "已后台启动 $EXP (pid $!)"
echo "  进度:tail -f $RUN/RUN.log"
echo "  结束后结果包:$XN_OUTPUT/bundles/${EXP}_*.tar.gz,同时生成 SUMMARY.md(可直接贴给 Claude)"
