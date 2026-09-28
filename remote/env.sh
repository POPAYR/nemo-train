#!/bin/bash
# 被其他 remote/*.sh 与 exps/*.sh source:加载本机路径配置,并做最基本的校验。
XN_REPO_GUESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XN_HOST=${XN_HOST:-$(hostname -s)}
ENVF="$XN_REPO_GUESS/configs/paths/$XN_HOST.env"
if [ -f "$ENVF" ]; then source "$ENVF"
else echo "✗ 缺少 $ENVF —— 从 configs/paths/example.env 复制一份并改成本机路径(或 export XN_HOST=<名字>)"; return 1 2>/dev/null || exit 1; fi
export XN_REPO=${XN_REPO:-$XN_REPO_GUESS}
export XN_OUTPUT=${XN_OUTPUT:-$XN_REPO/output}
export XN_PY=${XN_PY:-python}
cd "$XN_REPO"
