#!/bin/sh
# Mac equivalent of the handoff PowerShell entry. Exit status propagates failures.
set -eu
PROJECT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$PROJECT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo '请先按 docs/week02/README_Mac运行指南_中文.md 建立本项目独立环境。' >&2
    exit 1
fi
cd "$PROJECT"
exec "$PYTHON" tools/run_week02_mac.py "$@"
