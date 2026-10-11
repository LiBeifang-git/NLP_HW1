#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONHASHSEED=42 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
W2V_PYTHON="${W2V_PYTHON:-$PYTHON}"
case "${1:-all}" in
  task1) "$PYTHON" task1.py ;;
  task2) "$W2V_PYTHON" task2.py ;;
  task3) "$PYTHON" task3.py ;;
  prefix) "$PYTHON" task3_prefix.py ;;
  size) "$PYTHON" training_size.py ;;
  analyze) "$PYTHON" analyze.py ;;
  report)
    cd report
    xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape HW1实验报告.tex
    xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape HW1实验报告.tex
    ;;
  all)
    "$PYTHON" task1.py
    "$W2V_PYTHON" task2.py
    "$PYTHON" task3.py
    "$PYTHON" task3_prefix.py
    "$PYTHON" training_size.py
    "$PYTHON" analyze.py
    ;;
  *) echo '用法: bash run.sh {task1|task2|task3|prefix|size|analyze|report|all}'; exit 1 ;;
esac
