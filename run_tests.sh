#!/bin/bash
# =============================================================
# Test runner — 5 configurations x 50 problems + optional run F
# =============================================================
# Usage:
#   export GEMINI_API_KEY=your_key
#   bash run_tests.sh              # Runs A-E (Gemini Flash)
#   bash run_tests.sh --all        # Runs A-F (including Gemini Pro)
#
# Estimated cost: ~$0.14 (A-E) or ~$0.20 (A-F) on Gemini Flash
# =============================================================

set -e

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: Set GEMINI_API_KEY"
    echo "  export GEMINI_API_KEY=your_key"
    exit 1
fi

mkdir -p test_results

echo "============================================"
echo "  MINI-PROBLEMS: CONFIGURATION TESTS"
echo "============================================"
echo ""

# Run A: baseline (default settings)
echo "[A] Baseline: 2 same-cat + 3 cross-cat, batch=5"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --no-solve --no-evaluate \
    --output test_results/run_A.csv

# Run B: more category anchoring
echo "[B] More anchoring: 3 same-cat + 2 cross-cat"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --n-same-cat 3 --n-cross-cat 2 \
    --no-solve --no-evaluate \
    --output test_results/run_B.csv

# Run C: larger batch size
echo "[C] Larger batch: batch=10"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --batch-size 10 \
    --no-solve --no-evaluate \
    --output test_results/run_C.csv

# Run D: minimal input
echo "[D] Minimal input: 1 same + 2 cross + 1 rejected"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --n-same-cat 1 --n-cross-cat 2 --n-rejected 1 \
    --no-solve --no-evaluate \
    --output test_results/run_D.csv

# Run E: no rejected examples
echo "[E] No rejected: n-rejected=0"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --n-rejected 0 \
    --no-solve --no-evaluate \
    --output test_results/run_E.csv

# Run F: Gemini Pro (optional)
if [ "$1" = "--all" ]; then
    echo "[F] Gemini Pro (baseline settings)"
    python pipeline.py --n-problems 50 --cat-floor 3 \
        --model gemini-2.5-pro-preview-05-06 \
        --no-solve --no-evaluate \
        --output test_results/run_F.csv
fi

echo ""
echo "============================================"
echo "  DONE — results in test_results/"
echo "============================================"
echo ""
ls -la test_results/

echo ""
echo "Next steps:"
echo "  1. Send test_results/*.csv to a team member for evaluation"
echo "  2. Evaluation criteria for each run:"
echo "     - Missing constraints"
echo "     - Knowledge-based problems"
echo "     - Thematic duplicates"
echo "     - Creative solutions that are too ordinary"
echo "     - Implausible solutions that would actually work"
