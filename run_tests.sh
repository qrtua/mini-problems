#!/bin/bash
# =============================================================
# Test runner — 5 konfiguracji po 50 problemów + opcjonalny run F
# =============================================================
# Użycie:
#   export GEMINI_API_KEY=your_key
#   bash run_tests.sh              # Runy A-E (Gemini Flash)
#   bash run_tests.sh --all        # Runy A-F (włącznie z Gemini Pro)
#
# Szacunkowy koszt: ~$0.14 (A-E) lub ~$0.20 (A-F) na Gemini Flash
# =============================================================

set -e

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: Ustaw GEMINI_API_KEY"
    echo "  export GEMINI_API_KEY=your_key"
    exit 1
fi

mkdir -p test_results

echo "============================================"
echo "  MINI-PROBLEMS: TESTY KONFIGURACJI"
echo "============================================"
echo ""

# Run A: baseline (domyślne ustawienia)
echo "[A] Baseline: 2 same-cat + 3 cross-cat, batch=5"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --no-solve --no-evaluate \
    --output test_results/run_A.csv

# Run B: więcej anchoringu kategorii
echo "[B] Więcej anchoringu: 3 same-cat + 2 cross-cat"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --n-same-cat 3 --n-cross-cat 2 \
    --no-solve --no-evaluate \
    --output test_results/run_B.csv

# Run C: większy batch
echo "[C] Większy batch: batch=10"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --batch-size 10 \
    --no-solve --no-evaluate \
    --output test_results/run_C.csv

# Run D: minimalny input
echo "[D] Minimalny input: 1 same + 2 cross + 1 rejected"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --n-same-cat 1 --n-cross-cat 2 --n-rejected 1 \
    --no-solve --no-evaluate \
    --output test_results/run_D.csv

# Run E: bez odrzuconych przykładów
echo "[E] Bez rejected: n-rejected=0"
python pipeline.py --n-problems 50 --cat-floor 3 \
    --n-rejected 0 \
    --no-solve --no-evaluate \
    --output test_results/run_E.csv

# Run F: Gemini Pro (opcjonalny)
if [ "$1" = "--all" ]; then
    echo "[F] Gemini Pro (baseline settings)"
    python pipeline.py --n-problems 50 --cat-floor 3 \
        --model gemini-2.5-pro-preview-05-06 \
        --no-solve --no-evaluate \
        --output test_results/run_F.csv
fi

echo ""
echo "============================================"
echo "  GOTOWE — wyniki w test_results/"
echo "============================================"
echo ""
ls -la test_results/

echo ""
echo "Następne kroki:"
echo "  1. Prześlij test_results/*.csv do członka zespołu"
echo "  2. Niech oceni każdy run wg kryteriów:"
echo "     - Brakujące constrainty"
echo "     - Knowledge-based problems"
echo "     - Duplikaty tematyczne"
echo "     - Creative solutions zbyt zwyczajne"
echo "     - Implausible solutions które by zadziałały"
