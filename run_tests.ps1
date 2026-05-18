# =============================================================
# Test runner â€" 5 configurations x 50 problems + optional run F
# =============================================================
# Usage:
#   $env:GEMINI_API_KEY = "your_key"
#   .\run_tests.ps1              # Runs A-E (Gemini Flash)
#   .\run_tests.ps1 --all        # Runs A-F (including Gemini Pro)
#
# Estimated cost: ~$0.14 (A-E) or ~$0.20 (A-F) on Gemini Flash
# =============================================================

$ErrorActionPreference = "Stop"

if (-not $env:GEMINI_API_KEY) {
    Write-Error "ERROR: Set GEMINI_API_KEY`n  `$env:GEMINI_API_KEY = 'your_key'"
    exit 1
}

New-Item -ItemType Directory -Force -Path test_results | Out-Null

Write-Host "============================================"
Write-Host "  MINI-PROBLEMS: CONFIGURATION TESTS"
Write-Host "============================================"
Write-Host ""

# Run A: baseline (default settings)
Write-Host "[A] Baseline: 2 same-cat + 3 cross-cat, batch=5"
python pipeline.py --no-approved --n-problems 50 --cat-floor 3 `
    --no-solve --no-evaluate `
    --output test_results/run_A.csv

# Run B: more category anchoring
Write-Host "[B] More anchoring: 3 same-cat + 2 cross-cat"
python pipeline.py --no-approved --n-problems 50 --cat-floor 3 `
    --n-same-cat 3 --n-cross-cat 2 `
    --no-solve --no-evaluate `
    --output test_results/run_B.csv

# Run C: larger batch size
Write-Host "[C] Larger batch: batch=10"
python pipeline.py --no-approved --n-problems 50 --cat-floor 3 `
    --batch-size 10 `
    --no-solve --no-evaluate `
    --output test_results/run_C.csv

# Run D: minimal input
Write-Host "[D] Minimal input: 1 same + 2 cross + 1 rejected"
python pipeline.py --no-approved --n-problems 50 --cat-floor 3 `
    --n-same-cat 1 --n-cross-cat 2 --n-rejected 1 `
    --no-solve --no-evaluate `
    --output test_results/run_D.csv

# Run E: no rejected examples
Write-Host "[E] No rejected: n-rejected=0"
python pipeline.py --no-approved --n-problems 50 --cat-floor 3 `
    --n-rejected 0 `
    --no-solve --no-evaluate `
    --output test_results/run_E.csv

# Run F: Gemini Pro (optional)
if ($args[0] -eq "--all") {
    Write-Host "[F] Gemini Pro (baseline settings)"
    python pipeline.py --n-problems 50 --cat-floor 3 `
        --model gemini-2.5-pro-preview-05-06 `
        --no-solve --no-evaluate `
        --output test_results/run_F.csv
}

Write-Host ""
Write-Host "============================================"
Write-Host "  DONE â€" results in test_results/"
Write-Host "============================================"
Write-Host ""
Get-ChildItem test_results/

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Send test_results/*.csv to a team member for evaluation"
Write-Host "  2. Evaluation criteria for each run:"
Write-Host "     - Missing constraints"
Write-Host "     - Knowledge-based problems"
Write-Host "     - Thematic duplicates"
Write-Host "     - Creative solutions that are too ordinary"
Write-Host "     - Implausible solutions that would actually work"



