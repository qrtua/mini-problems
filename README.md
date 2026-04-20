# Mini-Problems Pipeline

Single-file pipeline for generating, solving, and evaluating creativity mini-problems using the **Gemini API**. Produces up to 500 semantically unique problems, each with ordinary/creative/implausible solutions.

## Quick Start

```bash
pip install google-generativeai numpy pandas sentence-transformers
export GEMINI_API_KEY=your_key

# Quick test (20 problems, generation only)
python pipeline.py --n-problems 20 --no-solve --no-evaluate

# Full run (500 problems)
python pipeline.py --n-problems 500 --cat-floor 35 --batch-size 10

# Run test suite (5 configurations x 50 problems)
bash run_tests.sh
```

## Architecture

Everything is in `pipeline.py` — four steps:

1. **Step 1: Generate** — problems + ordinary solutions in batches with stratified example sampling, category balancing, deduplication, and stop-loss
2. **Step 2a/2b: Creative & Implausible** — generated separately to prevent contamination between solution types
3. **Step 3: Solve** — model re-solves each problem without seeing original solutions (cross-validation)
4. **Step 4: Evaluate** — rates feasibility/novelty (1-5) and flags quality issues

## 10 Categories

Categories describe problem domains (what needs solving), not solution mechanisms:

| Category | Description | Approved examples |
|---|---|---|
| grip-friction | Getting grip, preventing sliding/slipping/rolling | 20 |
| makeshift-tool | Creating improvised tools/objects | 19 |
| containment-closure | Keeping things closed, together, bundled | 14 |
| protection-shielding | Blocking damage, weather, heat, insects | 13 |
| support-stabilization | Propping, standing, preventing tipping | 10 |
| cleaning-removal | Removing substances, stains, cleaning | 10 |
| attachment-fastening | Connecting, securing, sealing, patching | 8 |
| separation-extraction | Separating stuck things, filtering | 7 |
| reaching-retrieval | Accessing hard-to-reach things, extracting | 6 |
| temperature-management | Insulating, cooling, handling hot/cold | 5 |

## Configurable Parameters

### Example Injection (per API call)

| Parameter | Default | Description |
|---|---|---|
| `--n-same-cat` | 2 | Examples from target category (anchoring) |
| `--n-cross-cat` | 3 | Examples from other categories (diversity) |
| `--n-rejected` | 2 | Rejected examples, 1 per reason type |

### Generation

| Parameter | Default | Description |
|---|---|---|
| `--batch-size` | 5 | Problems generated per API call |
| `--cat-floor` | 35 | Minimum problems per category |
| `--temperature` | 0.8 | Generation temperature |
| `--model` | gemini-2.5-flash-preview-05-20 | Model to use |

### Diversity Control

| Parameter | Default | Description |
|---|---|---|
| `--similarity-threshold` | 0.75 | Starting semantic similarity threshold |
| `--similarity-step` | 0.05 | Threshold drop per 100 problems |
| `--similarity-floor` | 0.40 | Lowest the threshold can drop to |

### Stop-Loss

| Parameter | Default | Description |
|---|---|---|
| `--stoploss-window` | 20 | Sliding window size |
| `--stoploss-min-accepted` | 3 | Min accepted in window before stopping |
| `--stoploss-cat-streak` | 5 | Consecutive failures to skip a category |

### Pipeline Steps

| Parameter | Description |
|---|---|
| `--no-solve` | Skip solve step (saves time during testing) |
| `--no-evaluate` | Skip evaluate step |
| `--no-approved` | Don't include 112 approved problems in output |

## Test Configurations

```bash
# A: baseline
python pipeline.py --n-problems 50 --cat-floor 3 --no-solve --no-evaluate --output run_A.csv

# B: more category anchoring
python pipeline.py --n-problems 50 --cat-floor 3 --n-same-cat 3 --n-cross-cat 2 --no-solve --no-evaluate --output run_B.csv

# C: larger batches
python pipeline.py --n-problems 50 --cat-floor 3 --batch-size 10 --no-solve --no-evaluate --output run_C.csv

# D: minimal input
python pipeline.py --n-problems 50 --cat-floor 3 --n-same-cat 1 --n-cross-cat 2 --n-rejected 1 --no-solve --no-evaluate --output run_D.csv

# E: no rejected examples
python pipeline.py --n-problems 50 --cat-floor 3 --n-rejected 0 --no-solve --no-evaluate --output run_E.csv

# F: Gemini Pro (quality comparison)
python pipeline.py --n-problems 50 --cat-floor 3 --model gemini-2.5-pro-preview-05-06 --no-solve --no-evaluate --output run_F.csv
```

## Output Files

- `generated_problems.csv` — accepted problems with solutions and evaluations
- `rejected_during_generation.csv` — every rejected problem with reason, most similar existing problem, similarity score, and threshold used

## Data

- `data/examples.json` — 112 human-approved problems with category assignments
- `data/rejected.json` — 53 rejected/non-approved examples with rejection reasons

## Problem Format

- Problem: starts with "To", 4-7 words after "To", must include one explicit constraint
- Solutions: exactly 2-3 words each, concrete physical objects
- Problems must be physical, universal, knowledge-neutral

## Cost Estimates

| Scenario | Gemini Flash | Claude Sonnet 4.6 |
|---|---|---|
| 5 test runs x 50 (no solve/eval) | ~$0.14 | ~$1.02 |
| 500 problems (full pipeline) | ~$0.78 | ~$5.49 |
| Pessimistic (retries, more tokens) | ~$2 | ~$14 |
