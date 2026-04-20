# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file pipeline for generating, solving, and evaluating creativity mini-problems using **Gemini API**. Produces 500 semantically unique problems, each with ordinary/creative/implausible solutions. Follows the Instructions PDF spec.

## Commands

```bash
# Install dependencies
pip install google-generativeai numpy pandas sentence-transformers

# Run pipeline
export GEMINI_API_KEY=your_key

# Quick test (20 problems, generation only)
python pipeline.py --n-problems 20 --no-solve --no-evaluate

# Full run (500 problems, floor 35 per category)
python pipeline.py --n-problems 500 --cat-floor 35 --batch-size 10

# Test different injection ratios
python pipeline.py --n-problems 20 --n-same-cat 3 --n-cross-cat 2   # more anchoring
python pipeline.py --n-problems 20 --n-same-cat 0 --n-cross-cat 5   # no anchoring
python pipeline.py --n-problems 20 --n-same-cat 1 --n-cross-cat 2 --n-rejected 1  # minimal

# Test different batch sizes
python pipeline.py --n-problems 50 --batch-size 1    # one at a time (old behavior)
python pipeline.py --n-problems 50 --batch-size 10   # larger batches

# Skip approved examples in output
python pipeline.py --n-problems 500 --no-approved
```

No test suite exists.

## Architecture

Everything is in `pipeline.py` — a single-file pipeline with 4 steps:

1. **Step 1: Generate** — problems + ordinary solutions in batches with:
   - Stratified example sampling (n_same_cat from target category + n_cross_cat from others)
   - Category balancing with minimum floor (cat_floor per category, rest random)
   - Rejected example sampling (1 per rejection reason type, maximizing info density)
   - DuplicateTracker: keyword overlap (0.80) + semantic similarity (adaptive threshold)
2. **Step 2a: Creative** — generated separately, uses secondary-feature mapping prompts
3. **Step 2b: Implausible** — generated separately, must have NO physical property that could solve
4. **Step 3: Solve** — Gemini re-solves each problem without seeing original solutions
5. **Step 4: Evaluate** — rates feasibility/novelty (1-5), flags quality issues

Key classes: `Config` (all knobs), `GeminiModel` (API wrapper), `DuplicateTracker` (2-layer dedup), `CategoryBalancer` (floor + random distribution).

## Configurable Parameters

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `--n-same-cat` | 2 | Examples from target category shown per call |
| `--n-cross-cat` | 3 | Examples from other categories shown per call |
| `--n-rejected` | 2 | Rejected examples shown per call (1 per reason type) |
| `--batch-size` | 5 | Problems generated per API call |
| `--cat-floor` | 35 | Minimum problems per category |
| `--similarity-threshold` | 0.75 | Base semantic similarity threshold (adaptive) |
| `--temperature` | 0.8 | Generation temperature |

## 10 Categories

Categories describe problem domains (what needs solving), not solution mechanisms:

1. **grip-friction** — getting grip, preventing sliding/slipping/rolling
2. **containment-closure** — keeping things closed, together, bundled
3. **protection-shielding** — blocking damage, weather, heat, insects
4. **cleaning-removal** — removing substances, stains, cleaning
5. **attachment-fastening** — connecting, securing, sealing, patching
6. **support-stabilization** — propping, standing, preventing tipping
7. **reaching-retrieval** — accessing hard-to-reach things, extracting from tight spaces
8. **makeshift-tool** — creating improvised tools/objects
9. **separation-extraction** — separating stuck things, filtering
10. **temperature-management** — insulating, cooling, handling hot/cold

## Data

- `data/examples.json` — 112 human-approved problems WITH category assignments
- `data/rejected.json` — 53 rejected/non-approved examples with reasons
