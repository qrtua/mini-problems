# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file pipeline for generating, solving, and evaluating creativity mini-problems using **Gemini 3 Flash Preview** API. Produces 300 semantically unique problems, each with ordinary/creative/implausible solutions. Follows the Instructions PDF spec.

## Commands

```bash
# Install dependencies
pip install google-generativeai numpy pandas sentence-transformers

# Run pipeline
export GEMINI_API_KEY=your_key
python pipeline.py                              # 300 problems (default)
python pipeline.py --n-problems 10              # quick test
python pipeline.py --output results.csv         # custom output
python pipeline.py --similarity-threshold 0.65  # stricter diversity
python pipeline.py --no-rejected                # skip rejected examples
python pipeline.py --no-approved                # generate all 300 from scratch
```

By default, the 112 human-approved problems from the CSV are included in the output. The pipeline generates only the remaining ~188 new problems to reach the 300 target. Use `--no-approved` to generate all from scratch.

No test suite exists.

## Architecture

Everything is in `pipeline.py` — a single-file pipeline with 4 steps:

1. **Step 1: Generate** — problems + ordinary solutions in batches with diversity control (DuplicateTracker: keyword overlap + semantic similarity via sentence-transformers). Category balancing across 8 categories. Adaptive similarity threshold decreases as the set grows (0.75→0.65→0.55).
2. **Step 2a/2b: Creative & Implausible** — generated separately for existing problems. Creative uses secondary-feature mapping. Implausible must have NO physical property that could solve the problem.
3. **Step 3: Solve** — Gemini re-solves each problem without seeing original solutions (cross-validation).
4. **Step 4: Evaluate** — rates feasibility/novelty (1-5) and flags quality issues (creative_too_ordinary, implausible_is_plausible, knowledge_based, missing_constraint).

Key classes: `Config` (dataclass), `GeminiModel` (API wrapper with retry), `DuplicateTracker` (2-layer dedup).

## Constraints (Instructions PDF)

- Problem: starts with "To", 4-7 words after "To", must include one explicit constraint
- Solutions: 2-3 words each, concrete physical objects
- Problems must be physical, universal, knowledge-neutral
- 300 problems must be semantically distinct (no meaning overlap)

## Data

- `data/examples.json` — 112 human-approved problems from `#Revised_MPv2 - Sheet1.csv` (used for few-shot prompting + included in output by default)
- `data/rejected.json` — 53 rejected/non-approved examples with reasons (anti-overfitting)
- `#Revised_MPv2 - Sheet1.csv` — source CSV with 1000 human-rated problems (112 approved, 33 rejected, 63 non-approved, 95 repeated, 697 unreviewed)
