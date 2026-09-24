# Mini-Problems Pipeline

Single-file pipeline for generating, solving, and evaluating short creativity
"mini-problems" for a research study. Each problem gets three solutions —
**ordinary** (common & appropriate), **creative** (novel & appropriate), and
**implausible** (novel but inappropriate).

The pipeline is **multi-provider**: it runs on **Gemini** (`google-genai`),
**Anthropic** (Claude), or **Grok / xAI**, auto-detected from the model name.

## Quick Start

```bash
# Install the SDK for the provider you use (plus the shared deps)
pip install google-genai numpy pandas sentence-transformers
# optional, only if you use those providers:
# pip install anthropic        # Claude
# pip install openai           # Grok / xAI

export GEMINI_API_KEY=your_key        # or ANTHROPIC_API_KEY / XAI_API_KEY

# Quick test — problems + ordinary solutions only
python pipeline.py --examples data/approved.json --n-problems 20 --problems-only

# Generate against a seed, staying novel vs that seed
python pipeline.py --examples data/approved.json --n-problems 100 \
    --ref-threshold 0.65 --problems-only --output run_1.csv

# Full pipeline (generate → solve → evaluate)
python pipeline.py --examples data/approved.json --n-problems 200 --cat-floor 6
```

> **Note:** `sentence-transformers` is required for the semantic novelty gate
> and deduplication. Without it both silently disable and only exact-match
> protection remains — check the startup log for `Novelty gate: ENABLED`.

## Providers

The provider is auto-detected from the model name (override with `--provider`):

| Model prefix | Provider | SDK | Key |
|---|---|---|---|
| `gemini*` | Gemini | `google-genai` | `GEMINI_API_KEY` |
| `claude*` | Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| `grok*` | Grok / xAI | `openai` | `XAI_API_KEY` |

Default model: **`gemini-3.5-flash`**.

## Architecture

Everything lives in `pipeline.py`:

1. **Step 1 — Generate**: problems + ordinary solutions, in small batches, with
   stratified example sampling, category balancing, a novelty gate, deduplication,
   and two-level stop-loss.
2. **Step 2a / 2b — Creative & Implausible**: generated separately from the
   problem to prevent contamination between solution types.
3. **Step 3 — Solve**: a model re-solves each problem *without* seeing the
   original solutions (enables cross-model testing).
4. **Step 4 — Evaluate**: rates feasibility and novelty (1–5) per solution and
   flags quality issues.

Data flows through CSV files between steps.

## Novelty & Diversity

Two independent mechanisms keep output fresh:

- **Novelty gate (`ReferenceGate`)** — rejects any candidate that is semantically
  too close (cosine ≥ `--ref-threshold`, default **0.65**, *fixed* — no decay) to
  **any** problem in the reference set. The reference set is the seed passed via
  `--examples`, plus any prior-run output passed via `--extra-reference`. This is
  what makes a run "completely new" relative to what already exists. The seed is
  used as few-shot context and as this reference **only** — it is never added to
  the output.
- **Deduplicator** — removes near-duplicates *among the newly generated problems*:
  keyword Jaccard (0.80) plus cosine similarity (`all-MiniLM-L6-v2`) with an
  adaptive threshold (starts 0.75, drops 0.05 per 100 problems, floor 0.40).

## Target Categories (14)

Categories describe the physical *barrier* each problem is built around:

`The Access` · `The Attachment` · `The Calibration` · `The Containment` ·
`The Deformation` · `The Friction` · `The Improvisation` · `The Moisture` ·
`The Noise` · `The Protection` · `The Separation` · `The Stabilization` ·
`The Temperature` · `The Vision` Barrier.

## Seed / examples file

The pipeline needs a **seed file** of approved problems, supplied with
`--examples` (default path `data/examples.json`). The seed is used only as the
few-shot pool and as the novelty reference — it is never written to the output,
so every run generates entirely new problems.

It is a JSON array of objects. `problem` and `category` are required;
`constraint` and `ordinary_solution` are optional and may be left empty:

```json
[
  {
    "problem": "To open a jar with a stuck lid",
    "category": "The Access Barrier",
    "constraint": "",
    "ordinary_solution": ""
  }
]
```

`category` should be one of the 14 target categories above; any other value
(e.g. `"Other"`) still counts for novelty and deduplication but is never used as
a generation target. Optionally, a `data/rejected.json` file of the same shape,
plus a `rejection_reason` field, can be supplied to show the model concrete
failure modes to avoid (passed with `--n-rejected`).

## Configurable Parameters

### Seed & novelty

| Parameter | Default | Description |
|---|---|---|
| `--examples` | `data/examples.json` | Seed JSON: few-shot pool + novelty reference |
| `--ref-threshold` | 0.65 | Reject candidates with cosine ≥ this to any seed problem |
| `--extra-reference` | — | CSV of a prior run's problems to also stay novel against (run chaining) |

### Example injection (per API call)

| Parameter | Default | Description |
|---|---|---|
| `--n-same-cat` | 2 | Examples from the target category (anchoring) |
| `--n-cross-cat` | 3 | Examples from other categories (diversity) |
| `--n-rejected` | 2 | Rejected examples, 1 per reason type |

### Generation

| Parameter | Default | Description |
|---|---|---|
| `--batch-size` | 2 | Problems generated per API call (small = less intra-call correlation) |
| `--cat-floor` | 35 | Minimum problems per category |
| `--temperature` | 0.8 | Generation temperature |
| `--model` | gemini-3.5-flash | Model to use (provider auto-detected) |

### Diversity threshold (within-run dedup)

| Parameter | Default | Description |
|---|---|---|
| `--similarity-threshold` | 0.75 | Starting semantic threshold |
| `--similarity-step` | 0.05 | Threshold drop per 100 problems |
| `--similarity-floor` | 0.40 | Lowest the threshold can drop to |

### Stop-loss

| Parameter | Default | Description |
|---|---|---|
| `--stoploss-window` | 20 | Sliding window size |
| `--stoploss-min-accepted` | 3 | Min accepted in window before stopping |
| `--stoploss-cat-streak` | 5 | Consecutive failures to skip a category |

### Pipeline steps

| Parameter | Description |
|---|---|
| `--problems-only` | Only problems + ordinary (skips creative, implausible, solve, evaluate) |
| `--no-creative` | Skip creative-solution generation |
| `--no-implausible` | Skip implausible-solution generation |
| `--no-solve` | Skip the solve step |
| `--no-evaluate` | Skip the evaluate step |

## Output Files

- `generated_problems.csv` — accepted problems with solutions and evaluations.
- `rejected_during_generation.csv` — every rejected problem with reason, most
  similar existing problem, similarity score, and threshold used.

## Problem Format

- **Problem**: 4–10 words; a leading "To" is optional. A constraint that blocks
  the obvious solution is encouraged but not required, and need not appear
  verbatim.
- **Solutions**: 1–3 words each, a concrete physical object or action.
- Problems must be physical, universal, and knowledge-neutral.
