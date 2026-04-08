"""
Mini-Problems Pipeline — Generate, Solve & Evaluate
=====================================================
Single-file pipeline using Gemini 3 Flash Preview to produce 300 unique
mini-problems for a creativity study. One API call per problem.

Usage:
    export GEMINI_API_KEY=your_key
    python pipeline.py                              # 300 problems (default)
    python pipeline.py --n-problems 10              # quick test
    python pipeline.py --output results.csv
    python pipeline.py --no-approved                # skip pre-approved examples
"""
import os
import re
import csv
import json
import time
import random
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import google.generativeai as genai

log = logging.getLogger(__name__)

CATEGORIES = [
    "visuo-spatial", "material-substitution", "mechanical-advantage",
    "containment", "attachment", "support-stabilization",
    "protection-barrier", "tool-repurposing",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    api_key: str = ""
    model_name: str = "gemini-3-flash-preview"
    n_problems: int = 300
    n_examples: int = 5
    include_approved: bool = True
    use_rejected: bool = True
    seed: int = 42

    examples_path: str = "data/examples.json"
    rejected_path: str = "data/rejected.json"
    output_path: str = "generated_problems.csv"

    problem_min_words: int = 4
    problem_max_words: int = 7
    solution_min_words: int = 2
    solution_max_words: int = 3

    similarity_threshold: float = 0.75
    embed_model: str = "all-MiniLM-L6-v2"
    max_exclusion_items: int = 20

    temperature: float = 0.8
    max_output_tokens: int = 4096


# ---------------------------------------------------------------------------
# Gemini wrapper
# ---------------------------------------------------------------------------
# JSON schemas for structured output
GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "constraint": {"type": "string"},
        "category": {"type": "string"},
        "ordinary_solution": {"type": "string"},
        "creative_solution": {"type": "string"},
        "secondary_feature_used": {"type": "string"},
        "implausible_solution": {"type": "string"},
        "why_it_fails": {"type": "string"},
    },
    "required": [
        "problem", "constraint", "category",
        "ordinary_solution", "creative_solution", "secondary_feature_used",
        "implausible_solution", "why_it_fails",
    ],
}

SOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "ordinary_solution": {"type": "string"},
        "creative_solution": {"type": "string"},
        "implausible_solution": {"type": "string"},
    },
    "required": ["ordinary_solution", "creative_solution", "implausible_solution"],
}

EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "ordinary": {
            "type": "object",
            "properties": {"feasibility": {"type": "integer"}, "novelty": {"type": "integer"}},
            "required": ["feasibility", "novelty"],
        },
        "creative": {
            "type": "object",
            "properties": {"feasibility": {"type": "integer"}, "novelty": {"type": "integer"}},
            "required": ["feasibility", "novelty"],
        },
        "implausible": {
            "type": "object",
            "properties": {"feasibility": {"type": "integer"}, "novelty": {"type": "integer"}},
            "required": ["feasibility", "novelty"],
        },
        "quality_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ordinary", "creative", "implausible", "quality_flags"],
}


class GeminiModel:
    def __init__(self, cfg: Config):
        genai.configure(api_key=cfg.api_key)
        self.model_name = cfg.model_name
        self.cfg = cfg

    def _make_config(self, schema: dict) -> genai.GenerationConfig:
        return genai.GenerationConfig(
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

    def generate(self, system_prompt: str, user_prompt: str, schema: dict = None, retries: int = 3) -> str:
        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        gen_config = self._make_config(schema) if schema else genai.GenerationConfig(
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_output_tokens,
            response_mime_type="application/json",
        )
        for attempt in range(retries):
            try:
                resp = model.generate_content(user_prompt, generation_config=gen_config)
                return resp.text.strip()
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"API error (attempt {attempt+1}/{retries}): {e}. Retry in {wait}s...")
                time.sleep(wait)
        log.error("All retries failed.")
        return ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
GENERATE_SYSTEM = (
    "You generate a single mini-problem for a creativity study.\n\n"
    "FORMAT:\n"
    "- problem: starts with \"To\", then 4-7 words. Contains one explicit constraint.\n"
    "- constraint: the constraint word/phrase from the problem\n"
    "- category: one of the provided categories\n"
    "- ordinary_solution: EXACTLY 2-3 words. Most obvious solution.\n"
    "- creative_solution: EXACTLY 2-3 words. Unusual but works. Uses secondary feature of an object.\n"
    "- secondary_feature_used: what property makes the creative solution work\n"
    "- implausible_solution: EXACTLY 2-3 words. Real object that CANNOT solve the problem.\n"
    "- why_it_fails: why this object cannot solve the problem\n\n"
    "CRITICAL: Every solution MUST be EXACTLY 2 or 3 words. NOT a sentence. NOT a description. "
    "Just a short noun phrase like \"duct tape\" or \"a heavy book\".\n\n"
    "Examples of correct solutions: \"adhesive tape\", \"a towel\", \"nail polish\", \"rubber band\", \"a shoe\".\n"
    "Examples of WRONG solutions (too long): \"Wrap it in a towel\", \"Use a piece of string\".\n\n"
    "RULES:\n"
    "- Problems: physical, tangible, universal (no brands, no specialized knowledge)\n"
    "- NO knowledge-based problems (no life-hacks, no memorized tricks)\n\n"
    "IMPLAUSIBLE PITFALL: AI often picks objects that could actually work. "
    "Test: \"Could this object help via ANY physical property?\" If yes → not implausible.\n"
    "BAD: \"a metal spoon\" to cool something. GOOD: \"a pencil\" to keep shoes dry."
)

GENERATE_USER = (
    "Generate ONE new mini-problem that is DIFFERENT from all examples below.\n\n"
    "ACCEPTED examples:\n{examples_text}\n\n"
    "{rejected_text}"
    "Target category: {target_category}\n\n"
    "{exclusion_text}"
    "Return a single JSON object:\n"
    "{{\n"
    "  \"problem\": \"To ... (4-7 words after To)\",\n"
    "  \"constraint\": \"the constraint from the problem\",\n"
    "  \"category\": \"{target_category}\",\n"
    "  \"ordinary_solution\": \"2-3 word solution\",\n"
    "  \"creative_solution\": \"2-3 word novel but working solution\",\n"
    "  \"secondary_feature_used\": \"what property makes it work\",\n"
    "  \"implausible_solution\": \"2-3 word non-working solution\",\n"
    "  \"why_it_fails\": \"why this object cannot solve the problem\"\n"
    "}}"
)

SOLVE_SYSTEM = (
    "You solve short real-world mini-problems. Each has a constraint.\n\n"
    "Provide THREE solutions. Each MUST be EXACTLY 2 or 3 words — a short noun phrase, not a sentence.\n"
    "1. ORDINARY — most common, obvious\n"
    "2. CREATIVE — unusual but WORKS (uses secondary features of objects)\n"
    "3. IMPLAUSIBLE — real object that clearly WOULD NOT WORK"
)

SOLVE_USER = (
    "Solve: \"{problem}\"\n\n"
    "{{\n"
    "  \"ordinary_solution\": \"2-3 words\",\n"
    "  \"creative_solution\": \"2-3 words\",\n"
    "  \"implausible_solution\": \"2-3 words\"\n"
    "}}"
)

EVALUATE_SYSTEM = (
    "Evaluate solutions to a mini-problem.\n\n"
    "Rate each on:\n"
    "- FEASIBILITY (1-5): would it work physically?\n"
    "- NOVELTY (1-5): how surprising?\n\n"
    "Flag issues: creative_too_ordinary, implausible_is_plausible, "
    "knowledge_based, missing_constraint.\n\n"
    "Respond with a single JSON object."
)

EVALUATE_USER = (
    "Problem: \"{problem}\"\n"
    "Ordinary: \"{ordinary_solution}\"\n"
    "Creative: \"{creative_solution}\"\n"
    "Implausible: \"{implausible_solution}\"\n\n"
    "{{\n"
    "  \"ordinary\": {{\"feasibility\": <1-5>, \"novelty\": <1-5>}},\n"
    "  \"creative\": {{\"feasibility\": <1-5>, \"novelty\": <1-5>}},\n"
    "  \"implausible\": {{\"feasibility\": <1-5>, \"novelty\": <1-5>}},\n"
    "  \"quality_flags\": []\n"
    "}}"
)


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------
def parse_json(text: str) -> Optional[any]:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r'\{.*\}', r'\[.*\]']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    log.warning(f"JSON parse failed (len={len(text)}): {text[:200]}...")
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def count_words(text: str) -> int:
    return len(text.strip().split())


def validate_problem(item: Dict, cfg: Config) -> List[str]:
    issues = []
    problem = item.get("problem", "").strip()

    if not problem.lower().startswith("to "):
        issues.append("Must start with 'To'")
        return issues

    wc = count_words(problem[3:].strip())
    if not (cfg.problem_min_words <= wc <= cfg.problem_max_words):
        issues.append(f"Problem has {wc} words (need {cfg.problem_min_words}-{cfg.problem_max_words})")

    if not item.get("constraint", "").strip():
        issues.append("Missing constraint")

    for key in ["ordinary_solution", "creative_solution", "implausible_solution"]:
        sol = item.get(key, "").strip()
        if not sol:
            issues.append(f"Empty {key}")
        else:
            swc = count_words(sol)
            if not (cfg.solution_min_words <= swc <= cfg.solution_max_words):
                issues.append(f"{key} has {swc} words (need {cfg.solution_min_words}-{cfg.solution_max_words})")

    return issues


# ---------------------------------------------------------------------------
# Duplicate Tracker
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "to", "in", "on", "of", "for", "with", "without",
    "from", "by", "at", "is", "it", "my", "your", "when", "while",
    "and", "or", "not", "no", "that", "this", "its", "their", "you",
    "something", "someone", "things",
}


def _extract_keywords(text: str) -> set:
    words = re.sub(r'[^a-z\s]', '', text.lower()).split()
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _extract_verb(problem: str) -> str:
    core = problem.strip()
    if core.lower().startswith("to "):
        core = core[3:].strip()
    words = core.split()
    return words[0].lower() if words else ""


class DuplicateTracker:
    def __init__(self, similarity_threshold: float = 0.75, embed_model: str = "all-MiniLM-L6-v2"):
        self.similarity_threshold = similarity_threshold
        self.used_problems: List[str] = []
        self.used_verbs: Dict[str, int] = {}
        self.used_keywords: List[set] = []
        self.embeddings: List[np.ndarray] = []
        self.encoder = None

        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(embed_model)
            log.info("Semantic similarity: ENABLED")
        except ImportError:
            log.warning("sentence-transformers not installed — semantic filter DISABLED")

    def add(self, problem: str):
        problem = problem.strip()
        self.used_problems.append(problem)
        verb = _extract_verb(problem)
        self.used_verbs[verb] = self.used_verbs.get(verb, 0) + 1
        self.used_keywords.append(_extract_keywords(problem))
        if self.encoder:
            self.embeddings.append(self.encoder.encode(problem, normalize_embeddings=True))

    def get_threshold(self) -> float:
        n = len(self.used_problems)
        if n < 100:
            return self.similarity_threshold
        elif n < 200:
            return max(self.similarity_threshold - 0.10, 0.50)
        else:
            return max(self.similarity_threshold - 0.20, 0.45)

    def is_duplicate(self, problem: str) -> tuple:
        """Returns (is_dup: bool, reason: str)."""
        problem = problem.strip()
        keywords = _extract_keywords(problem)

        # Keyword overlap
        for used_kw in self.used_keywords:
            if keywords and used_kw:
                overlap = len(keywords & used_kw) / max(len(keywords | used_kw), 1)
                if overlap >= 0.8:
                    return True, f"keyword overlap {overlap:.2f}"

        # Semantic similarity
        if self.encoder and self.embeddings:
            emb = self.encoder.encode(problem, normalize_embeddings=True)
            stored = np.stack(self.embeddings)
            max_sim = float((stored @ emb).max())
            threshold = self.get_threshold()
            if max_sim >= threshold:
                return True, f"semantic sim {max_sim:.3f} >= {threshold:.2f}"

        return False, ""

    def get_exclusion_text(self) -> str:
        if not self.used_problems:
            return ""
        recent = self.used_problems[-20:]
        lines = ["DO NOT generate problems similar to these:"]
        for i, p in enumerate(recent, 1):
            lines.append(f"  {i}. \"{p}\"")

        top_verbs = sorted(self.used_verbs.items(), key=lambda x: -x[1])[:5]
        overused = [f"'{v}' ({c}x)" for v, c in top_verbs if c >= 2]
        if overused:
            lines.append(f"\nOverused verbs (avoid): {', '.join(overused)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_examples(examples: List[Dict], n: int = 5) -> str:
    selected = random.sample(examples, min(n, len(examples)))
    lines = []
    for i, ex in enumerate(selected, 1):
        parts = [f"{i}. \"{ex['problem']}\""]
        parts.append(f"   Ordinary: \"{ex.get('ordinary_solution', '')}\"")
        if ex.get("creative_solution"):
            parts.append(f"   Creative: \"{ex['creative_solution']}\"")
        if ex.get("implausible_solution"):
            parts.append(f"   Implausible: \"{ex['implausible_solution']}\"")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def format_rejected(rejected: List[Dict], n: int = 3) -> str:
    if not rejected:
        return ""
    selected = random.sample(rejected, min(n, len(rejected)))
    lines = ["REJECTED examples (do NOT make these mistakes):"]
    for i, ex in enumerate(selected, 1):
        lines.append(f"  {i}. \"{ex['problem']}\" — {ex.get('rejection_reason', 'bad quality')}")
    return "\n".join(lines) + "\n\n"


def get_target_category(counts: Dict[str, int]) -> str:
    min_count = min(counts.values())
    return random.choice([c for c, n in counts.items() if n == min_count])


# ---------------------------------------------------------------------------
# Step 1: Generate (one problem per request)
# ---------------------------------------------------------------------------
def step_generate(model: GeminiModel, examples: List[Dict], rejected: List[Dict],
                  cfg: Config, dedup: DuplicateTracker) -> List[Dict]:
    problems = []
    category_counts = {c: 0 for c in CATEGORIES}
    duplicates = 0
    failures = 0
    total = cfg.n_problems

    for ex in examples:
        dedup.add(ex["problem"])

    log.info(f"GENERATE: {total} problems (seeded {len(examples)} into dedup)")

    while len(problems) < total and failures < total * 3:
        target_cat = get_target_category(category_counts)
        examples_text = format_examples(examples, n=cfg.n_examples)
        rejected_text = format_rejected(rejected) if cfg.use_rejected else ""
        exclusion_text = dedup.get_exclusion_text()
        if exclusion_text:
            exclusion_text += "\n\n"

        user_prompt = GENERATE_USER.format(
            examples_text=examples_text,
            rejected_text=rejected_text,
            target_category=target_cat,
            exclusion_text=exclusion_text,
        )

        response = model.generate(GENERATE_SYSTEM, user_prompt, schema=GENERATE_SCHEMA)
        parsed = parse_json(response)

        if not parsed or not isinstance(parsed, dict):
            failures += 1
            log.warning(f"  Parse failed ({failures} total)")
            continue

        issues = validate_problem(parsed, cfg)
        if issues:
            failures += 1
            log.warning(f"  INVALID: {parsed.get('problem', '?')} — {issues}")
            continue

        problem_text = parsed["problem"].strip()
        is_dup, reason = dedup.is_duplicate(problem_text)
        if is_dup:
            duplicates += 1
            log.info(f"  DUPLICATE: \"{problem_text}\" — {reason}")
            continue

        dedup.add(problem_text)
        problems.append(parsed)
        cat = parsed.get("category", "")
        if cat in category_counts:
            category_counts[cat] += 1

        log.info(f"  [{len(problems)}/{total}] \"{problem_text}\"")
        time.sleep(0.3)

    log.info(f"  Done: {len(problems)} generated, {duplicates} duplicates, {failures} failures")
    return problems


# ---------------------------------------------------------------------------
# Step 2: Solve (cross-validation)
# ---------------------------------------------------------------------------
def step_solve(model: GeminiModel, problems: List[Dict], cfg: Config) -> List[Dict]:
    log.info(f"SOLVE: {len(problems)} problems")

    for i, item in enumerate(problems):
        response = model.generate(SOLVE_SYSTEM, SOLVE_USER.format(problem=item["problem"]), schema=SOLVE_SCHEMA)
        parsed = parse_json(response)

        if parsed and isinstance(parsed, dict):
            item["solved_ordinary"] = parsed.get("ordinary_solution", "")
            item["solved_creative"] = parsed.get("creative_solution", "")
            item["solved_implausible"] = parsed.get("implausible_solution", "")
        else:
            item["solved_ordinary"] = ""
            item["solved_creative"] = ""
            item["solved_implausible"] = ""

        if (i + 1) % 50 == 0:
            log.info(f"  Solved {i+1}/{len(problems)}")
        time.sleep(0.3)

    log.info(f"  Done")
    return problems


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------
def step_evaluate(model: GeminiModel, problems: List[Dict], cfg: Config) -> List[Dict]:
    log.info(f"EVALUATE: {len(problems)} problems")

    for i, item in enumerate(problems):
        response = model.generate(
            EVALUATE_SYSTEM,
            EVALUATE_USER.format(
                problem=item.get("problem", ""),
                ordinary_solution=item.get("ordinary_solution", ""),
                creative_solution=item.get("creative_solution", ""),
                implausible_solution=item.get("implausible_solution", ""),
            ),
            schema=EVALUATE_SCHEMA,
        )
        parsed = parse_json(response)

        if parsed and isinstance(parsed, dict):
            for sol_type in ["ordinary", "creative", "implausible"]:
                ratings = parsed.get(sol_type, {})
                item[f"eval_{sol_type}_feasibility"] = ratings.get("feasibility")
                item[f"eval_{sol_type}_novelty"] = ratings.get("novelty")
            item["quality_flags"] = parsed.get("quality_flags", [])
        else:
            item["quality_flags"] = ["parse_error"]

        if (i + 1) % 50 == 0:
            log.info(f"  Evaluated {i+1}/{len(problems)}")
        time.sleep(0.3)

    log.info(f"  Done")
    return problems


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(problems: List[Dict], path: str):
    cols = [
        "problem", "category", "constraint",
        "ordinary_solution", "creative_solution", "implausible_solution",
        "secondary_feature_used", "why_it_fails",
        "solved_ordinary", "solved_creative", "solved_implausible",
        "eval_ordinary_feasibility", "eval_ordinary_novelty",
        "eval_creative_feasibility", "eval_creative_novelty",
        "eval_implausible_feasibility", "eval_implausible_novelty",
        "quality_flags",
    ]
    rows = []
    for item in problems:
        row = {}
        for c in cols:
            val = item.get(c, "")
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            row[c] = val
        rows.append(row)

    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    log.info(f"Saved {len(rows)} problems to {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    with open(cfg.examples_path) as f:
        examples = json.load(f)
    log.info(f"Loaded {len(examples)} examples")

    rejected = []
    if cfg.use_rejected and os.path.exists(cfg.rejected_path):
        with open(cfg.rejected_path) as f:
            rejected = json.load(f)
        log.info(f"Loaded {len(rejected)} rejected")

    model = GeminiModel(cfg)
    dedup = DuplicateTracker(cfg.similarity_threshold, cfg.embed_model)

    # How many new problems to generate?
    approved = []
    if cfg.include_approved:
        approved = [dict(ex) for ex in examples if ex.get("ordinary_solution")]
        n_new = max(0, cfg.n_problems - len(approved))
        log.info(f"Including {len(approved)} approved, generating {n_new} new")
    else:
        n_new = cfg.n_problems

    gen_cfg = Config(**{**cfg.__dict__, "n_problems": n_new})
    new_problems = step_generate(model, examples, rejected, gen_cfg, dedup) if n_new > 0 else []

    all_problems = approved + new_problems

    all_problems = step_solve(model, all_problems, cfg)
    all_problems = step_evaluate(model, all_problems, cfg)

    save_csv(all_problems, cfg.output_path)

    valid = sum(1 for p in all_problems if not p.get("quality_flags"))
    log.info(f"DONE: {len(all_problems)} total, {valid} without quality flags")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Mini-Problems Pipeline")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--n-problems", type=int, default=300)
    parser.add_argument("--output", default="generated_problems.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--no-rejected", action="store_true")
    parser.add_argument("--no-approved", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if not args.api_key:
        parser.error("Set GEMINI_API_KEY or use --api-key")

    cfg = Config(
        api_key=args.api_key,
        model_name=args.model,
        n_problems=args.n_problems,
        output_path=args.output,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
        use_rejected=not args.no_rejected,
        include_approved=not args.no_approved,
        temperature=args.temperature,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
