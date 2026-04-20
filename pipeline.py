"""
Mini-Problems Pipeline — Generate, Solve & Evaluate
=====================================================
Single-file pipeline using Gemini API to produce semantically unique
mini-problems for a creativity study.

Key features:
  - Stratified example sampling (same-category + cross-category)
  - Configurable batch size (problems per API call)
  - Category balancing with minimum floor
  - Configurable injection ratios (approved/rejected per call)
  - 2-step generation: problems+ordinary first, creative+implausible separate

Usage:
    export GEMINI_API_KEY=your_key
    python pipeline.py                                  # defaults
    python pipeline.py --n-problems 500                 # 500 new problems
    python pipeline.py --batch-size 10                  # 10 per API call
    python pipeline.py --n-same-cat 2 --n-cross-cat 3   # example injection
    python pipeline.py --n-rejected 2                   # rejected per call
    python pipeline.py --cat-floor 35                   # min per category
    python pipeline.py --no-approved                    # don't include 112 in output
"""
import os
import re
import csv
import json
import time
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import google.generativeai as genai

log = logging.getLogger(__name__)

CATEGORIES = [
    "grip-friction",
    "containment-closure",
    "protection-shielding",
    "cleaning-removal",
    "attachment-fastening",
    "support-stabilization",
    "reaching-retrieval",
    "makeshift-tool",
    "separation-extraction",
    "temperature-management",
]


# ---------------------------------------------------------------------------
# Config — all knobs in one place
# ---------------------------------------------------------------------------
@dataclass
class Config:
    api_key: str = ""
    model_name: str = "gemini-2.5-flash-preview-05-20"
    n_problems: int = 500
    seed: int = 42

    # --- Example injection (per API call) ---
    n_same_cat: int = 2          # examples from target category
    n_cross_cat: int = 3         # examples from other categories
    n_rejected: int = 2          # rejected examples (1 per reason type)

    # --- Batching ---
    batch_size: int = 5          # problems generated per API call

    # --- Category balancing ---
    cat_floor: int = 35          # minimum problems per category
    # Remaining (n_problems - cat_floor * n_cats) distributed randomly

    # --- Include approved in output ---
    include_approved: bool = True

    # --- Data paths ---
    examples_path: str = "data/examples.json"
    rejected_path: str = "data/rejected.json"
    output_path: str = "generated_problems.csv"

    # --- Validation ---
    problem_min_words: int = 4
    problem_max_words: int = 7
    solution_min_words: int = 2
    solution_max_words: int = 3

    # --- Diversity ---
    similarity_threshold: float = 0.75
    similarity_floor: float = 0.40       # lowest the threshold can drop to
    similarity_step: float = 0.05        # how much threshold drops per 100 problems
    embed_model: str = "all-MiniLM-L6-v2"
    max_exclusion_items: int = 20

    # --- Stop-loss ---
    stoploss_window: int = 20            # look at last N attempts
    stoploss_min_accepted: int = 3       # if fewer than this accepted in window → stop
    stoploss_cat_streak: int = 5         # consecutive failures per category → skip it

    # --- Rejection tracking ---
    save_rejected: bool = True           # save rejected problems to file
    rejected_output_path: str = "rejected_during_generation.csv"

    # --- Generation ---
    temperature: float = 0.8
    max_output_tokens: int = 4096

    # --- Pipeline steps ---
    run_solve: bool = True
    run_evaluate: bool = True


# ---------------------------------------------------------------------------
# Gemini wrapper
# ---------------------------------------------------------------------------
GENERATE_STEP1_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "problem": {"type": "string"},
            "constraint": {"type": "string"},
            "category": {"type": "string"},
            "ordinary_solution": {"type": "string"},
        },
        "required": ["problem", "constraint", "category", "ordinary_solution"],
    },
}

GENERATE_CREATIVE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "problem": {"type": "string"},
            "creative_solution": {"type": "string"},
            "secondary_feature_used": {"type": "string"},
        },
        "required": ["problem", "creative_solution", "secondary_feature_used"],
    },
}

GENERATE_IMPLAUSIBLE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "problem": {"type": "string"},
            "implausible_solution": {"type": "string"},
            "why_it_fails": {"type": "string"},
        },
        "required": ["problem", "implausible_solution", "why_it_fails"],
    },
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

    def _make_config(self, schema: dict = None) -> genai.GenerationConfig:
        kwargs = {
            "temperature": self.cfg.temperature,
            "max_output_tokens": self.cfg.max_output_tokens,
            "response_mime_type": "application/json",
        }
        if schema:
            kwargs["response_schema"] = schema
        return genai.GenerationConfig(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str,
                 schema: dict = None, retries: int = 3) -> str:
        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        gen_config = self._make_config(schema)
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
# JSON parser
# ---------------------------------------------------------------------------
def parse_json(text: str) -> Optional[any]:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    log.warning(f"JSON parse failed: {text[:200]}...")
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def count_words(text: str) -> int:
    return len(text.strip().split())


def validate_problem_step1(item: Dict, cfg: Config) -> List[str]:
    """Validate problem + ordinary solution from Step 1."""
    issues = []
    problem = item.get("problem", "").strip()
    if not problem.lower().startswith("to "):
        issues.append("Must start with 'To'")
        return issues
    wc = count_words(problem[3:].strip())
    if not (cfg.problem_min_words <= wc <= cfg.problem_max_words):
        issues.append(f"Problem: {wc} words (need {cfg.problem_min_words}-{cfg.problem_max_words})")
    if not item.get("constraint", "").strip():
        issues.append("Missing constraint")
    sol = item.get("ordinary_solution", "").strip()
    if not sol:
        issues.append("Empty ordinary_solution")
    else:
        swc = count_words(sol)
        if not (cfg.solution_min_words <= swc <= cfg.solution_max_words):
            issues.append(f"ordinary_solution: {swc} words (need {cfg.solution_min_words}-{cfg.solution_max_words})")
    return issues


def validate_solution(sol: str, cfg: Config) -> bool:
    """Check if a solution meets word count requirements."""
    if not sol.strip():
        return False
    wc = count_words(sol)
    return cfg.solution_min_words <= wc <= cfg.solution_max_words


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
    def __init__(self, similarity_threshold: float = 0.75,
                 similarity_floor: float = 0.40,
                 similarity_step: float = 0.05,
                 embed_model: str = "all-MiniLM-L6-v2"):
        self.base_threshold = similarity_threshold
        self.similarity_floor = similarity_floor
        self.similarity_step = similarity_step
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

    def get_threshold(self) -> float:
        """
        Adaptive threshold: drops by similarity_step for every 100 problems,
        never below similarity_floor.

        With defaults (base=0.75, step=0.05, floor=0.40):
          0-99:   0.75
          100-199: 0.70
          200-299: 0.65
          300-399: 0.60
          400-499: 0.55
          500-599: 0.50
          600+:    0.45, 0.40 (floor)

        Configurable via CLI:
          --similarity-threshold 0.80 --similarity-step 0.03 --similarity-floor 0.50
        """
        n = len(self.used_problems)
        steps_down = n // 100
        threshold = self.base_threshold - (steps_down * self.similarity_step)
        return max(threshold, self.similarity_floor)

    def add(self, problem: str):
        problem = problem.strip()
        self.used_problems.append(problem)
        verb = _extract_verb(problem)
        self.used_verbs[verb] = self.used_verbs.get(verb, 0) + 1
        self.used_keywords.append(_extract_keywords(problem))
        if self.encoder:
            self.embeddings.append(
                self.encoder.encode(problem, normalize_embeddings=True)
            )

    def is_duplicate(self, problem: str) -> Dict:
        """
        Check if problem is a duplicate.
        Returns dict with full details:
            {
                "is_dup": bool,
                "reason": str,
                "most_similar_problem": str,   # which existing problem it's closest to
                "similarity_score": float,      # how similar (0-1)
                "similarity_type": str,         # "keyword" | "semantic" | ""
                "threshold_used": float,        # current threshold at time of check
            }
        """
        problem = problem.strip()
        keywords = _extract_keywords(problem)
        result = {
            "is_dup": False,
            "reason": "",
            "most_similar_problem": "",
            "similarity_score": 0.0,
            "similarity_type": "",
            "threshold_used": self.get_threshold(),
        }

        # Layer 1: keyword overlap
        best_kw_overlap = 0.0
        best_kw_idx = -1
        for i, used_kw in enumerate(self.used_keywords):
            if keywords and used_kw:
                overlap = len(keywords & used_kw) / max(len(keywords | used_kw), 1)
                if overlap > best_kw_overlap:
                    best_kw_overlap = overlap
                    best_kw_idx = i

        if best_kw_overlap >= 0.8:
            result["is_dup"] = True
            result["reason"] = f"keyword overlap {best_kw_overlap:.2f}"
            result["most_similar_problem"] = self.used_problems[best_kw_idx]
            result["similarity_score"] = best_kw_overlap
            result["similarity_type"] = "keyword"
            return result

        # Layer 2: semantic similarity
        if self.encoder and self.embeddings:
            emb = self.encoder.encode(problem, normalize_embeddings=True)
            stored = np.stack(self.embeddings)
            sims = stored @ emb
            max_idx = int(sims.argmax())
            max_sim = float(sims[max_idx])
            threshold = self.get_threshold()

            result["most_similar_problem"] = self.used_problems[max_idx]
            result["similarity_score"] = round(max_sim, 3)
            result["similarity_type"] = "semantic"

            if max_sim >= threshold:
                result["is_dup"] = True
                result["reason"] = f"semantic sim {max_sim:.3f} >= {threshold:.2f}"
                return result

        return result

    def get_exclusion_text(self, max_items: int = 20) -> str:
        if not self.used_problems:
            return ""
        recent = self.used_problems[-max_items:]
        lines = ["DO NOT generate problems similar to these:"]
        for i, p in enumerate(recent, 1):
            lines.append(f"  {i}. \"{p}\"")

        top_verbs = sorted(self.used_verbs.items(), key=lambda x: -x[1])[:6]
        overused = [f"'{v}' ({c}x)" for v, c in top_verbs if c >= 2]
        if overused:
            lines.append(f"\nOverused verbs (avoid): {', '.join(overused)}")

        return "\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            "total": len(self.used_problems),
            "unique_verbs": len(self.used_verbs),
            "threshold": self.get_threshold(),
            "top_verbs": sorted(self.used_verbs.items(), key=lambda x: -x[1])[:5],
        }


# ---------------------------------------------------------------------------
# Stratified example sampling
# ---------------------------------------------------------------------------
def sample_examples(
    examples: List[Dict],
    target_category: str,
    n_same_cat: int,
    n_cross_cat: int,
) -> List[Dict]:
    """
    Sample examples with category awareness:
    - n_same_cat from the target category (anchors what the category means)
    - n_cross_cat from other categories (shows diversity of good problems)
    Cross-category sampling maximizes verb diversity.
    """
    same_cat = [ex for ex in examples if ex.get("category") == target_category]
    cross_cat = [ex for ex in examples if ex.get("category") != target_category]

    # Sample same-category (or fewer if not enough)
    n_same = min(n_same_cat, len(same_cat))
    selected_same = random.sample(same_cat, n_same) if n_same > 0 else []

    # Sample cross-category with verb diversity
    n_cross = min(n_cross_cat, len(cross_cat))
    if n_cross > 0:
        # Group by verb, pick one from each verb group to maximize diversity
        by_verb: Dict[str, List[Dict]] = {}
        for ex in cross_cat:
            v = _extract_verb(ex.get("problem", ""))
            by_verb.setdefault(v, []).append(ex)

        selected_cross = []
        verb_keys = list(by_verb.keys())
        random.shuffle(verb_keys)
        for v in verb_keys:
            if len(selected_cross) >= n_cross:
                break
            selected_cross.append(random.choice(by_verb[v]))

        # If we need more, fill randomly from remaining
        if len(selected_cross) < n_cross:
            remaining = [ex for ex in cross_cat if ex not in selected_cross]
            extra = random.sample(remaining, min(n_cross - len(selected_cross), len(remaining)))
            selected_cross.extend(extra)
    else:
        selected_cross = []

    return selected_same + selected_cross


def sample_rejected(
    rejected: List[Dict],
    n: int,
) -> List[Dict]:
    """
    Sample rejected examples: 1 per rejection reason type,
    then fill remaining randomly. Maximizes information per example.
    """
    if not rejected or n <= 0:
        return []

    # Group by reason
    by_reason: Dict[str, List[Dict]] = {}
    for ex in rejected:
        reason = ex.get("rejection_reason", "unknown")
        # Normalize compound reasons to first reason
        first_reason = reason.split(";")[0].strip()
        by_reason.setdefault(first_reason, []).append(ex)

    selected = []
    # One per reason type
    reason_keys = list(by_reason.keys())
    random.shuffle(reason_keys)
    for reason in reason_keys:
        if len(selected) >= n:
            break
        selected.append(random.choice(by_reason[reason]))

    # Fill if needed
    if len(selected) < n:
        remaining = [ex for ex in rejected if ex not in selected]
        extra = random.sample(remaining, min(n - len(selected), len(remaining)))
        selected.extend(extra)

    return selected


# ---------------------------------------------------------------------------
# Category balancer
# ---------------------------------------------------------------------------
class CategoryBalancer:
    """Tracks category counts and determines target for next batch."""
    def __init__(self, categories: List[str], cat_floor: int, total_target: int):
        self.categories = categories
        self.cat_floor = cat_floor
        self.total_target = total_target
        self.counts: Dict[str, int] = {c: 0 for c in categories}

    def seed(self, problems: List[Dict]):
        """Count categories from existing problems."""
        for p in problems:
            cat = p.get("category", "")
            if cat in self.counts:
                self.counts[cat] += 1

    def get_target_category(self) -> str:
        """Pick the category that's furthest below its target."""
        # Categories below floor get priority
        below_floor = {c: self.cat_floor - n for c, n in self.counts.items() if n < self.cat_floor}
        if below_floor:
            # Pick the one with biggest deficit
            max_deficit = max(below_floor.values())
            candidates = [c for c, d in below_floor.items() if d == max_deficit]
            return random.choice(candidates)
        # All at floor — pick least represented
        min_count = min(self.counts.values())
        candidates = [c for c, n in self.counts.items() if n == min_count]
        return random.choice(candidates)

    def record(self, category: str):
        if category in self.counts:
            self.counts[category] += 1

    def remaining(self) -> int:
        return max(0, self.total_target - sum(self.counts.values()))

    def summary(self) -> str:
        lines = ["Category distribution:"]
        for cat in self.categories:
            n = self.counts[cat]
            target = self.cat_floor
            status = "✓" if n >= target else f"need {target - n} more"
            lines.append(f"  {cat:25s}: {n:3d} ({status})")
        lines.append(f"  {'TOTAL':25s}: {sum(self.counts.values())}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts — Step 1: problems + ordinary solutions
# ---------------------------------------------------------------------------
STEP1_SYSTEM = (
    "You generate short, concrete real-world mini-problems for a creativity study.\n\n"
    "EVERY problem starts with the word \"To\" followed by 4-7 more words.\n\n"
    "STRICT RULES:\n"
    "1. FORMAT: \"To [verb] [object/situation] [constraint]\" — e.g., \"To seal a non-sticky envelope\"\n"
    "2. WORD COUNT: The part AFTER \"To\" must be 4-7 words. Count carefully.\n"
    "3. CONSTRAINT RULE: Every problem MUST include exactly ONE explicit constraint word/phrase "
    "that blocks the standard solution. Good constraints: \"non-sticky\", \"without a belt\", "
    "\"broken\", \"no scissors\", \"missing handle\". The constraint must appear literally in the problem text.\n"
    "4. PHYSICAL ONLY: Problems must involve tangible, physical objects and actions. "
    "No abstract, social, emotional, or moral problems.\n"
    "5. UNIVERSAL: No brand names, no regional products, no specialized knowledge. "
    "Any adult worldwide should understand.\n\n"
    "CRITICAL — AVOID THESE COMMON MISTAKES:\n"
    "- NO KNOWLEDGE-BASED PROBLEMS: Do NOT generate problems whose solutions rely on memorized "
    "tricks, life-hacks, or specialized know-how.\n"
    "- NO MISSING CONSTRAINTS: Every problem MUST have a constraint.\n\n"
    "ORDINARY SOLUTION RULES:\n"
    "- The most common, first-thing-anyone-would-think-of solution.\n"
    "- Must be EXACTLY 2-3 words.\n"
    "- Must be a concrete physical object or action.\n\n"
    "You MUST respond with ONLY a valid JSON array. No explanation, no markdown."
)

STEP1_USER = (
    "Here are examples of ACCEPTED mini-problems:\n\n"
    "{examples_text}\n\n"
    "{rejected_text}"
    "Generate {batch_size} NEW and DIFFERENT mini-problems in the category \"{target_category}\".\n\n"
    "Diversity requirements:\n"
    "- Use different verbs (don't repeat the same verb within this batch)\n"
    "- Use different objects and situations\n"
    "- Use different types of constraints\n\n"
    "Before outputting, CHECK each problem:\n"
    "1. Does it have an explicit constraint word? If NO → rewrite it.\n"
    "2. Could someone solve it by just remembering a life-hack? If YES → discard and make a new one.\n"
    "3. Does it involve physical objects and actions? If NO → discard.\n"
    "4. Is the ordinary solution EXACTLY 2-3 words? If NO → fix it.\n\n"
    "{exclusion_text}"
    "Respond with a JSON array of {batch_size} items. Each item:\n"
    "{{\n"
    "  \"problem\": \"To ... (4-7 words after To)\",\n"
    "  \"constraint\": \"the specific constraint word/phrase in the problem\",\n"
    "  \"category\": \"{target_category}\",\n"
    "  \"ordinary_solution\": \"2-3 word common solution\"\n"
    "}}"
)


# ---------------------------------------------------------------------------
# Prompts — Step 2a: creative solutions
# ---------------------------------------------------------------------------
STEP2_CREATIVE_SYSTEM = (
    "You are finding CREATIVE solutions to real-world mini-problems.\n\n"
    "A creative solution is:\n"
    "- NOVEL: unusual, surprising, not the first thing anyone would think of\n"
    "- FUNCTIONAL: it must ACTUALLY WORK to solve the problem in real life\n"
    "- Based on SECONDARY FEATURES of everyday objects\n\n"
    "SECONDARY FEATURES means using an object for a property OTHER than its primary purpose:\n"
    "- A BOOK used for its WEIGHT or FLATNESS (not for reading)\n"
    "- A COIN used for its thin EDGE to act as a screwdriver (not for buying)\n"
    "- A BELT used for its LENGTH to tie something (not for wearing around waist)\n"
    "- A SOCK used for its SOFTNESS to pad/protect (not for wearing on foot)\n\n"
    "CRITICAL — AVOID THIS COMMON MISTAKE:\n"
    "Do NOT give solutions that are simply ANOTHER COMMON solution. "
    "Your answer must be SURPRISING.\n\n"
    "Each creative solution MUST be EXACTLY 2-3 words. A short noun phrase, NOT a sentence.\n\n"
    "Respond with ONLY a valid JSON array."
)

STEP2_CREATIVE_USER = (
    "For each problem below, the ORDINARY (obvious) solution is given. "
    "Find a CREATIVE alternative that ALSO WORKS but is genuinely unexpected.\n\n"
    "Think: what everyday object has a secondary property "
    "(weight, shape, texture, flexibility, stickiness, rigidity, size, material) "
    "that could solve this problem in a surprising way?\n\n"
    "{problems_text}\n\n"
    "Respond with a JSON array. Each item:\n"
    "{{\n"
    "  \"problem\": \"the original problem\",\n"
    "  \"creative_solution\": \"2-3 word novel but working solution\",\n"
    "  \"secondary_feature_used\": \"what property makes it work\"\n"
    "}}"
)


# ---------------------------------------------------------------------------
# Prompts — Step 2b: implausible solutions
# ---------------------------------------------------------------------------
STEP2_IMPLAUSIBLE_SYSTEM = (
    "You are generating IMPLAUSIBLE (nonsensical) solutions to real-world mini-problems.\n\n"
    "An implausible solution is:\n"
    "- NOVEL: unusual, surprising\n"
    "- NON-FUNCTIONAL: it clearly WOULD NOT WORK to solve the problem\n"
    "- Still a real, concrete physical object (not gibberish)\n\n"
    "CRITICAL — THE #1 AI FAILURE MODE:\n"
    "AI models consistently generate \"implausible\" solutions that ACTUALLY COULD WORK.\n\n"
    "THE TEST: For each solution, ask: \"Is there ANY physical mechanism by which "
    "this object could solve the problem?\" If YES → it is NOT implausible.\n\n"
    "GOOD implausible solutions use objects whose physical properties are IRRELEVANT:\n"
    "- Problem: \"To seal an envelope\" → \"cooking oil\" (liquid, makes paper wet)\n"
    "- Problem: \"To remove a jar lid\" → \"washing machine\" (cannot grip/twist a lid)\n"
    "- Problem: \"To keep shoes dry\" → \"a pencil\" (no waterproof/covering properties)\n\n"
    "Each implausible solution MUST be EXACTLY 2-3 words.\n\n"
    "Respond with ONLY a valid JSON array."
)

STEP2_IMPLAUSIBLE_USER = (
    "For each problem below, generate a solution that is a real object "
    "but whose physical properties make it COMPLETELY UNABLE to solve the problem.\n\n"
    "For EACH solution, verify: \"Does this object have ANY property "
    "(shape, material, weight, texture, temperature, size) that could help?\" "
    "If YES → discard and try a completely different object.\n\n"
    "{problems_text}\n\n"
    "Respond with a JSON array. Each item:\n"
    "{{\n"
    "  \"problem\": \"the original problem\",\n"
    "  \"implausible_solution\": \"2-3 word non-working solution\",\n"
    "  \"why_it_fails\": \"one sentence: what property would be needed and why this object lacks it\"\n"
    "}}"
)


# ---------------------------------------------------------------------------
# Prompts — Solve & Evaluate (unchanged from original)
# ---------------------------------------------------------------------------
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
# Formatting helpers
# ---------------------------------------------------------------------------
def format_examples_step1(examples: List[Dict]) -> str:
    """Format examples for Step 1 (problem + ordinary only)."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"{i}. \"{ex['problem']}\"\n"
            f"   Ordinary solution: \"{ex.get('ordinary_solution', '')}\""
        )
    return "\n\n".join(lines)


def format_rejected_text(rejected_examples: List[Dict]) -> str:
    """Format rejected examples for prompt injection."""
    if not rejected_examples:
        return ""
    lines = ["REJECTED examples (do NOT make these mistakes):"]
    for i, ex in enumerate(rejected_examples, 1):
        reason = ex.get("rejection_reason", "bad quality")
        lines.append(f"  {i}. \"{ex['problem']}\" — {reason}")
    return "\n".join(lines) + "\n\n"


def format_problems_for_step2(problems: List[Dict]) -> str:
    """Format problems for Step 2 (creative/implausible generation)."""
    lines = []
    for i, p in enumerate(problems, 1):
        lines.append(
            f"{i}. Problem: \"{p['problem']}\"\n"
            f"   Ordinary solution: \"{p.get('ordinary_solution', '')}\""
        )
    return "\n\n".join(lines)


# =========================================================================
# STEP 1: Generate problems + ordinary solutions (batched, stratified)
# =========================================================================
def step1_generate(
    model: GeminiModel,
    examples: List[Dict],
    rejected: List[Dict],
    cfg: Config,
    dedup: DuplicateTracker,
    balancer: CategoryBalancer,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate problems in batches with:
    - Stratified example sampling (same-cat + cross-cat)
    - Category balancing
    - Deduplication (only among NEW problems, not against approved)
    - Stop-loss: stops when model is stuck in a loop
    - Rejection tracking: saves all rejected problems with reasons

    Returns: (accepted_problems, rejected_problems)
    """
    all_problems = []
    all_rejected = []          # tracking every rejection
    duplicate_count = 0
    failure_count = 0
    api_calls = 0

    # --- Stop-loss state ---
    recent_outcomes: List[bool] = []   # sliding window: True=accepted, False=rejected
    cat_fail_streak: Dict[str, int] = {c: 0 for c in CATEGORIES}
    skipped_cats: set = set()
    stopped_early = False

    # Build a lightweight set of approved problem texts for exact-match rejection.
    approved_texts = {ex["problem"].strip().lower() for ex in examples}

    # NOTE: We do NOT seed the 112 approved examples into the dedup tracker.
    log.info(f"STEP 1: Generating {cfg.n_problems} problems")
    log.info(f"  Injection: {cfg.n_same_cat} same-cat + {cfg.n_cross_cat} cross-cat + "
             f"{cfg.n_rejected} rejected | batch_size={cfg.batch_size}")
    log.info(f"  Threshold: start={dedup.base_threshold}, "
             f"step={dedup.similarity_step}/100, floor={dedup.similarity_floor}")
    log.info(f"  Stop-loss: window={cfg.stoploss_window}, "
             f"min_accepted={cfg.stoploss_min_accepted}, "
             f"cat_streak={cfg.stoploss_cat_streak}")

    while balancer.remaining() > 0:
        # --- Stop-loss check: sliding window ---
        if len(recent_outcomes) >= cfg.stoploss_window:
            recent = recent_outcomes[-cfg.stoploss_window:]
            accepted_in_window = sum(recent)
            if accepted_in_window < cfg.stoploss_min_accepted:
                log.warning(
                    f"  STOP-LOSS: Only {accepted_in_window}/{cfg.stoploss_window} "
                    f"accepted in last window. Stopping generation."
                )
                log.warning(
                    f"  Generated {len(all_problems)}/{cfg.n_problems} problems "
                    f"before stop-loss. {duplicate_count} duplicates, "
                    f"{failure_count} failures, {api_calls} API calls."
                )
                stopped_early = True
                break

        # --- Pick target category (skip stuck ones) ---
        target_cat = balancer.get_target_category()

        # If this category is stuck, try others
        if target_cat in skipped_cats:
            available = [c for c in CATEGORIES
                         if c not in skipped_cats and balancer.counts[c] < cfg.cat_floor]
            if not available:
                # All below-floor categories are stuck — try any category
                available = [c for c in CATEGORIES if c not in skipped_cats]
            if not available:
                log.warning("  STOP-LOSS: All categories stuck. Stopping.")
                stopped_early = True
                break
            target_cat = min(available, key=lambda c: balancer.counts[c])

        current_batch = min(cfg.batch_size, balancer.remaining())

        # Stratified sampling
        selected_examples = sample_examples(
            examples, target_cat, cfg.n_same_cat, cfg.n_cross_cat
        )
        selected_rejected = sample_rejected(rejected, cfg.n_rejected)

        examples_text = format_examples_step1(selected_examples)
        rejected_text = format_rejected_text(selected_rejected)
        exclusion_text = dedup.get_exclusion_text(cfg.max_exclusion_items)
        if exclusion_text:
            exclusion_text += "\n\n"

        user_prompt = STEP1_USER.format(
            examples_text=examples_text,
            rejected_text=rejected_text,
            batch_size=current_batch,
            target_category=target_cat,
            exclusion_text=exclusion_text,
        )

        response = model.generate(STEP1_SYSTEM, user_prompt, schema=GENERATE_STEP1_SCHEMA)
        api_calls += 1
        parsed = parse_json(response)

        if not parsed:
            failure_count += 1
            recent_outcomes.append(False)
            log.warning(f"  Parse failed ({failure_count} total)")
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]

        accepted_this_batch = 0
        for item in parsed:
            # Validate
            issues = validate_problem_step1(item, cfg)
            if issues:
                failure_count += 1
                recent_outcomes.append(False)
                all_rejected.append({
                    "problem": item.get("problem", ""),
                    "category": target_cat,
                    "reason": f"validation: {'; '.join(issues)}",
                    "ordinary_solution": item.get("ordinary_solution", ""),
                })
                log.warning(f"    INVALID: {item.get('problem', '?')} — {issues}")
                continue

            # Dedup: first check against approved (exact match only)
            problem_text = item["problem"].strip()
            if problem_text.lower() in approved_texts:
                duplicate_count += 1
                recent_outcomes.append(False)
                all_rejected.append({
                    "problem": problem_text,
                    "category": target_cat,
                    "reason": "duplicate: exact match with approved example",
                    "ordinary_solution": item.get("ordinary_solution", ""),
                })
                log.info(f"    DUPLICATE (approved): \"{problem_text}\"")
                continue

            # Dedup: then check against other new problems (keyword + semantic)
            dup_check = dedup.is_duplicate(problem_text)
            if dup_check["is_dup"]:
                duplicate_count += 1
                recent_outcomes.append(False)
                all_rejected.append({
                    "problem": problem_text,
                    "category": target_cat,
                    "reason": f"duplicate: {dup_check['reason']}",
                    "most_similar_to": dup_check["most_similar_problem"],
                    "similarity_score": dup_check["similarity_score"],
                    "similarity_type": dup_check["similarity_type"],
                    "threshold_used": dup_check["threshold_used"],
                    "ordinary_solution": item.get("ordinary_solution", ""),
                })
                log.info(f"    DUPLICATE: \"{problem_text}\" — {dup_check['reason']}")
                log.info(f"      ↳ similar to: \"{dup_check['most_similar_problem']}\" "
                         f"(score={dup_check['similarity_score']:.3f})")
                continue

            # Accept
            item["category"] = target_cat
            dedup.add(problem_text)
            all_problems.append(item)
            balancer.record(target_cat)
            accepted_this_batch += 1
            recent_outcomes.append(True)

            # Reset category fail streak on success
            cat_fail_streak[target_cat] = 0

            log.info(f"    [{sum(balancer.counts.values())}/{cfg.n_problems}] "
                     f"[{target_cat}] \"{problem_text}\"")

        # --- Per-category stop-loss ---
        if accepted_this_batch == 0:
            failure_count += 1
            cat_fail_streak[target_cat] += 1
            if cat_fail_streak[target_cat] >= cfg.stoploss_cat_streak:
                skipped_cats.add(target_cat)
                log.warning(
                    f"  CATEGORY SKIP: '{target_cat}' failed {cfg.stoploss_cat_streak}x "
                    f"in a row — skipping temporarily "
                    f"({balancer.counts[target_cat]} generated so far)"
                )
        else:
            # Partial success resets streak
            cat_fail_streak[target_cat] = 0

        time.sleep(0.3)

    # --- Summary ---
    if stopped_early:
        log.warning(f"  Step 1 STOPPED EARLY: {len(all_problems)}/{cfg.n_problems} generated")
    else:
        log.info(f"  Step 1 done: {len(all_problems)} generated")

    log.info(f"  Stats: {duplicate_count} duplicates, {failure_count} failures, "
             f"{api_calls} API calls, {len(all_rejected)} total rejected")
    log.info(f"  {balancer.summary()}")

    if skipped_cats:
        log.warning(f"  Skipped categories: {skipped_cats}")

    stats = dedup.get_stats()
    log.info(f"  Dedup stats: {stats['unique_verbs']} unique verbs, "
             f"threshold={stats['threshold']:.2f}")
    log.info(f"  Top verbs: {stats['top_verbs']}")

    # --- Rejection analysis ---
    if all_rejected:
        reason_counts: Dict[str, int] = {}
        for r in all_rejected:
            # Extract reason type (first word before colon)
            rtype = r["reason"].split(":")[0].strip()
            reason_counts[rtype] = reason_counts.get(rtype, 0) + 1
        log.info(f"  Rejection breakdown:")
        for rtype, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {rtype}: {count}")

    return all_problems, all_rejected


# =========================================================================
# STEP 2a: Creative solutions (batched)
# =========================================================================
def step2_creative(
    model: GeminiModel,
    problems: List[Dict],
    cfg: Config,
) -> List[Dict]:
    """Generate creative solutions in batches."""
    log.info(f"STEP 2a: Generating creative solutions for {len(problems)} problems")

    batch_size = cfg.batch_size
    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        problems_text = format_problems_for_step2(batch)

        response = model.generate(
            STEP2_CREATIVE_SYSTEM,
            STEP2_CREATIVE_USER.format(problems_text=problems_text),
            schema=GENERATE_CREATIVE_SCHEMA,
        )
        parsed = parse_json(response)

        if not parsed or not isinstance(parsed, list):
            log.warning(f"  Failed to parse creative batch at {start}")
            for item in batch:
                item["creative_solution"] = ""
                item["secondary_feature_used"] = ""
            continue

        for i, item in enumerate(batch):
            if i < len(parsed):
                sol = parsed[i].get("creative_solution", "").strip()
                feature = parsed[i].get("secondary_feature_used", "").strip()
                item["creative_solution"] = sol
                item["secondary_feature_used"] = feature
                valid = validate_solution(sol, cfg)
                if not valid:
                    log.warning(f"    Creative WC issue: '{sol}' for '{item['problem']}'")
            else:
                item["creative_solution"] = ""
                item["secondary_feature_used"] = ""

        if (start + batch_size) % 50 < batch_size:
            log.info(f"  Creative: {min(start + batch_size, len(problems))}/{len(problems)}")
        time.sleep(0.3)

    return problems


# =========================================================================
# STEP 2b: Implausible solutions (batched)
# =========================================================================
def step2_implausible(
    model: GeminiModel,
    problems: List[Dict],
    cfg: Config,
) -> List[Dict]:
    """Generate implausible solutions in batches."""
    log.info(f"STEP 2b: Generating implausible solutions for {len(problems)} problems")

    batch_size = cfg.batch_size
    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        problems_text = format_problems_for_step2(batch)

        response = model.generate(
            STEP2_IMPLAUSIBLE_SYSTEM,
            STEP2_IMPLAUSIBLE_USER.format(problems_text=problems_text),
            schema=GENERATE_IMPLAUSIBLE_SCHEMA,
        )
        parsed = parse_json(response)

        if not parsed or not isinstance(parsed, list):
            log.warning(f"  Failed to parse implausible batch at {start}")
            for item in batch:
                item["implausible_solution"] = ""
                item["why_it_fails"] = ""
            continue

        for i, item in enumerate(batch):
            if i < len(parsed):
                sol = parsed[i].get("implausible_solution", "").strip()
                why = parsed[i].get("why_it_fails", "").strip()
                item["implausible_solution"] = sol
                item["why_it_fails"] = why
                valid = validate_solution(sol, cfg)
                if not valid:
                    log.warning(f"    Implausible WC issue: '{sol}' for '{item['problem']}'")
            else:
                item["implausible_solution"] = ""
                item["why_it_fails"] = ""

        if (start + batch_size) % 50 < batch_size:
            log.info(f"  Implausible: {min(start + batch_size, len(problems))}/{len(problems)}")
        time.sleep(0.3)

    return problems


# =========================================================================
# STEP 3: Solve (cross-validation)
# =========================================================================
def step_solve(model: GeminiModel, problems: List[Dict], cfg: Config) -> List[Dict]:
    log.info(f"SOLVE: {len(problems)} problems")
    for i, item in enumerate(problems):
        response = model.generate(
            SOLVE_SYSTEM,
            SOLVE_USER.format(problem=item["problem"]),
            schema=SOLVE_SCHEMA,
        )
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
    log.info("  Done")
    return problems


# =========================================================================
# STEP 4: Evaluate
# =========================================================================
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
    log.info("  Done")
    return problems


# =========================================================================
# Save CSV
# =========================================================================
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


# =========================================================================
# Main pipeline
# =========================================================================
def run_pipeline(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    with open(cfg.examples_path) as f:
        examples = json.load(f)
    log.info(f"Loaded {len(examples)} examples")

    rejected = []
    if os.path.exists(cfg.rejected_path):
        with open(cfg.rejected_path) as f:
            rejected = json.load(f)
        log.info(f"Loaded {len(rejected)} rejected")

    model = GeminiModel(cfg)
    dedup = DuplicateTracker(
        cfg.similarity_threshold, cfg.similarity_floor,
        cfg.similarity_step, cfg.embed_model,
    )

    # Category balancer
    balancer = CategoryBalancer(CATEGORIES, cfg.cat_floor, cfg.n_problems)

    # If including approved, count them toward balancer
    approved = []
    if cfg.include_approved:
        approved = [dict(ex) for ex in examples if ex.get("ordinary_solution")]
        balancer.seed(approved)
        remaining = balancer.remaining()
        log.info(f"Including {len(approved)} approved, generating {remaining} new")
        log.info(f"  {balancer.summary()}")

    # --- STEP 1: Generate problems + ordinary ---
    new_problems, gen_rejected = step1_generate(
        model, examples, rejected, cfg, dedup, balancer
    )

    # Save rejected problems for analysis
    if cfg.save_rejected and gen_rejected:
        rej_cols = [
            "problem", "category", "reason",
            "most_similar_to", "similarity_score", "similarity_type",
            "threshold_used", "ordinary_solution",
        ]
        rej_rows = [{c: r.get(c, "") for c in rej_cols} for r in gen_rejected]
        pd.DataFrame(rej_rows, columns=rej_cols).to_csv(
            cfg.rejected_output_path, index=False, quoting=csv.QUOTE_ALL
        )
        log.info(f"Saved {len(gen_rejected)} rejected problems to {cfg.rejected_output_path}")

    # --- STEP 2a: Creative solutions ---
    new_problems = step2_creative(model, new_problems, cfg)

    # --- STEP 2b: Implausible solutions ---
    new_problems = step2_implausible(model, new_problems, cfg)

    # Combine
    all_problems = approved + new_problems

    # --- STEP 3: Solve (cross-validation) ---
    if cfg.run_solve:
        all_problems = step_solve(model, all_problems, cfg)

    # --- STEP 4: Evaluate ---
    if cfg.run_evaluate:
        all_problems = step_evaluate(model, all_problems, cfg)

    # Save
    save_csv(all_problems, cfg.output_path)

    # Summary
    valid = sum(1 for p in all_problems if not p.get("quality_flags"))
    log.info(f"DONE: {len(all_problems)} total, {valid} without quality flags")
    log.info(f"  {balancer.summary()}")


# =========================================================================
# CLI — all knobs exposed
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Mini-Problems Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example configurations to test:

  # Default: 5 examples (2 same-cat + 3 cross), batch of 5
  python pipeline.py --n-problems 20

  # More same-category anchoring
  python pipeline.py --n-problems 20 --n-same-cat 3 --n-cross-cat 2

  # All cross-category (no anchoring)
  python pipeline.py --n-problems 20 --n-same-cat 0 --n-cross-cat 5

  # Larger batches, fewer examples
  python pipeline.py --n-problems 50 --batch-size 10 --n-same-cat 1 --n-cross-cat 2

  # No rejected examples
  python pipeline.py --n-problems 50 --n-rejected 0

  # Full run: 500 problems, floor 35
  python pipeline.py --n-problems 500 --cat-floor 35 --batch-size 10

  # Skip solve & evaluate (generation only)
  python pipeline.py --n-problems 50 --no-solve --no-evaluate
        """,
    )
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""))
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20")
    parser.add_argument("--n-problems", type=int, default=500)
    parser.add_argument("--output", default="generated_problems.csv")
    parser.add_argument("--seed", type=int, default=42)

    # Injection control
    parser.add_argument("--n-same-cat", type=int, default=2,
                        help="Examples from target category per call (default: 2)")
    parser.add_argument("--n-cross-cat", type=int, default=3,
                        help="Examples from other categories per call (default: 3)")
    parser.add_argument("--n-rejected", type=int, default=2,
                        help="Rejected examples per call (default: 2)")

    # Batching
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Problems generated per API call (default: 5)")

    # Category balancing
    parser.add_argument("--cat-floor", type=int, default=35,
                        help="Minimum problems per category (default: 35)")

    # Diversity
    parser.add_argument("--similarity-threshold", type=float, default=0.75,
                        help="Starting semantic similarity threshold (default: 0.75)")
    parser.add_argument("--similarity-floor", type=float, default=0.40,
                        help="Lowest the threshold can drop to (default: 0.40)")
    parser.add_argument("--similarity-step", type=float, default=0.05,
                        help="Threshold drop per 100 problems (default: 0.05)")
    parser.add_argument("--temperature", type=float, default=0.8)

    # Stop-loss
    parser.add_argument("--stoploss-window", type=int, default=20,
                        help="Sliding window size for stop-loss (default: 20)")
    parser.add_argument("--stoploss-min-accepted", type=int, default=3,
                        help="Min accepted in window before stopping (default: 3)")
    parser.add_argument("--stoploss-cat-streak", type=int, default=5,
                        help="Consecutive failures to skip a category (default: 5)")

    # Include/exclude
    parser.add_argument("--no-approved", action="store_true",
                        help="Don't include 112 approved in output")

    # Pipeline steps
    parser.add_argument("--no-solve", action="store_true",
                        help="Skip the solve step")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="Skip the evaluate step")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.api_key:
        parser.error("Set GEMINI_API_KEY or use --api-key")

    cfg = Config(
        api_key=args.api_key,
        model_name=args.model,
        n_problems=args.n_problems,
        output_path=args.output,
        seed=args.seed,
        n_same_cat=args.n_same_cat,
        n_cross_cat=args.n_cross_cat,
        n_rejected=args.n_rejected,
        batch_size=args.batch_size,
        cat_floor=args.cat_floor,
        similarity_threshold=args.similarity_threshold,
        similarity_floor=args.similarity_floor,
        similarity_step=args.similarity_step,
        temperature=args.temperature,
        stoploss_window=args.stoploss_window,
        stoploss_min_accepted=args.stoploss_min_accepted,
        stoploss_cat_streak=args.stoploss_cat_streak,
        include_approved=not args.no_approved,
        run_solve=not args.no_solve,
        run_evaluate=not args.no_evaluate,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
