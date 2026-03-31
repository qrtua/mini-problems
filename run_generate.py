"""
Module 1: Mini-Problem Generator (2-Step)
==========================================
Step 1: Generate problems + ordinary solutions
Step 2a: Generate creative solutions (separately, with feature-mapping guidance)
Step 2b: Generate implausible solutions (separately, with anti-plausibility check)

This separation improves quality — creative and implausible solutions don't
contaminate each other when generated independently.

Usage:
    python run_generate.py
    python run_generate.py model=llama3-70b n_problems=50
"""
import os
import json
import random
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict

from utils import (
    set_all_seeds, ModelWrapper, PromptBuilder, DataHandler, ResultLogger,
    parse_json_response, validate_mini_problem, count_words
)

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log = logging.getLogger(__name__)

CATEGORIES = [
    "visuo-spatial", "material-substitution", "mechanical-advantage",
    "containment", "attachment", "support-stabilization",
    "protection-barrier", "tool-repurposing"
]


def format_examples_step1(examples: List[Dict], n: int = 5) -> str:
    """Format examples for Step 1 (problem + ordinary only). Uses random dropout."""
    selected = random.sample(examples, min(n, len(examples)))
    lines = []
    for i, ex in enumerate(selected, 1):
        lines.append(
            f"{i}. Problem: \"{ex['problem']}\"\n"
            f"   Ordinary solution: \"{ex['ordinary_solution']}\""
        )
    return "\n\n".join(lines)


def format_problems_for_step2(problems: List[Dict]) -> str:
    """Format problems for Step 2 (creative/implausible generation)."""
    lines = []
    for i, p in enumerate(problems, 1):
        lines.append(
            f"{i}. Problem: \"{p['problem']}\"\n"
            f"   Ordinary solution: \"{p['ordinary_solution']}\""
        )
    return "\n\n".join(lines)


def format_rejected(rejected: List[Dict], n: int = 5) -> str:
    """Format rejected examples with reasons for anti-overfitting."""
    selected = random.sample(rejected, min(n, len(rejected)))
    lines = []
    for i, ex in enumerate(selected, 1):
        reason = ex.get("rejection_reason", "does not meet requirements")
        ai_impl = ex.get("ai_implausible", "")
        lines.append(
            f"{i}. REJECTED: \"{ex.get('problem', '?')}\"\n"
            f"   Ordinary: \"{ex.get('ordinary_solution', '?')}\"\n"
            f"   Creative: \"{ex.get('creative_solution', '?')}\"\n"
            f"   AI tried implausible: \"{ai_impl}\" (but it was actually plausible!)\n"
            f"   Why rejected: {reason}"
        )
    return "\n\n".join(lines)


def validate_problem_format(item: Dict, cfg: DictConfig) -> Dict:
    """
    Validate problem format. Word count excludes leading 'To'.
    """
    issues = []
    problem = item.get("problem", "").strip()

    # Check "To" prefix
    if not problem.lower().startswith("to "):
        issues.append("Problem must start with 'To'")

    # Count words AFTER "To"
    core = problem[3:].strip() if problem.lower().startswith("to ") else problem
    wc = count_words(core)
    if wc < cfg.validation.problem_min_words or wc > cfg.validation.problem_max_words:
        issues.append(f"Problem core has {wc} words (need {cfg.validation.problem_min_words}-{cfg.validation.problem_max_words})")

    # Check ordinary solution
    ord_sol = item.get("ordinary_solution", "").strip()
    ord_wc = count_words(ord_sol)
    if ord_wc < cfg.validation.solution_min_words or ord_wc > cfg.validation.solution_max_words:
        issues.append(f"Ordinary solution has {ord_wc} words (need {cfg.validation.solution_min_words}-{cfg.validation.solution_max_words})")

    if not ord_sol:
        issues.append("Ordinary solution is empty")

    return {"valid": len(issues) == 0, "issues": issues, "word_count": wc}


# =========================================================================
# STEP 1: Generate problems + ordinary solutions
# =========================================================================

def step1_generate_problems(
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
    examples: List[Dict],
    cfg: DictConfig,
    rejected: List[Dict] = None,
) -> List[Dict]:
    """Generate problems with ordinary solutions in batches."""
    all_problems = []
    valid_count = 0
    total_needed = cfg.n_problems
    batch_size = cfg.batch_size
    attempt = 0

    log.info(f"STEP 1: Generating {total_needed} problems + ordinary solutions")

    while valid_count < total_needed and attempt < total_needed * 3:
        attempt += 1
        remaining = total_needed - valid_count
        current_batch = min(batch_size, remaining)

        log.info(f"  Batch {attempt}: requesting {current_batch} "
                 f"({valid_count}/{total_needed} valid)")

        # Build prompt
        examples_text = format_examples_step1(examples, n=cfg.n_examples)
        categories_str = ", ".join(CATEGORIES)

        if rejected and cfg.use_rejected:
            rejected_text = format_rejected(rejected)
            conversation = prompt_builder.build_conversation(
                "generate_with_rejected_step1",
                examples_text=examples_text,
                rejected_text=rejected_text,
                n_generate=current_batch,
                categories=categories_str,
            )
        else:
            conversation = prompt_builder.build_conversation(
                "generate_step1",
                n_examples=min(cfg.n_examples, len(examples)),
                examples_text=examples_text,
                n_generate=current_batch,
                categories=categories_str,
            )

        response = model.generate_single(conversation)
        parsed = parse_json_response(response)

        if parsed is None:
            log.warning("  Failed to parse response, retrying...")
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]

        # Validate each problem
        for item in parsed:
            validation = validate_problem_format(item, cfg)
            if validation["valid"]:
                item["_step1_valid"] = True
                all_problems.append(item)
                valid_count += 1
                log.info(f"    VALID: {item.get('problem', '?')}")
            else:
                log.warning(f"    INVALID: {item.get('problem', '?')} — {validation['issues']}")

    log.info(f"  Step 1 complete: {valid_count} valid problems")
    return all_problems


# =========================================================================
# STEP 2a: Generate creative solutions
# =========================================================================

def step2_creative(
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
    problems: List[Dict],
    cfg: DictConfig,
) -> List[Dict]:
    """Generate creative solutions for existing problems."""
    log.info(f"STEP 2a: Generating creative solutions for {len(problems)} problems")

    # Process in batches
    batch_size = cfg.batch_size
    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        problems_text = format_problems_for_step2(batch)

        conversation = prompt_builder.build_conversation(
            "generate_step2_creative",
            problems_text=problems_text,
        )
        response = model.generate_single(conversation)
        parsed = parse_json_response(response)

        if parsed is None:
            log.warning(f"  Failed to parse creative batch starting at {start}")
            for item in batch:
                item["creative_solution"] = ""
                item["secondary_feature_used"] = ""
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]

        # Match responses back to problems
        for i, item in enumerate(batch):
            if i < len(parsed):
                sol = parsed[i].get("creative_solution", "").strip()
                feature = parsed[i].get("secondary_feature_used", "").strip()

                # Validate word count
                wc = count_words(sol)
                if cfg.validation.solution_min_words <= wc <= cfg.validation.solution_max_words:
                    item["creative_solution"] = sol
                    item["secondary_feature_used"] = feature
                    log.info(f"    Creative for '{item['problem']}': {sol} (feature: {feature})")
                else:
                    item["creative_solution"] = sol
                    item["_creative_wc_issue"] = f"{wc} words"
                    log.warning(f"    Creative WC issue: '{sol}' has {wc} words")
            else:
                item["creative_solution"] = ""

    return problems


# =========================================================================
# STEP 2b: Generate implausible solutions
# =========================================================================

def step2_implausible(
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
    problems: List[Dict],
    cfg: DictConfig,
) -> List[Dict]:
    """Generate implausible solutions for existing problems."""
    log.info(f"STEP 2b: Generating implausible solutions for {len(problems)} problems")

    batch_size = cfg.batch_size
    for start in range(0, len(problems), batch_size):
        batch = problems[start:start + batch_size]
        problems_text = format_problems_for_step2(batch)

        conversation = prompt_builder.build_conversation(
            "generate_step2_implausible",
            problems_text=problems_text,
        )
        response = model.generate_single(conversation)
        parsed = parse_json_response(response)

        if parsed is None:
            log.warning(f"  Failed to parse implausible batch starting at {start}")
            for item in batch:
                item["implausible_solution"] = ""
                item["why_it_fails"] = ""
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]

        for i, item in enumerate(batch):
            if i < len(parsed):
                sol = parsed[i].get("implausible_solution", "").strip()
                why = parsed[i].get("why_it_fails", "").strip()

                wc = count_words(sol)
                if cfg.validation.solution_min_words <= wc <= cfg.validation.solution_max_words:
                    item["implausible_solution"] = sol
                    item["why_it_fails"] = why
                    log.info(f"    Implausible for '{item['problem']}': {sol} (reason: {why})")
                else:
                    item["implausible_solution"] = sol
                    item["_implausible_wc_issue"] = f"{wc} words"
                    log.warning(f"    Implausible WC issue: '{sol}' has {wc} words")
            else:
                item["implausible_solution"] = ""

    return problems


# =========================================================================
# Main
# =========================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config-generate")
def main(cfg: DictConfig) -> None:
    """Main entry point — runs the 2-step generation pipeline."""
    log.info("=" * 60)
    log.info("MINI-PROBLEMS GENERATOR (2-STEP)")
    log.info("=" * 60)
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    set_all_seeds(cfg.seed)

    # --- Load data ---
    data_handler = DataHandler(cfg)
    examples = data_handler.load_examples()

    rejected = None
    if cfg.use_rejected and os.path.exists(cfg.data.rejected_path):
        with open(cfg.data.rejected_path, 'r') as f:
            rejected = json.load(f)
        log.info(f"Loaded {len(rejected)} rejected examples")

    # --- Init model & prompts ---
    model = ModelWrapper(cfg)
    prompt_builder = PromptBuilder(cfg.prompt_config_path, cfg.model.name)

    # --- STEP 1: Problems + Ordinary ---
    problems = step1_generate_problems(model, prompt_builder, examples, cfg, rejected)

    # --- STEP 2a: Creative solutions ---
    problems = step2_creative(model, prompt_builder, problems, cfg)

    # --- STEP 2b: Implausible solutions ---
    problems = step2_implausible(model, prompt_builder, problems, cfg)

    # --- Final validation ---
    for item in problems:
        val = validate_mini_problem(item, cfg)
        item["_fully_valid"] = val["valid"]

    # --- Save & Log ---
    # Clean internal fields for output
    output_records = []
    for item in problems:
        output_records.append({
            "problem": item.get("problem", ""),
            "category": item.get("category", ""),
            "constraint": item.get("constraint", ""),
            "ordinary_solution": item.get("ordinary_solution", ""),
            "creative_solution": item.get("creative_solution", ""),
            "secondary_feature_used": item.get("secondary_feature_used", ""),
            "implausible_solution": item.get("implausible_solution", ""),
            "why_it_fails": item.get("why_it_fails", ""),
            "fully_valid": item.get("_fully_valid", False),
        })

    logger = ResultLogger(cfg)
    output_path = logger.save_to_csv(output_records)

    stats = {
        "total_generated": len(output_records),
        "fully_valid": sum(1 for r in output_records if r["fully_valid"]),
        "validity_rate": sum(1 for r in output_records if r["fully_valid"]) / max(len(output_records), 1),
        "missing_creative": sum(1 for r in output_records if not r["creative_solution"]),
        "missing_implausible": sum(1 for r in output_records if not r["implausible_solution"]),
    }
    log.info(f"Summary: {stats}")
    logger.log_to_wandb(stats, output_records, output_path, cfg)

    log.info("Generator finished successfully.")


if __name__ == "__main__":
    main()
