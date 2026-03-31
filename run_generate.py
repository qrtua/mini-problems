"""
Module 1: Mini-Problem Generator (2-Step) with Diversity Control
=================================================================
Step 1: Generate problems + ordinary solutions (with dedup + similarity filter)
Step 2a: Generate creative solutions (separately, with feature-mapping guidance)
Step 2b: Generate implausible solutions (separately, with anti-plausibility check)

Anti-redundancy mechanisms:
- Deduplication buffer: injects "already used" verbs/objects/problems into prompt
- Semantic similarity filter: rejects problems with cosine sim > threshold

Usage:
    python run_generate.py
    python run_generate.py model=qwen2.5-7b n_problems=50
    python run_generate.py diversity.similarity_threshold=0.65  # stricter
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
    parse_json_response, validate_mini_problem, count_words,
    DuplicateTracker
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
    """Validate problem format. Word count excludes leading 'To'."""
    issues = []
    problem = item.get("problem", "").strip()

    if not problem.lower().startswith("to "):
        issues.append("Problem must start with 'To'")

    core = problem[3:].strip() if problem.lower().startswith("to ") else problem
    wc = count_words(core)
    if wc < cfg.validation.problem_min_words or wc > cfg.validation.problem_max_words:
        issues.append(f"Problem core has {wc} words (need {cfg.validation.problem_min_words}-{cfg.validation.problem_max_words})")

    ord_sol = item.get("ordinary_solution", "").strip()
    ord_wc = count_words(ord_sol)
    if ord_wc < cfg.validation.solution_min_words or ord_wc > cfg.validation.solution_max_words:
        issues.append(f"Ordinary solution has {ord_wc} words (need {cfg.validation.solution_min_words}-{cfg.validation.solution_max_words})")

    if not ord_sol:
        issues.append("Ordinary solution is empty")

    return {"valid": len(issues) == 0, "issues": issues, "word_count": wc}


# =========================================================================
# STEP 1: Generate problems + ordinary solutions (with diversity control)
# =========================================================================

def step1_generate_problems(
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
    examples: List[Dict],
    cfg: DictConfig,
    dedup: DuplicateTracker,
    rejected: List[Dict] = None,
) -> List[Dict]:
    """
    Generate problems with ordinary solutions in batches.
    Uses DuplicateTracker to inject exclusion history into prompts
    and filter out redundant problems post-generation.
    """
    all_problems = []
    valid_count = 0
    duplicate_count = 0
    total_needed = cfg.n_problems
    batch_size = cfg.batch_size
    attempt = 0

    # Seed the tracker with existing examples so we don't regenerate them
    for ex in examples:
        dedup.add(ex["problem"])
    log.info(f"Seeded dedup tracker with {len(examples)} existing examples")

    log.info(f"STEP 1: Generating {total_needed} problems + ordinary solutions")
    log.info(f"  Diversity: similarity_threshold={dedup.similarity_threshold}")

    while valid_count < total_needed and attempt < total_needed * 5:
        attempt += 1
        remaining = total_needed - valid_count
        current_batch = min(batch_size, remaining)

        log.info(f"  Batch {attempt}: requesting {current_batch} "
                 f"({valid_count}/{total_needed} valid, {duplicate_count} duplicates filtered)")

        # Build prompt WITH exclusion history
        examples_text = format_examples_step1(examples, n=cfg.n_examples)
        categories_str = ", ".join(CATEGORIES)
        exclusion_text = dedup.get_exclusion_summary(max_items=cfg.diversity.max_exclusion_items)

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

        # Inject exclusion history into the last user message
        if exclusion_text:
            conversation[-1]["content"] += f"\n\n{exclusion_text}"

        response = model.generate_single(conversation)
        parsed = parse_json_response(response)

        if parsed is None:
            log.warning("  Failed to parse response, retrying...")
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]

        # Validate + deduplicate each problem
        for item in parsed:
            # Step A: structural validation
            validation = validate_problem_format(item, cfg)
            if not validation["valid"]:
                log.warning(f"    INVALID: {item.get('problem', '?')} — {validation['issues']}")
                continue

            # Step B: duplicate check
            problem_text = item.get("problem", "").strip()
            dup_check = dedup.check_duplicate(problem_text)

            if dup_check["is_duplicate"]:
                duplicate_count += 1
                log.warning(f"    DUPLICATE: \"{problem_text}\" — {dup_check['reason']}")
                continue

            # Soft warning (not a rejection, just a log)
            if dup_check["reason"]:
                log.info(f"    NOTE: \"{problem_text}\" — {dup_check['reason']}")

            # Passed both checks — accept
            item["_step1_valid"] = True
            item["_keyword_overlap"] = dup_check["keyword_overlap"]
            item["_semantic_similarity"] = dup_check["semantic_similarity"]
            all_problems.append(item)
            dedup.add(problem_text)
            valid_count += 1
            log.info(f"    ACCEPTED: \"{problem_text}\" "
                     f"(kw_overlap={dup_check['keyword_overlap']:.2f}, "
                     f"sem_sim={dup_check['semantic_similarity']:.3f})")

    # Log diversity stats
    stats = dedup.get_stats()
    log.info(f"  Step 1 complete: {valid_count} valid, {duplicate_count} duplicates filtered")
    log.info(f"  Diversity stats: {stats['unique_verbs']} unique verbs, "
             f"{stats['unique_objects']} unique objects")
    log.info(f"  Top verbs: {stats['top_verbs']}")
    log.info(f"  Top objects: {stats['top_objects']}")

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

        for i, item in enumerate(batch):
            if i < len(parsed):
                sol = parsed[i].get("creative_solution", "").strip()
                feature = parsed[i].get("secondary_feature_used", "").strip()

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
    """Main entry point — runs the 2-step generation pipeline with diversity control."""
    log.info("=" * 60)
    log.info("MINI-PROBLEMS GENERATOR (2-STEP + DIVERSITY)")
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

    # --- Init diversity tracker ---
    dedup = DuplicateTracker(
        similarity_threshold=cfg.diversity.similarity_threshold,
        embed_model=cfg.diversity.embed_model,
    )

    # --- Init model & prompts ---
    model = ModelWrapper(cfg)
    prompt_builder = PromptBuilder(cfg.prompt_config_path, cfg.model.name)

    # --- STEP 1: Problems + Ordinary (with dedup) ---
    problems = step1_generate_problems(
        model, prompt_builder, examples, cfg, dedup, rejected
    )

    # --- STEP 2a: Creative solutions ---
    problems = step2_creative(model, prompt_builder, problems, cfg)

    # --- STEP 2b: Implausible solutions ---
    problems = step2_implausible(model, prompt_builder, problems, cfg)

    # --- Final validation ---
    for item in problems:
        val = validate_mini_problem(item, cfg)
        item["_fully_valid"] = val["valid"]

    # --- Save & Log ---
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
            "keyword_overlap": item.get("_keyword_overlap", 0.0),
            "semantic_similarity": item.get("_semantic_similarity", -1.0),
        })

    logger = ResultLogger(cfg)
    output_path = logger.save_to_csv(output_records)

    dedup_stats = dedup.get_stats()
    stats = {
        "total_generated": len(output_records),
        "fully_valid": sum(1 for r in output_records if r["fully_valid"]),
        "validity_rate": sum(1 for r in output_records if r["fully_valid"]) / max(len(output_records), 1),
        "missing_creative": sum(1 for r in output_records if not r["creative_solution"]),
        "missing_implausible": sum(1 for r in output_records if not r["implausible_solution"]),
        "unique_verbs": dedup_stats["unique_verbs"],
        "unique_objects": dedup_stats["unique_objects"],
        "avg_keyword_overlap": round(
            sum(r["keyword_overlap"] for r in output_records) / max(len(output_records), 1), 3
        ),
        "avg_semantic_similarity": round(
            sum(r["semantic_similarity"] for r in output_records if r["semantic_similarity"] >= 0)
            / max(sum(1 for r in output_records if r["semantic_similarity"] >= 0), 1), 3
        ),
    }
    log.info(f"Summary: {stats}")
    logger.log_to_wandb(stats, output_records, output_path, cfg)

    log.info("Generator finished successfully.")


if __name__ == "__main__":
    main()
