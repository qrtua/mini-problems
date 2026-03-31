"""
Module 3: Mini-Problem Evaluator
==================================
Evaluates problem-solution pairs for feasibility and novelty.
Can be used to:
  - Validate generated problems (are creative solutions actually feasible?)
  - Compare solutions from different models
  - Classify unknown solutions as ordinary/creative/implausible

Usage:
    python run_evaluate.py
    python run_evaluate.py model=llama3-70b eval_mode=classify
"""
import json
import logging

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import List, Dict

from utils import (
    set_all_seeds, ModelWrapper, PromptBuilder, DataHandler, ResultLogger,
    parse_json_response
)

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


def evaluate_batch(
    problems: List[Dict],
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
    cfg: DictConfig,
) -> List[Dict]:
    """
    Evaluate all three solutions for each problem in batch mode.
    Returns feasibility and novelty ratings.
    """
    results = []

    for item in tqdm(problems, desc="Evaluating problems"):
        problem = item.get("problem", "")
        if not problem:
            continue

        result = {
            "problem": problem,
            "category": item.get("category", ""),
        }

        # Determine which solution columns to use
        # (supports both original and solved columns)
        ordinary = item.get("solved_ordinary", item.get("ordinary_solution", ""))
        creative = item.get("solved_creative", item.get("creative_solution", ""))
        implausible = item.get("solved_implausible", item.get("implausible_solution", ""))

        result["ordinary_solution"] = ordinary
        result["creative_solution"] = creative
        result["implausible_solution"] = implausible

        conv = prompt_builder.build_conversation(
            "evaluate_batch",
            problem=problem,
            ordinary_solution=ordinary,
            creative_solution=creative,
            implausible_solution=implausible,
        )
        response = model.generate_single(conv)
        parsed = parse_json_response(response)

        if parsed:
            for sol_type in ["ordinary", "creative", "implausible"]:
                ratings = parsed.get(sol_type, {})
                result[f"{sol_type}_feasibility"] = ratings.get("feasibility", None)
                result[f"{sol_type}_novelty"] = ratings.get("novelty", None)
                result[f"{sol_type}_reasoning"] = ratings.get("reasoning", "")
        else:
            result["_parse_error"] = True

        results.append(result)

    return results


def evaluate_feasibility(
    problems: List[Dict],
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
) -> List[Dict]:
    """Evaluate individual problem-solution pairs."""
    results = []

    for item in tqdm(problems, desc="Evaluating feasibility"):
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        conv = prompt_builder.build_conversation(
            "evaluate_feasibility",
            problem=problem,
            solution=solution,
        )
        response = model.generate_single(conv)
        parsed = parse_json_response(response)

        result = {"problem": problem, "solution": solution}
        if parsed:
            result["feasibility"] = parsed.get("feasibility", None)
            result["novelty"] = parsed.get("novelty", None)
            result["reasoning"] = parsed.get("reasoning", "")
        else:
            result["_parse_error"] = True

        results.append(result)

    return results


def classify_solutions(
    problems: List[Dict],
    model: ModelWrapper,
    prompt_builder: PromptBuilder,
) -> List[Dict]:
    """Classify solutions as ordinary/creative/implausible."""
    results = []

    for item in tqdm(problems, desc="Classifying solutions"):
        problem = item.get("problem", "")
        solution = item.get("solution", "")

        conv = prompt_builder.build_conversation(
            "classify_solution_type",
            problem=problem,
            solution=solution,
        )
        response = model.generate_single(conv)
        parsed = parse_json_response(response)

        result = {"problem": problem, "solution": solution}
        if parsed:
            result["classification"] = parsed.get("classification", "")
            result["reasoning"] = parsed.get("reasoning", "")
        else:
            result["_parse_error"] = True

        results.append(result)

    return results


def calculate_eval_stats(results: List[Dict]) -> Dict:
    """Calculate summary statistics from evaluation results."""
    stats = {}
    df = pd.DataFrame(results)

    for sol_type in ["ordinary", "creative", "implausible"]:
        feas_col = f"{sol_type}_feasibility"
        nov_col = f"{sol_type}_novelty"

        if feas_col in df.columns:
            valid = df[feas_col].dropna()
            if len(valid) > 0:
                stats[f"{sol_type}_mean_feasibility"] = round(valid.mean(), 3)
                stats[f"{sol_type}_mean_novelty"] = round(df[nov_col].dropna().mean(), 3)

    # Quality check: how many problems have the expected pattern?
    # (ordinary=high feas/low nov, creative=high feas/high nov, implausible=low feas/high nov)
    if "ordinary_feasibility" in df.columns:
        well_formed = (
            (df["ordinary_feasibility"] >= 4) &
            (df["ordinary_novelty"] <= 2) &
            (df["creative_feasibility"] >= 3) &
            (df["creative_novelty"] >= 3) &
            (df["implausible_feasibility"] <= 2) &
            (df["implausible_novelty"] >= 3)
        )
        stats["well_formed_rate"] = round(well_formed.mean(), 3)

    stats["parse_error_rate"] = round(
        sum(1 for r in results if r.get("_parse_error")) / max(len(results), 1), 3
    )

    log.info(f"Evaluation stats: {stats}")
    return stats


@hydra.main(version_base=None, config_path="configs", config_name="config-evaluate")
def main(cfg: DictConfig) -> None:
    """Main entry point for the Evaluator module."""
    log.info("=" * 60)
    log.info("MINI-PROBLEMS EVALUATOR")
    log.info("=" * 60)
    log.info("Configuration:\n" + OmegaConf.to_yaml(cfg))

    set_all_seeds(cfg.seed)

    # --- Load data ---
    data_handler = DataHandler(cfg)
    problems = data_handler.load_problems()
    log.info(f"Loaded {len(problems)} problems to evaluate")

    # --- Init model & prompts ---
    model = ModelWrapper(cfg)
    prompt_builder = PromptBuilder(cfg.prompt_config_path, cfg.model.name)

    # --- Evaluate ---
    if cfg.eval_mode == "batch":
        results = evaluate_batch(problems, model, prompt_builder, cfg)
    elif cfg.eval_mode == "feasibility":
        results = evaluate_feasibility(problems, model, prompt_builder)
    elif cfg.eval_mode == "classify":
        results = classify_solutions(problems, model, prompt_builder)
    else:
        raise ValueError(f"Unknown eval_mode: {cfg.eval_mode}")

    # --- Stats, Save & Log ---
    stats = calculate_eval_stats(results)
    logger = ResultLogger(cfg)
    output_path = logger.save_to_csv(results)
    logger.log_to_wandb(stats, results, output_path, cfg)

    log.info("Evaluator finished successfully.")


if __name__ == "__main__":
    main()
