"""
Shared utilities for the Mini-Problems pipeline.
Mirrors the structure of run_reg.py: seed setting, model init, result logging.
"""
import os
import csv
import json
import random
import logging
import re

import numpy as np
import torch
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed & reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    log.info(f"All random seeds set to: {seed}")


# ---------------------------------------------------------------------------
# Validation helpers (Chain of Verification from Instructions)
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.strip().split())


def validate_problem(problem: str, min_words: int = 4, max_words: int = 9) -> Dict:
    """
    Validate a mini-problem against structural requirements.
    Word count EXCLUDES the leading "To" (per project convention).
    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []
    if not problem.strip():
        issues.append("Problem is empty")
        return {"valid": False, "word_count": 0, "issues": issues}

    # Strip leading "To " for word count
    core = problem.strip()
    if core.lower().startswith("to "):
        core = core[3:].strip()

    wc = count_words(core)
    if wc < min_words or wc > max_words:
        issues.append(f"Problem core has {wc} words (need {min_words}-{max_words})")
    return {"valid": len(issues) == 0, "word_count": wc, "issues": issues}


def validate_solution(solution: str, min_words: int = 1, max_words: int = 3) -> Dict:
    """
    Validate a solution against structural requirements.
    Returns dict with 'valid' bool and 'issues' list.
    """
    issues = []
    wc = count_words(solution)
    if wc < min_words or wc > max_words:
        issues.append(f"Solution has {wc} words (need {min_words}-{max_words})")
    if not solution.strip():
        issues.append("Solution is empty")
    return {"valid": len(issues) == 0, "word_count": wc, "issues": issues}


def validate_mini_problem(item: Dict, cfg: DictConfig) -> Dict:
    """
    Full validation of a generated mini-problem (problem + 3 solutions).
    Implements the Chain of Verification from the instructions.
    """
    results = {"valid": True, "checks": {}}

    # Check problem length
    p_check = validate_problem(
        item.get("problem", ""),
        cfg.validation.problem_min_words,
        cfg.validation.problem_max_words
    )
    results["checks"]["problem"] = p_check
    if not p_check["valid"]:
        results["valid"] = False

    # Check each solution
    for sol_type in ["ordinary_solution", "creative_solution", "implausible_solution"]:
        s_check = validate_solution(
            item.get(sol_type, ""),
            cfg.validation.solution_min_words,
            cfg.validation.solution_max_words
        )
        results["checks"][sol_type] = s_check
        if not s_check["valid"]:
            results["valid"] = False

    return results


# ---------------------------------------------------------------------------
# Model wrapper (supports vLLM for open models)
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    Unified model interface. Mirrors ModelEvaluator from run_reg.py
    but adapted for text-only generation (no images).
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_cfg = cfg.model
        log.info(f"Initializing vLLM model: {self.model_cfg.model_path}")
        self.llm = LLM(model=self.model_cfg.model_path, **self.model_cfg.vllm_args)
        self.sampling_params = SamplingParams(**cfg.sampling_params)

    def generate(self, conversations: List[List[Dict]]) -> List[str]:
        """
        Send a batch of conversations to the model and return responses.
        Each conversation is a list of {"role": ..., "content": ...} dicts.
        """
        outputs = self.llm.chat(conversations, sampling_params=self.sampling_params, use_tqdm=False)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            results.append(text)
        return results

    def generate_single(self, conversation: List[Dict]) -> str:
        """Send a single conversation and return response."""
        return self.generate([conversation])[0]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Loads prompt templates from JSON and builds conversations."""
    def __init__(self, prompt_path: str, model_name: str = ""):
        log.info(f"Loading prompt templates from: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.templates = json.load(f)
        # Some models (e.g. gemma) don't support system role
        self.system_role = "user" if "gemma" in model_name.lower() else "system"

    def build_conversation(self, template_key: str, **kwargs) -> List[Dict]:
        """
        Build a conversation from a template key and format variables.
        Templates should have 'system' and 'user' keys.
        """
        template = self.templates[template_key]
        messages = []

        if "system" in template:
            messages.append({
                "role": self.system_role,
                "content": template["system"].format(**kwargs)
            })

        if "user" in template:
            messages.append({
                "role": "user",
                "content": template["user"].format(**kwargs)
            })

        return messages


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> Optional[Dict]:
    """
    Try to parse a JSON object from model output.
    Handles common issues: markdown fences, trailing text, etc.
    """
    # Remove markdown code fences if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Try to find a JSON object
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract a JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    log.warning(f"Failed to parse JSON from response: {text[:200]}...")
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class DataHandler:
    """Handles loading examples and previous results."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load_examples(self) -> List[Dict]:
        """Load the seed examples for few-shot prompting."""
        path = self.cfg.data.examples_path
        log.info(f"Loading examples from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        log.info(f"Loaded {len(examples)} examples")
        return examples

    def load_problems(self, path: str = None) -> List[Dict]:
        """Load previously generated problems from CSV."""
        path = path or self.cfg.data.input_path
        log.info(f"Loading problems from: {path}")
        df = pd.read_csv(path)
        return df.to_dict('records')


# ---------------------------------------------------------------------------
# Result logger (mirrors ResultLogger from run_reg.py)
# ---------------------------------------------------------------------------

class ResultLogger:
    """Handles saving results to CSV and optionally logging to W&B."""
    def __init__(self, cfg: DictConfig):
        self.output_cfg = cfg.output
        self.wandb_cfg = cfg.get("wandb", None)

    def save_to_csv(self, records: List[Dict], filename: str = None) -> str:
        """Save a list of dicts to CSV."""
        filename = filename or self.output_cfg.filename
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        log.info(f"Results saved to: {os.path.join(os.getcwd(), filename)}")
        return filename

    def log_to_wandb(self, stats: Dict, records: List[Dict], output_path: str, full_config: DictConfig):
        """Log results to Weights & Biases."""
        if not self.wandb_cfg:
            log.info("W&B config not found, skipping.")
            return

        with wandb.init(
            project=self.wandb_cfg.project,
            name=self.wandb_cfg.run_name,
            job_type=self.wandb_cfg.job_type,
            config=OmegaConf.to_container(full_config, resolve=True),
        ) as run:
            if stats:
                run.log(stats)

            artifact = wandb.Artifact(name=f'{run.name}-results', type='dataset')
            artifact.add_file(output_path)
            run.log_artifact(artifact)

            table = wandb.Table(dataframe=pd.DataFrame(records))
            run.log({"results_table": table})

        log.info("W&B logging complete.")
