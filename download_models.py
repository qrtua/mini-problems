#!/usr/bin/env python3
"""
Download models from HuggingFace Hub for the Mini-Problems pipeline.

Usage:
    # Download all models:
    python download_models.py --all

    # Download specific model(s):
    python download_models.py --models llama3.1-8b qwen2.5-7b

    # List available models:
    python download_models.py --list

    # Download to custom directory:
    python download_models.py --all --cache-dir /data/hf_cache

    # Dry run (show what would be downloaded):
    python download_models.py --all --dry-run

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  (needed for gated models: Llama, Gemma)
"""
import argparse
import os
import sys
import yaml
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, HfApi, login
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# =========================================================================
# Model registry — matches configs/model/*.yaml
# =========================================================================
MODELS = {
    # --- Tier 1: Small, fast, fits easily on 4090 (~8-16GB VRAM) ---
    "phi3.5-mini": {
        "repo": "microsoft/Phi-3.5-mini-instruct",
        "size_gb": 7,
        "vram_gb": 8,
        "license": "MIT",
        "gated": False,
        "notes": "Smallest model, good for quick iteration",
    },
    "mistral-7b": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "size_gb": 15,
        "vram_gb": 15,
        "license": "Apache 2.0",
        "gated": False,
        "notes": "Solid all-rounder, good creative generation",
    },
    "qwen2.5-7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 15,
        "vram_gb": 15,
        "license": "Apache 2.0",
        "gated": False,
        "notes": "Excellent at structured JSON output",
    },
    "llama3.1-8b": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "size_gb": 16,
        "vram_gb": 16,
        "license": "Llama 3.1 Community",
        "gated": True,
        "notes": "Strong baseline, requires Meta license agreement on HF",
    },
    "deepseek-r1-7b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "size_gb": 15,
        "vram_gb": 15,
        "license": "MIT",
        "gated": False,
        "notes": "Reasoning-focused, good for evaluation tasks",
    },

    # --- Tier 2: Medium, fits on 4090 but tight (~18-23GB VRAM) ---
    "gemma2-9b": {
        "repo": "google/gemma-2-9b-it",
        "size_gb": 18,
        "vram_gb": 18,
        "license": "Gemma",
        "gated": True,
        "notes": "Google model, different perspective. Requires Gemma license on HF",
    },
    "qwen2.5-14b": {
        "repo": "Qwen/Qwen2.5-14B-Instruct",
        "size_gb": 28,
        "vram_gb": 23,
        "license": "Apache 2.0",
        "gated": False,
        "notes": "Best quality in our lineup, tight fit on 4090 (max_model_len=2048)",
    },
    "deepseek-r1-14b": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "size_gb": 28,
        "vram_gb": 23,
        "license": "MIT",
        "gated": False,
        "notes": "Best for evaluation/verification (chain-of-thought reasoning)",
    },
}

# Recommended download order: start small, validate pipeline, then scale
RECOMMENDED_ORDER = [
    "qwen2.5-7b",       # 1st: best JSON output, ungated, fast
    "llama3.1-8b",       # 2nd: strong baseline to compare
    "deepseek-r1-7b",    # 3rd: reasoning model for evaluator
    "mistral-7b",        # 4th: different perspective
    "qwen2.5-14b",       # 5th: quality upgrade
    "deepseek-r1-14b",   # 6th: best evaluator
    "gemma2-9b",         # 7th: Google's perspective
    "phi3.5-mini",       # 8th: speed baseline
]


def list_models():
    """Print available models with details."""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS FOR MINI-PROBLEMS PIPELINE")
    print("Target GPU: RTX 4090 (24GB VRAM)")
    print("=" * 80)

    print("\n--- Tier 1: Comfortable fit (8-16GB VRAM) ---\n")
    for name in RECOMMENDED_ORDER:
        info = MODELS[name]
        if info["vram_gb"] <= 16:
            gated = " [GATED - needs HF license]" if info["gated"] else ""
            print(f"  {name:20s}  ~{info['size_gb']:3d}GB disk  ~{info['vram_gb']:2d}GB VRAM  "
                  f"{info['license']:20s}{gated}")
            print(f"  {'':20s}  {info['notes']}")
            print()

    print("--- Tier 2: Tight fit (18-23GB VRAM, may need reduced context) ---\n")
    for name in RECOMMENDED_ORDER:
        info = MODELS[name]
        if info["vram_gb"] > 16:
            gated = " [GATED - needs HF license]" if info["gated"] else ""
            print(f"  {name:20s}  ~{info['size_gb']:3d}GB disk  ~{info['vram_gb']:2d}GB VRAM  "
                  f"{info['license']:20s}{gated}")
            print(f"  {'':20s}  {info['notes']}")
            print()

    print("RECOMMENDED START: qwen2.5-7b (ungated, best JSON, fast)")
    print(f"TOTAL DISK if all downloaded: ~{sum(m['size_gb'] for m in MODELS.values())}GB")
    print()


def check_gated_access(repo_id: str, token: str = None) -> bool:
    """Check if user has access to a gated model."""
    try:
        api = HfApi()
        api.model_info(repo_id, token=token)
        return True
    except GatedRepoError:
        return False
    except RepositoryNotFoundError:
        return False
    except Exception:
        return False


def download_model(name: str, cache_dir: str = None, token: str = None, dry_run: bool = False):
    """Download a single model."""
    if name not in MODELS:
        print(f"ERROR: Unknown model '{name}'. Use --list to see available models.")
        return False

    info = MODELS[name]
    repo = info["repo"]

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"Repo:  {repo}")
    print(f"Size:  ~{info['size_gb']}GB disk | ~{info['vram_gb']}GB VRAM")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"[DRY RUN] Would download {repo}")
        return True

    # Check gated access
    if info["gated"]:
        print(f"Checking access to gated model...")
        if not check_gated_access(repo, token):
            print(f"ERROR: No access to {repo}.")
            print(f"Please visit https://huggingface.co/{repo} and accept the license agreement.")
            print(f"Then run: huggingface-cli login")
            return False
        print(f"Access confirmed.")

    # Download
    try:
        print(f"Downloading {repo}...")
        path = snapshot_download(
            repo,
            cache_dir=cache_dir,
            token=token,
            # Skip large unnecessary files
            ignore_patterns=[
                "*.bin",           # prefer safetensors
                "original/*",      # skip original pytorch format if safetensors available
                "consolidated*",   # skip consolidated weights
            ],
        )
        print(f"Downloaded to: {path}")
        return True

    except GatedRepoError:
        print(f"ERROR: Access denied. Visit https://huggingface.co/{repo} to accept the license.")
        return False
    except Exception as e:
        print(f"ERROR downloading {repo}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for the Mini-Problems pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_models.py --list
  python download_models.py --models qwen2.5-7b llama3.1-8b
  python download_models.py --all --cache-dir /data/hf_cache
  python download_models.py --recommended  # downloads in optimal order
        """
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--models", nargs="+", help="Specific model names to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--recommended", action="store_true",
                        help="Download in recommended order (start with best, easiest)")
    parser.add_argument("--tier1", action="store_true",
                        help="Download only Tier 1 models (comfortable 4090 fit)")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory (default: ~/.cache/huggingface)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")

    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not any([args.models, args.all, args.recommended, args.tier1]):
        parser.print_help()
        print("\nTip: start with --list to see available models")
        return

    # Resolve token
    token = args.token or os.environ.get("HF_TOKEN", None)
    if not token:
        print("Note: No HF_TOKEN found. Gated models (Llama, Gemma) will need it.")
        print("Run: huggingface-cli login\n")

    # Determine which models to download
    if args.all:
        to_download = RECOMMENDED_ORDER
    elif args.recommended:
        to_download = RECOMMENDED_ORDER
    elif args.tier1:
        to_download = [n for n in RECOMMENDED_ORDER if MODELS[n]["vram_gb"] <= 16]
    else:
        to_download = args.models

    # Validate names
    for name in to_download:
        if name not in MODELS:
            print(f"ERROR: Unknown model '{name}'. Use --list to see options.")
            return

    # Summary
    total_gb = sum(MODELS[n]["size_gb"] for n in to_download)
    print(f"\nWill download {len(to_download)} models (~{total_gb}GB total):")
    for name in to_download:
        info = MODELS[name]
        gated = " [GATED]" if info["gated"] else ""
        print(f"  {name:20s} ~{info['size_gb']:3d}GB{gated}")
    print()

    if not args.dry_run:
        response = input("Proceed? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Download
    results = {}
    for name in to_download:
        success = download_model(name, args.cache_dir, token, args.dry_run)
        results[name] = "OK" if success else "FAILED"

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:20s} {status}")

    failed = [n for n, s in results.items() if s == "FAILED"]
    if failed:
        print(f"\n{len(failed)} model(s) failed. Check errors above.")
    else:
        print(f"\nAll {len(results)} models downloaded successfully!")
        print(f"\nNext step: test with:")
        first = to_download[0]
        print(f"  python run_generate.py model={first}")


if __name__ == "__main__":
    main()
