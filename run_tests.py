#!/usr/bin/env python3
"""
Test runner - 5 configurations x 50 problems + optional run F

Usage:
    $env:GEMINI_API_KEY = "your_key"   (PowerShell)
    python run_tests.py                # Runs A-E (Gemini Flash)
    python run_tests.py --all          # Runs A-F (including Gemini Pro)

Estimated cost: ~$0.14 (A-E) or ~$0.20 (A-F) on Gemini Flash
"""

import os
import sys
import subprocess

def run(label, description, args):
    print(f"[{label}] {description}")
    cmd = ["python", "pipeline.py"] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Run {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY")
        print("  PowerShell: $env:GEMINI_API_KEY = 'your_key'")
        sys.exit(1)

    os.makedirs("test_results", exist_ok=True)

    print("============================================")
    print("  MINI-PROBLEMS: CONFIGURATION TESTS")
    print("============================================")
    print()

    run("A", "Baseline: 2 same-cat + 3 cross-cat, batch=3", [
        "--no-approved", "--n-problems", "50", "--cat-floor", "3",
        "--no-solve", "--no-evaluate",
        "--output", "test_results/run_A.csv"
    ])

    run("B", "More anchoring: 3 same-cat + 2 cross-cat", [
        "--no-approved", "--n-problems", "50", "--cat-floor", "3",
        "--n-same-cat", "3", "--n-cross-cat", "2",
        "--no-solve", "--no-evaluate",
        "--output", "test_results/run_B.csv"
    ])

    run("C", "Minimal input: 1 same + 2 cross + 1 rejected", [
        "--no-approved", "--n-problems", "50", "--cat-floor", "3",
        "--n-same-cat", "1", "--n-cross-cat", "2", "--n-rejected", "1",
        "--no-solve", "--no-evaluate",
        "--output", "test_results/run_C.csv"
    ])

    run("D", "No rejected: n-rejected=0", [
        "--no-approved", "--n-problems", "50", "--cat-floor", "3",
        "--n-rejected", "0",
        "--no-solve", "--no-evaluate",
        "--output", "test_results/run_D.csv"
    ])

    if "--all" in sys.argv:
        run("F", "Gemini Pro (baseline settings)", [
            "--n-problems", "50", "--cat-floor", "3",
            "--model", "gemini-2.5-pro-preview-05-06",
            "--no-solve", "--no-evaluate",
            "--output", "test_results/run_F.csv"
        ])

    print()
    print("============================================")
    print("  DONE - results in test_results/")
    print("============================================")
    print()

    for f in sorted(os.listdir("test_results")):
        path = os.path.join("test_results", f)
        size = os.path.getsize(path)
        print(f"  {path}  ({size} bytes)")

    print()
    print("Next steps:")
    print("  1. Send test_results/*.csv to a team member for evaluation")
    print("  2. Evaluation criteria for each run:")
    print("     - Missing constraints")
    print("     - Knowledge-based problems")
    print("     - Thematic duplicates")
    print("     - Creative solutions that are too ordinary")
    print("     - Implausible solutions that would actually work")

if __name__ == "__main__":
    main()
