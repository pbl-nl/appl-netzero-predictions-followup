import argparse
import subprocess
import sys
import os
import time

STAGES = ["download", "process", "merge", "clean", "train"]

STAGE_CONFIG = {
    "download": {"script": "download.py",             "sentinel": "data/raw/primap.csv"},
    "process":  {"script": "process.py",              "sentinel": "data/processed/worldbank.csv"},
    "merge":    {"script": "merge_target.py",         "sentinel": "data/merged/merged.csv"},
    "clean":    {"script": "cleaning_engineering.py", "sentinel": "data/cleaned/cleaned_features.csv"},
    "train":    {"script": "training.py",             "sentinel": "results/final_models.pkl"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline orchestrator for netzero predictions.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stages", nargs="+", choices=STAGES, metavar="STAGE",
        help="Run only these stages (any of: download, process, merge, clean, train).",
    )
    group.add_argument(
        "--from", dest="from_stage", choices=STAGES, metavar="STAGE",
        help="Run from this stage to the end (inclusive).",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=STAGES, metavar="STAGE", default=[],
        help="Skip these stages.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Force re-run even if a stage's output already exists.",
    )
    return parser.parse_args()


def run_stage(name, config, force):
    sentinel = config["sentinel"]
    if not force and os.path.exists(sentinel):
        print(f"[{name}] Skipping — output already exists: {sentinel}")
        return True, 0.0

    script = config["script"]
    print(f"[{name}] Running {script} ...")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[{name}] FAILED (exit code {result.returncode})")
        return False, elapsed

    print(f"[{name}] Done in {elapsed:.1f}s")
    return True, elapsed


def main():
    args = parse_args()

    # Determine ordered list of stages to run
    if args.stages:
        stages_to_run = [s for s in STAGES if s in args.stages]
    elif args.from_stage:
        start_idx = STAGES.index(args.from_stage)
        stages_to_run = STAGES[start_idx:]
    else:
        stages_to_run = list(STAGES)

    # Apply --skip
    if args.skip:
        stages_to_run = [s for s in stages_to_run if s not in args.skip]

    force = args.no_skip_existing
    timings = {}
    total_start = time.time()

    for stage in stages_to_run:
        ok, elapsed = run_stage(stage, STAGE_CONFIG[stage], force)
        timings[stage] = elapsed
        if not ok:
            print(f"\nPipeline aborted at stage '{stage}'.")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    # Summary table
    print("\n--- Timing Summary ---")
    for stage, elapsed in timings.items():
        print(f"  {stage:<10} {elapsed:>7.1f}s")
    print(f"  {'TOTAL':<10} {total_elapsed:>7.1f}s")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
