"""
main.py

Orchestrator: runs all reproduce scripts in sequence with timing.
"""

import os
import json
import time


def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Resolve output path relative to script directory
    config["output_path"] = os.path.join(script_dir, config["output_path"])
    os.makedirs(config["output_path"], exist_ok=True)

    # Resolve old_data input_path relative to script directory
    if "old_data" in config:
        config["old_data"]["input_path"] = os.path.join(
            script_dir, config["old_data"]["input_path"]
        )

    if "new_data" in config:
        config["new_data"]["merged_csv"] = os.path.join(
            script_dir, config["new_data"]["merged_csv"]
        )

    steps = [
        ("data_loading", "data_loading"),
        ("feature_engineering", "feature_engineering"),
        ("target_variable", "target_variable"),
        ("clustering", "clustering"),
        ("training", "training"),
        ("prediction", "prediction"),
    ]

    total_start = time.time()

    for step_name, module_name in steps:
        print(f"\n{'='*60}")
        print(f"  Step: {step_name}")
        print(f"{'='*60}")

        step_start = time.time()

        # Dynamic import
        module = __import__(module_name)
        module.run(config)

        elapsed = time.time() - step_start
        print(f"  [{step_name}] completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All steps completed in {total_elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
