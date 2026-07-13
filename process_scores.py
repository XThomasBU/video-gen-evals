#!/usr/bin/env python3
"""
Script to process scores.json and create normalized comparison tables.

This script:
1. Reads scores.json
2. Extracts model and action information from filenames
3. Normalizes AC and TC scores to 0-100 scale
4. Computes averages per action and model
5. Outputs structured data for table generation
"""

import json
import re
from collections import defaultdict
from pathlib import Path


def parse_filename(filename):
    """
    Parse filename to extract model name and action.

    Examples:
    - "Hunyuan_BodyWeightSquats_01_08d40ea1.mp4" -> ("Hunyuan", "BodyWeightSquats")
    - "Opensora_768_BodyWeightSquats_01_73f1e099.mp4" -> ("Opensora_768", "BodyWeightSquats")
    """
    # Remove .mp4 extension
    name = filename.replace(".mp4", "")

    # Split by underscore
    parts = name.split("_")

    # Find the index where action name starts (typically after model name and numbers)
    # Models can have underscores, so we need to identify the action pattern
    # Actions typically start with capital letters and are one or two words

    # Known actions
    actions = [
        "BodyWeightSquats",
        "HulaHoop",
        "JumpingJack",
        "PullUps",
        "PushUps",
        "Shotput",
        "SoccerJuggling",
        "TennisSwing",
        "ThrowDiscus",
        "WallPushups",
    ]

    # Find which action appears in the filename
    action = None
    action_idx = None
    for act in actions:
        if act in name:
            action = act
            action_idx = name.find(act)
            break

    if not action:
        # Fallback: assume action is the last part before the hash
        # This is a heuristic and might not work for all cases
        for i in range(len(parts) - 1, -1, -1):
            if re.match(r"^[A-Z][a-z]+[A-Z][a-z]+", parts[i]):
                action = parts[i]
                break

    # Extract model name (everything before the action)
    if action_idx:
        model_part = name[:action_idx].rstrip("_")
        # Split model part and take the last meaningful segment
        model_parts = model_part.split("_")
        # Remove trailing numbers if they're standalone
        while model_parts and model_parts[-1].isdigit():
            model_parts.pop()
        model = "_".join(model_parts) if model_parts else model_part
    else:
        # Fallback: assume model is first part
        model = parts[0]

    return model, action


def normalize_to_0_100(value, min_val, max_val):
    """
    Normalize a value from [min_val, max_val] to [0, 100].

    Formula: normalized = ((value - min_val) / (max_val - min_val)) * 100
    """
    if max_val == min_val:
        return 50.0  # Return middle value if range is zero
    return ((value - min_val) / (max_val - min_val)) * 100.0


def main():
    # Read scores.json
    scores_path = Path("static/images/scores.json")
    if not scores_path.exists():
        print(f"Error: {scores_path} not found")
        return

    with open(scores_path, "r") as f:
        scores_data = json.load(f)

    # Collect all scores to find min/max for normalization
    all_ac_scores = []
    all_tc_scores = []

    # Parse data and group by model and action
    model_action_scores = defaultdict(lambda: defaultdict(lambda: {"ac": [], "tc": []}))
    model_name_mapping = {}

    for filename, scores in scores_data.items():
        model, action = parse_filename(filename)

        if not model or not action:
            print(f"Warning: Could not parse {filename}")
            continue

        ac = scores["ac"]
        tc = scores["tc"]

        all_ac_scores.append(ac)
        all_tc_scores.append(tc)

        model_action_scores[model][action]["ac"].append(ac)
        model_action_scores[model][action]["tc"].append(tc)

    # Find min and max for normalization
    ac_min = min(all_ac_scores)
    ac_max = max(all_ac_scores)
    tc_min = min(all_tc_scores)
    tc_max = max(all_tc_scores)

    print(f"AC range: [{ac_min:.4f}, {ac_max:.4f}]")
    print(f"TC range: [{tc_min:.4f}, {tc_max:.4f}]")
    print()

    # Get all unique actions and models
    all_actions = set()
    all_models = set()

    for model in model_action_scores:
        all_models.add(model)
        for action in model_action_scores[model]:
            all_actions.add(action)

    # Sort for consistent output
    all_actions = sorted(all_actions)
    all_models = sorted(all_models)

    print(f"Found {len(all_models)} models: {all_models}")
    print(f"Found {len(all_actions)} actions: {all_actions}")
    print()

    # Compute normalized averages
    # Structure: table_data[action][model] = {'ac': float, 'tc': float, 'avg': float}
    table_data = {}

    for action in all_actions:
        table_data[action] = {}
        for model in all_models:
            if action in model_action_scores[model]:
                ac_scores = model_action_scores[model][action]["ac"]
                tc_scores = model_action_scores[model][action]["tc"]

                # Compute raw averages
                ac_avg_raw = sum(ac_scores) / len(ac_scores) if ac_scores else 0
                tc_avg_raw = sum(tc_scores) / len(tc_scores) if tc_scores else 0

                # Normalize to 0-100
                ac_normalized = normalize_to_0_100(ac_avg_raw, ac_min, ac_max)
                tc_normalized = normalize_to_0_100(tc_avg_raw, tc_min, tc_max)

                # Average of normalized scores
                avg_normalized = (ac_normalized + tc_normalized) / 2

                table_data[action][model] = {
                    "ac": round(ac_normalized, 2),
                    "tc": round(tc_normalized, 2),
                    "avg": round(avg_normalized, 2),
                    "ac_raw": round(ac_avg_raw, 4),
                    "tc_raw": round(tc_avg_raw, 4),
                }
            else:
                table_data[action][model] = {"ac": None, "tc": None, "avg": None}

    # Compute aggregated scores (average across all actions) for each model
    aggregated_scores = {}
    for model in all_models:
        ac_scores = []
        tc_scores = []
        avg_scores = []

        for action in all_actions:
            if action in model_action_scores[model]:
                ac_scores.append(table_data[action][model]["ac"])
                tc_scores.append(table_data[action][model]["tc"])
                avg_scores.append(table_data[action][model]["avg"])

        if ac_scores:
            aggregated_scores[model] = {
                "ac": round(sum(ac_scores) / len(ac_scores), 2),
                "tc": round(sum(tc_scores) / len(tc_scores), 2),
                "avg": round(sum(avg_scores) / len(avg_scores), 2),
            }
        else:
            aggregated_scores[model] = {"ac": None, "tc": None, "avg": None}

    # Output JSON for table generation
    output_data = {
        "normalization_ranges": {
            "ac": {"min": ac_min, "max": ac_max},
            "tc": {"min": tc_min, "max": tc_max},
        },
        "models": all_models,
        "actions": all_actions,
        "table_data": table_data,
        "aggregated_scores": aggregated_scores,
    }

    output_path = Path("static/images/comparison_table.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Output saved to {output_path}")
    print()

    # Print a preview table (AC scores)
    print("Preview - AC Scores (normalized 0-100):")
    print("-" * 80)
    header = f"{'Action':<25}"
    for model in all_models:
        # Truncate model name for display
        model_display = model[:12] if len(model) > 12 else model
        header += f" {model_display:>12}"
    print(header)
    print("-" * 80)

    for action in all_actions:
        row = f"{action:<25}"
        for model in all_models:
            if table_data[action][model]["ac"] is not None:
                row += f" {table_data[action][model]['ac']:>12.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print()
    print("TC Scores (normalized 0-100):")
    print("-" * 80)
    print(header)
    print("-" * 80)

    for action in all_actions:
        row = f"{action:<25}"
        for model in all_models:
            if table_data[action][model]["tc"] is not None:
                row += f" {table_data[action][model]['tc']:>12.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print()
    print("Average Scores (normalized 0-100):")
    print("-" * 80)
    print(header)
    print("-" * 80)

    for action in all_actions:
        row = f"{action:<25}"
        for model in all_models:
            if table_data[action][model]["avg"] is not None:
                row += f" {table_data[action][model]['avg']:>12.2f}"
            else:
                row += f" {'N/A':>12}"
        print(row)


if __name__ == "__main__":
    main()
