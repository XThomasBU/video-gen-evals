#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def load_list_json(p: Path):
    try:
        with p.open("r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []

def load_err_json(p: Path):
    try:
        with p.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def summarize(base_dir: Path):
    single_dir = base_dir / "single"
    not_dir    = base_dir / "not_single"
    err_dir    = base_dir / "errors"

    singles_set = set()
    nots_set    = set()

    # Read per-action single lists
    for jf in sorted(single_dir.glob("*.json")):
        action = jf.stem  # filename without .json
        vids = load_list_json(jf)
        for v in vids:
            # Make a stable id like "ApplyEyeMakeup/v_X.mp4"
            singles_set.add(f"{action}/{v}")

    # Read per-action not_single lists
    for jf in sorted(not_dir.glob("*.json")):
        action = jf.stem
        vids = load_list_json(jf)
        for v in vids:
            nots_set.add(f"{action}/{v}")

    processed = singles_set | nots_set
    dup_both  = singles_set & nots_set  # ideally empty

    # Optional: collect error counts
    total_errors = 0
    errors_by_action = {}
    if err_dir.exists():
        for jf in sorted(err_dir.glob("*.json")):
            action = jf.stem
            edict = load_err_json(jf)
            cnt = len(edict)
            if cnt:
                errors_by_action[action] = cnt
                total_errors += cnt

    singles = len(singles_set)
    nots    = len(nots_set)
    total   = len(processed)
    pct_single = (100.0 * singles / total) if total else 0.0

    print("=== UCF101 Mesh Processing Summary (from JSONs only) ===")
    print(f"Base dir: {base_dir}")
    print(f"Processed total: {total}")
    print(f"  - Single-person: {singles}")
    print(f"  - Not single:    {nots}")
    print(f"Percent single among processed: {pct_single:.2f}%")
    if dup_both:
        print(f"WARNING: {len(dup_both)} videos appear in BOTH single and not_single lists (check your JSONs).")

    if errors_by_action:
        print(f"\nErrors recorded (videos that raised exceptions): {total_errors}")
        # Show top 10 actions by errors
        top = sorted(errors_by_action.items(), key=lambda x: x[1], reverse=True)[:10]
        for act, cnt in top:
            print(f"  {act}: {cnt}")

def main():
    ap = argparse.ArgumentParser(description="Summarize progress from per-action JSONs (single/not_single/errors).")
    ap.add_argument(
        "--base",
        default="FINAL_MESH_UCF101",
        help="Base directory containing 'single/', 'not_single/', and optional 'errors/' subdirs.",
    )
    args = ap.parse_args()
    summarize(Path(args.base))

if __name__ == "__main__":
    main()