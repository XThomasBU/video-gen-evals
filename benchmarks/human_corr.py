import json
from scipy.stats import spearmanr

def normalize_key(k: str) -> str:
    """Remove .mp4 suffix if present."""
    return k[:-4] if k.endswith(".mp4") else k

def compute_spearman(human_json_path, model_json_path, is_lower_better=False):
    # Load human and model scores
    with open(human_json_path, "r") as f:
        human_scores = json.load(f)
    with open(model_json_path, "r") as f:
        model_scores = json.load(f)

    # Normalize keys
    human_scores_norm = {normalize_key(k): v for k, v in human_scores.items()}
    model_scores_norm = {normalize_key(k): v for k, v in model_scores.items()}

    if is_lower_better:
        # Invert scores if lower is better
        model_scores_norm = {k: -v for k, v in model_scores_norm.items()}

    # Align keys present in both
    common_keys = set(human_scores_norm.keys()) & set(model_scores_norm.keys())
    if not common_keys:
        raise ValueError("No overlapping keys found between human and model JSONs.")

    # Extract aligned values
    human_vals = [human_scores_norm[k] for k in common_keys]
    model_vals = [model_scores_norm[k] for k in common_keys]

    # Compute Spearman correlation
    rho, pval = spearmanr(human_vals, model_vals)
    return rho, pval
if __name__ == "__main__":

    action_path = "human_scores_analysis_action_mos_centered.json"
    anatomy_path = "human_scores_analysis_anatomy_mos_centered.json"
    appearance_path = "human_scores_analysis_appearance_mos_centered.json"
    motion_path = "human_scores_analysis_motion_mos_centered.json"
    is_lower_better = False  # Set based on your evaluation metric

    EVAL_PATH = "" # Path to Json of the format {"SoccerJuggling__21D84C98FE": 0.605084638812564, ...}

    for path in [action_path, anatomy_path, appearance_path, motion_path]:
        rho, pval = compute_spearman(path, EVAL_PATH, is_lower_better)
        print()
        print(f"Spearman correlation (rho) for {path}: {rho:.4f}")
        print(f"P-value for {path}: {pval:.4e}")
        print("-" * 40)