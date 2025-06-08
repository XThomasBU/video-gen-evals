import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from tabulate import tabulate
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

def compute_avg_token_changes(seq):
    diffs = []
    for i in range(len(seq) - 1):
        if len(seq[i]) == 0 or len(seq[i + 1]) == 0:
            continue
        changes = np.count_nonzero(seq[i] != seq[i + 1])
        diffs.append(changes)
    return np.mean(diffs) if diffs else None

def compute_confidence_metrics(logits_seq):
    """
    Compute average entropy, top-1 probability, and top-5 cumulative probability
    across all tokens in all frames of logits_seq.
    logits_seq: list of np.arrays, each either shape (160, 2048) or (2048,)
    Returns a dict with keys: 'Entropy', 'Top1Prob', 'Top5Prob'.
    """
    entropies = []
    top1_probs = []
    top5_probs = []

    for logits in logits_seq:
        # If per-frame logits have shape (160, 2048), use directly.
        if logits.ndim == 2:
            softmaxed = logits
        else:
            # If flattened (e.g., shape (2048,)), reshape into (1, 2048)
            softmaxed = logits.reshape(-1, 2048)

        for token_probs in softmaxed:
            entropies.append(entropy(token_probs, base=2))
            sorted_probs = np.sort(token_probs)[::-1]
            top1_probs.append(sorted_probs[0])
            top5_probs.append(np.sum(sorted_probs[:5]))

    return {
        "Entropy": np.mean(entropies) if entropies else None,
        "Top1Prob": np.mean(top1_probs) if top1_probs else None,
        "Top5Prob": np.mean(top5_probs) if top5_probs else None
    }

def convert_video_to_npz(video_folder):
    frames_path = Path(video_folder) / "tokenhmr_mesh"
    out_path = Path(video_folder) / "tokens_logits.npz"
    if not frames_path.exists():
        return
    json_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".json")])
    all_tokens, all_logits = [], []
    for jf in json_files:
        with open(frames_path / jf) as f:
            data = json.load(f)

        tokens = np.array(data.get("discrete_token", []))
        logits = np.array(data.get("cls_logits_softmax", []))

        # Collapse first dimension if it exists
        if tokens.ndim > 1:
            tokens = tokens[0]
        if logits.ndim > 1:
            logits = logits[0]

        # Flatten tokens if still >1D
        if tokens.ndim > 1:
            tokens = tokens.flatten()

        all_tokens.append(tokens)
        all_logits.append(logits)

    np.savez_compressed(out_path, tokens=all_tokens, logits=all_logits)

def load_video_tokens(video_folder):
    """
    Returns:
      - list of discrete-token arrays (one per frame)
      - list of softmax-logit arrays (one per frame)
    """
    npz_path = Path(video_folder) / "tokens_logits.npz"
    # convert_video_to_npz(video_folder)
    data = np.load(npz_path, allow_pickle=True)
    return list(data["tokens"]), list(data["logits"])

def get_token_set(token_seq):
    return set(np.concatenate([np.unique(t) for t in token_seq]))

def compute_token_transition_histogram(seq):
    counts = Counter()
    total = 0
    for i in range(len(seq) - 1):
        if len(seq[i]) == 0 or len(seq[i + 1]) == 0:
            continue
        for t1, t2 in zip(seq[i], seq[i + 1]):
            counts[(t1, t2)] += 1
            total += 1
    return {k: v / total for k, v in counts.items()} if total > 0 else None

def compute_jsd(hist1, hist2):
    keys = list(set(hist1.keys()) | set(hist2.keys()))
    p = np.array([hist1.get(k, 0.0) for k in keys])
    q = np.array([hist2.get(k, 0.0) for k in keys])
    return jensenshannon(p, q, base=2)

def get_model_from_path(path):
    if "mesh_real_videos" in path:
        return "real"
    elif "mesh_cogvideox_videos" in path:
        return "cogvideox"
    elif "mesh_runway_gen4_videos" in path:
        return "runway_gen4"
    else:
        return "unknown"

def process_all_videos(action_folders):
    raw_seqs, raw_logits = [], []
    labels, video_names, models = [], [], []
    class_to_idxs = defaultdict(list)

    # Load all sequences and logits
    for path in tqdm(action_folders, desc="Loading videos"):
        cls = Path(path).name
        model = get_model_from_path(str(path))
        for vid in sorted(os.listdir(path)):
            seq, logits = load_video_tokens(str(Path(path) / vid))
            if not seq:
                continue
            idx = len(raw_seqs)
            raw_seqs.append(seq)
            raw_logits.append(logits)
            labels.append(cls)
            models.append(model)
            video_names.append(vid)
            class_to_idxs[(cls, model)].append(idx)

    # Precompute per-video metrics
    token_sets = [get_token_set(seq) for seq in raw_seqs]
    token_trans_hists = [compute_token_transition_histogram(seq) for seq in raw_seqs]
    change_rates = [compute_avg_token_changes(seq) for seq in raw_seqs]

    per_video = []
    for i, name in enumerate(video_names):
        cls = labels[i]
        model = models[i]
        seq = raw_seqs[i]
        logits_seq = raw_logits[i]

        cls_union = set().union(*(token_sets[j] for j in class_to_idxs[(cls, model)]))
        token_cov = (len(token_sets[i] & cls_union) / len(cls_union)) if cls_union else None
        change_rate = change_rates[i]
        trans_hist = token_trans_hists[i]
        conf_metrics = compute_confidence_metrics(logits_seq)

        if token_cov is not None and change_rate is not None and trans_hist is not None:
            per_video.append({
                "Video": name,
                "Class": cls,
                "Model": model,
                "TokenCoverage": token_cov,
                "TokenTransHist": trans_hist,
                "TokenChangeRate": change_rate,
                "Entropy": conf_metrics["Entropy"],
                "Top1Prob": conf_metrics["Top1Prob"],
                "Top5Prob": conf_metrics["Top5Prob"],
            })

    # Compute real-video baselines per class
    baselines = defaultdict(lambda: defaultdict(list))
    for row in per_video:
        if row["Model"] == "real":
            cls = row["Class"]
            baselines[cls]["TokenCoverage"].append(row["TokenCoverage"])
            baselines[cls]["TokenTransHist"].append(row["TokenTransHist"])
            baselines[cls]["TokenChangeRate"].append(row["TokenChangeRate"])

    mean_baseline = {}
    for cls, metrics in baselines.items():
        if not metrics["TokenCoverage"] or not metrics["TokenChangeRate"]:
            continue
        mean_baseline[cls] = {
            "TokenCoverage": np.mean(metrics["TokenCoverage"]),
            "TokenChangeRate": np.mean(metrics["TokenChangeRate"])
        }
        avg_hist = defaultdict(float)
        valid_hists = [h for h in metrics["TokenTransHist"] if h is not None]
        if valid_hists:
            for hist in valid_hists:
                for k, v in hist.items():
                    avg_hist[k] += v
            for k in avg_hist:
                avg_hist[k] /= len(valid_hists)
            mean_baseline[cls]["TokenTransHist"] = dict(avg_hist)
        else:
            mean_baseline[cls]["TokenTransHist"] = {}

    # Normalize generated videos against real-video baselines
    normalized = []
    for row in per_video:
        if row["Model"] == "real":
            continue
        cls = row["Class"]
        base = mean_baseline.get(cls, {})
        if not base:
            continue
        real_cov = base.get("TokenCoverage", 1.0)
        real_change = base.get("TokenChangeRate", 1.0)
        if real_cov == 0 or real_change == 0:
            continue

        jsd = compute_jsd(row["TokenTransHist"], base["TokenTransHist"])
        normalized.append({
            "Video": row["Video"],
            "Class": cls,
            "Model": row["Model"],
            "TokenCoverage": row["TokenCoverage"],
            "TokenChangeRate": row["TokenChangeRate"],
            "TransitionSetJSD": jsd,
            "Entropy": row["Entropy"],
            "Top1Prob": row["Top1Prob"],
            "Top5Prob": row["Top5Prob"],
        })

    print("\n=== Normalized Generated vs Real Metrics + TransitionSetJSD ===")
    print(tabulate(normalized, headers="keys", tablefmt="grid", floatfmt=".4f", showindex=False))

    # Aggregate per-class, per-model summaries
    class_model_metrics = defaultdict(lambda: defaultdict(list))
    for row in normalized:
        key = (row["Class"], row["Model"])
        class_model_metrics[key]["TokenCoverage"].append(row["TokenCoverage"])
        class_model_metrics[key]["TransitionSetJSD"].append(row["TransitionSetJSD"])
        class_model_metrics[key]["TokenChangeRate"].append(row["TokenChangeRate"])
        class_model_metrics[key]["Entropy"].append(row["Entropy"])
        class_model_metrics[key]["Top1Prob"].append(row["Top1Prob"])
        class_model_metrics[key]["Top5Prob"].append(row["Top5Prob"])

    summary_rows = []
    for (cls, model), metrics in sorted(class_model_metrics.items()):
        if not metrics["TokenCoverage"] or not metrics["TokenChangeRate"]:
            continue
        summary_rows.append({
            "Class": cls,
            "Model": model,
            "TokenCoverage (avg)": np.mean(metrics["TokenCoverage"]),
            "TransitionSetJSD (avg)": np.mean(metrics["TransitionSetJSD"]),
            "TokenChangeRate (avg)": np.mean(metrics["TokenChangeRate"]),
            "Entropy (avg)": np.mean(metrics["Entropy"]),
            "Top1Prob (avg)": np.mean(metrics["Top1Prob"]),
            "Top5Prob (avg)": np.mean(metrics["Top5Prob"]),
        })

    print("\n=== Per-Class Model-Wise Summary ===")
    print(tabulate(summary_rows, headers="keys", tablefmt="grid", floatfmt=".4f"))

    # ------------------------
    # Additional Confidence Analysis (save plots instead of showing)
    # ------------------------

    # Separate real vs generated confidence lists
    real_entropies = [row["Entropy"] for row in per_video if row["Model"] == "real"]
    gen_entropies = [row["Entropy"] for row in per_video if row["Model"] != "real"]
    real_top1 = [row["Top1Prob"] for row in per_video if row["Model"] == "real"]
    gen_top1 = [row["Top1Prob"] for row in per_video if row["Model"] != "real"]

    # Print mean comparisons
    if real_entropies and gen_entropies:
        print(f"\nMean Entropy - Real: {np.mean(real_entropies):.4f}, Generated: {np.mean(gen_entropies):.4f}")
    if real_top1 and gen_top1:
        print(f"Mean Top-1 Prob - Real: {np.mean(real_top1):.4f}, Generated: {np.mean(gen_top1):.4f}")

    # Create output directory for plots
    plot_dir = Path("confidence_plots")
    plot_dir.mkdir(exist_ok=True)

    # Histogram of Entropy
    plt.figure()
    plt.hist(real_entropies, bins=20, alpha=0.7, label="Real")
    plt.hist(gen_entropies, bins=20, alpha=0.7, label="Generated")
    plt.title("Entropy Distribution: Real vs Generated")
    plt.xlabel("Entropy (bits)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(plot_dir / "entropy_histogram.png")
    plt.close()

    # Histogram of Top-1 Probability
    plt.figure()
    plt.hist(real_top1, bins=20, alpha=0.7, label="Real")
    plt.hist(gen_top1, bins=20, alpha=0.7, label="Generated")
    plt.title("Top-1 Probability Distribution: Real vs Generated")
    plt.xlabel("Top-1 Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(plot_dir / "top1_histogram.png")
    plt.close()

    # Boxplot combining Entropy
    plt.figure()
    plt.boxplot([real_entropies, gen_entropies], labels=["Real", "Generated"])
    plt.title("Entropy Boxplot: Real vs Generated")
    plt.ylabel("Entropy (bits)")
    plt.savefig(plot_dir / "entropy_boxplot.png")
    plt.close()

    # Boxplot combining Top-1 Prob
    plt.figure()
    plt.boxplot([real_top1, gen_top1], labels=["Real", "Generated"])
    plt.title("Top-1 Probability Boxplot: Real vs Generated")
    plt.ylabel("Top-1 Probability")
    plt.savefig(plot_dir / "top1_boxplot.png")
    plt.close()

    # Per-video quality ranking by Top-1 Prob (descending: highest confidence)
    gen_videos = [(row["Video"], row["Model"], row["Entropy"], row["Top1Prob"]) 
                  for row in normalized]
    gen_videos_sorted = sorted(gen_videos, key=lambda x: x[3], reverse=True)
    print("\n=== Generated Videos Ranked by Top-1 Probability (High to Low) ===")
    print(tabulate(gen_videos_sorted, headers=["Video", "Model", "Entropy", "Top1Prob"], floatfmt=".4f"))

if __name__ == "__main__":
    action_folders = [
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos/BodyWeightSquats",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos/PushUps",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos/HandstandPushups",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos/BabyCrawling",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_real_videos/JumpingJack",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/BodyWeightSquats",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/PushUps",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/HandstandPushups",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/BabyCrawling",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_cogvideox_videos/JumpingJack",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos/BodyWeightSquats",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos/PushUps",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos/HandstandPushups",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos/BabyCrawling",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/videos/ucf101/mesh_runway_gen4_videos/JumpingJack",
    ]
    process_all_videos(action_folders)