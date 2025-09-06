import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import itertools

LOWERIS_BETTER = {"mean_intra", "spread", "energy", "percentile_penalty", "hausdorff_score", "max_intra"}

# ADD THIS HELPER
def build_unified_df(human_scores: Dict[str, pd.DataFrame],
                     model_scores: Dict[str, pd.DataFrame],
                     metric_cols: List[str]) -> pd.DataFrame:
    """
    Create a single long dataframe with columns:
    ['video', 'class', 'model', 'human_score', <metric_cols...>]
    Only keeps rows where both human and metric exist for the same (video, class, model).
    """
    rows = []
    for model, hdf in human_scores.items():
        if model not in model_scores:
            continue
        mdf = model_scores[model]
        # keep only columns we need
        keep_cols = ['video', 'class'] + [c for c in metric_cols if c in mdf.columns]
        mdf = mdf[keep_cols].copy()

        merged = pd.merge(
            hdf[['video', 'class', 'human_score']],
            mdf,
            on=['video', 'class'],
            how='inner'
        )
        if merged.empty:
            continue
        merged['model'] = model
        rows.append(merged)

    if not rows:
        return pd.DataFrame(columns=['video','class','model','human_score'] + metric_cols)
    return pd.concat(rows, ignore_index=True)

def _orient(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return -s if series.name in LOWERIS_BETTER else s

def load_model_scores(path: str) -> pd.DataFrame:
    data = json.loads(Path(path).read_text())
    # convert list of dictionaries to dataframe
    df = pd.DataFrame(data)
    print(df.head())
    return df

def load_human_scores(human_json: str, id_map_json: str, model_name: str) -> pd.DataFrame:
    human_data = json.loads(Path(human_json).read_text())
    human_data = {
        (k[:-4] if isinstance(k, str) and k.lower().endswith(".mp4") else k): float(v)
        for k, v in human_data.items() if isinstance(v, (int, float))
    }
    map_obj = json.loads(Path(id_map_json).read_text())
    model_map = (map_obj.get("model_to_videoName_to_id") or {}).get(model_name) or {}
    id_to_video = {vid_id: video_name for video_name, vid_id in model_map.items()}

    rows = []
    for vid_id, score in human_data.items():
        video_name = id_to_video.get(vid_id)
        if video_name is None:
            continue
        cls = video_name.split("_")[1] if isinstance(video_name, str) and video_name.startswith("v_") else str(vid_id).split("__")[0]
        rows.append({"class": cls, "video": video_name, "video_id": vid_id, "human_score": score})
    
    human_df = pd.DataFrame(rows)
    print(human_df.head())
    return human_df

def pairwise_counts_no_ties(humans: np.ndarray, metrics: np.ndarray) -> Tuple[int, int]:
    n = len(humans)
    agree = 0
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            if humans[i] == humans[j]:
                continue
            total += 1
            human_pref = np.sign(humans[i] - humans[j])
            metric_pref = np.sign(metrics[i] - metrics[j])
            if human_pref == metric_pref:
                agree += 1
    return agree, total

def compute_win_ratio(scores_dict: Dict[str, pd.DataFrame], score_column: str = 'human_score') -> Dict[str, float]:
    """
    Compute win ratio for each model based on pairwise comparisons.
    
    Args:
        scores_dict: Dictionary mapping model names to DataFrames with scores
        score_column: Column name containing the scores to compare
        
    Returns:
        Dictionary mapping model names to their win ratios
    """
    model_scores = {}
    model_counts = {}
    models = list(scores_dict.keys())
    
    # Initialize counters
    for model in models:
        model_scores[model] = 0.0
        model_counts[model] = 0
    
    # Perform pairwise comparisons between all models
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i >= j:  # Avoid duplicate comparisons and self-comparison
                continue
                
            df1 = scores_dict[model1]
            df2 = scores_dict[model2]
            
            # Find common videos between the two models
            common_videos = set(df1['video']).intersection(set(df2['video']))
            
            for video in common_videos:
                score1 = df1[df1['video'] == video][score_column].iloc[0]
                score2 = df2[df2['video'] == video][score_column].iloc[0]
                
                # Award points based on comparison
                if score1 > score2:
                    model_scores[model1] += 1.0
                    model_scores[model2] += 0.0
                elif score1 < score2:
                    model_scores[model1] += 0.0
                    model_scores[model2] += 1.0
                else:  # Tie
                    model_scores[model1] += 0.5
                    model_scores[model2] += 0.5
                
                # Increment comparison counts
                model_counts[model1] += 1
                model_counts[model2] += 1
    
    # Calculate win ratios
    win_ratios = {}
    for model in models:
        if model_counts[model] > 0:
            win_ratios[model] = model_scores[model] / model_counts[model]
        else:
            win_ratios[model] = 0.0
    
    return win_ratios

def compute_vbench_win_ratios(model_scores_dict: Dict[str, pd.DataFrame], score_column: str) -> Dict[str, float]:
    """
    Compute VBench evaluation win ratios for each model using evaluation metrics.
    
    Args:
        model_scores_dict: Dictionary mapping model names to DataFrames with VBench scores
        score_column: Column name containing the VBench scores (e.g., 'mean_intra')
        
    Returns:
        Dictionary mapping model names to their VBench win ratios
    """
    # Apply orientation (negate if lower is better)
    oriented_scores = {}
    for model_name, df in model_scores_dict.items():
        if score_column in df.columns:
            df_copy = df.copy()
            df_copy[score_column] = _orient(df_copy[score_column])
            oriented_scores[model_name] = df_copy
    
    return compute_win_ratio(oriented_scores, score_column)

def compute_spearman_correlation_with_plot(human_win_ratios: Dict[str, float], 
                                         vbench_win_ratios: Dict[str, float],
                                         dimension_name: str = "VBench Dimension") -> Tuple[float, float]:
    """
    Compute Spearman correlation and create scatter plot with linear fit.
    
    Args:
        human_win_ratios: Dictionary of model -> human win ratio
        vbench_win_ratios: Dictionary of model -> VBench win ratio  
        dimension_name: Name of the evaluation dimension for plot title
        
    Returns:
        Tuple of (Spearman correlation coefficient, p-value)
    """
    # Get common models
    common_models = set(human_win_ratios.keys()).intersection(set(vbench_win_ratios.keys()))
    
    if len(common_models) < 2:
        print(f"Warning: Only {len(common_models)} models in common for {dimension_name}")
        return np.nan, np.nan
    
    # Extract values for common models
    human_vals = [human_win_ratios[model] for model in common_models]
    vbench_vals = [vbench_win_ratios[model] for model in common_models]
    
    # Compute Spearman correlation
    rho, p_value = spearmanr(human_vals, vbench_vals)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(human_vals, vbench_vals, alpha=0.7, s=100)
    
    # Add linear fit line
    if len(human_vals) > 1:
        X = np.array(human_vals).reshape(-1, 1)
        y = np.array(vbench_vals)
        reg = LinearRegression().fit(X, y)
        x_line = np.linspace(min(human_vals), max(human_vals), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        plt.plot(x_line, y_line, 'r--', alpha=0.8)
    
    # Add model labels to points
    for model, human_val, vbench_val in zip(common_models, human_vals, vbench_vals):
        plt.annotate(model, (human_val, vbench_val), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Human Preference Win Ratio')
    plt.ylabel('VBench-2.0 Evaluation Win Ratio')
    plt.title(f'{dimension_name}\nSpearman ρ = {rho:.3f} (p = {p_value:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Del.png")
    
    return rho, p_value

def compute_pairwise_accuracy(human_df: pd.DataFrame,
                              model_df: pd.DataFrame,
                              score_column: str) -> float:
    """
    Compute pairwise accuracy between human scores and a metric for a given model.
    
    Args:
        human_df: DataFrame with ['video', 'human_score']
        model_df: DataFrame with ['video', score_column]
        score_column: Name of the metric column (e.g., 'mean_intra')
        
    Returns:
        Pairwise accuracy (proportion of correctly ordered pairs)
    """
    # Align videos present in both sets
    merged = pd.merge(
        human_df[['video', 'human_score']],
        model_df[['video', score_column]],
        on='video'
    )
    if len(merged) < 2:
        return np.nan
    
    agree, total = pairwise_counts_no_ties(
        merged['human_score'].to_numpy(),
        _orient(merged[score_column]).to_numpy()
    )
    return agree / total if total > 0 else np.nan


def compute_all_pairwise_accuracies(human_scores: Dict[str, pd.DataFrame],
                                    model_scores: Dict[str, pd.DataFrame],
                                    score_column: str) -> Dict[str, float]:
    """
    Compute pairwise accuracies per model.
    """
    results = {}
    for model, human_df in human_scores.items():
        if model not in model_scores:
            continue
        acc = compute_pairwise_accuracy(human_df, model_scores[model], score_column)
        results[model] = acc
    return results

def compute_global_pairwise_accuracy(human_scores: Dict[str, pd.DataFrame],
                                     model_scores: Dict[str, pd.DataFrame],
                                     score_column: str) -> float:
    """
    Compute global VideoScore-style pairwise accuracy across all models pooled.
    
    Args:
        human_scores: dict of model -> DataFrame with human scores
        model_scores: dict of model -> DataFrame with metric scores
        score_column: column name of metric (e.g., 'mean_intra')
        
    Returns:
        Global pairwise accuracy (VideoScore definition)
    """
    all_humans = []
    all_metrics = []
    
    for model, human_df in human_scores.items():
        if model not in model_scores:
            continue
        df = pd.merge(
            human_df[['video', 'human_score']],
            model_scores[model][['video', score_column]],
            on='video'
        )
        if not df.empty:
            all_humans.extend(df['human_score'].tolist())
            all_metrics.extend(_orient(df[score_column]).tolist())
    
    if len(all_humans) < 2:
        return np.nan
    
    agree, total = pairwise_counts_no_ties(
        np.array(all_humans), np.array(all_metrics)
    )
    return agree / total if total > 0 else np.nan

# REPLACE your compute_videoscore_correlation with this ORIENTATION-AWARE version
import os
from typing import List, Tuple

def build_eval_lists_like_paper(
    human_scores: Dict[str, pd.DataFrame],
    model_scores: Dict[str, pd.DataFrame],
    metric_cols: List[str],
    drop_classes: Optional[List[str]] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Build all_ref_scores / all_ans_scores lists exactly like the paper expects:
      - all_ref_scores: list of length N items; each item is a list [ref_dim1, ref_dim2, ...]
      - all_ans_scores: list of length N items; each item is a list [ans_dim1, ans_dim2, ...]
    Here we typically have 1 metric dimension (e.g., ['mean_intra']), but support multiple.

    We merge per-model (video,class) rows and pool across models.
    """
    all_ref_scores: List[List[float]] = []
    all_ans_scores: List[List[float]] = []

    for model, hdf in human_scores.items():
        if model not in model_scores:
            continue
        mdf = model_scores[model]

        # optional drop classes
        if drop_classes:
            hdf = hdf[~hdf['class'].isin(drop_classes)]
            mdf = mdf[~mdf['class'].isin(drop_classes)]

        # keep only needed metric cols that exist
        use_metric_cols = [c for c in metric_cols if c in mdf.columns]
        if not use_metric_cols:
            continue

        merged = pd.merge(
            hdf[['video', 'class', 'human_score']],
            mdf[['video', 'class'] + use_metric_cols],
            on=['video', 'class'],
            how='inner'
        )
        if merged.empty:
            continue

        # For each row, push one item with multiple dims (if provided)
        for _, row in merged.iterrows():
            ref_vec = [float(row['human_score'])] * len(use_metric_cols)  # human ref per "aspect" count
            # Orient each metric dim
            ans_vec = []
            for dim in use_metric_cols:
                val = pd.to_numeric(pd.Series([row[dim]], name=dim), errors="coerce")
                val = _orient(val).iloc[0]
                ans_vec.append(float(val))

            all_ref_scores.append(ref_vec)
            all_ans_scores.append(ans_vec)

    return all_ref_scores, all_ans_scores


def compute_spearman_like_paper_from_lists(
    all_ref_scores: List[List[float]],
    all_ans_scores: List[List[float]],
    round_digits: int = 2,
    scale_100: bool = False,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Replicates the paper's per-dimension Spearman: iterate dimension index i,
    collect ref_list = [ref[i] for item], ans_list = [ans[i] for item], then spearmanr.
    """
    if not all_ref_scores or not all_ans_scores:
        return [], []

    ref_arr = np.array(all_ref_scores, dtype=float)
    ans_arr = np.array(all_ans_scores, dtype=float)

    D = ref_arr.shape[1]  # number of dimensions/aspects we evaluate
    spearman_list: List[Optional[float]] = []
    p_value_list: List[Optional[float]] = []

    for i in range(D):
        ref_list = ref_arr[:, i]
        ans_list = ans_arr[:, i]
        rho, p_value = spearmanr(ref_list, ans_list)
        if not np.isnan(rho):
            if scale_100:
                rho *= 100.0
            spearman_list.append(round(float(rho), round_digits))
            p_value_list.append(float(p_value))
        else:
            spearman_list.append(None)
            p_value_list.append(None)

    return spearman_list, p_value_list


def compute_spearman_like_paper(
    human_scores: Dict[str, pd.DataFrame],
    model_scores: Dict[str, pd.DataFrame],
    metric_cols: List[str],
    drop_classes: Optional[List[str]] = None,
    round_digits: int = 2,
    scale_100: bool = False,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Convenience wrapper that builds the eval lists and computes the paper-style Spearman.
    """
    all_ref_scores, all_ans_scores = build_eval_lists_like_paper(
        human_scores, model_scores, metric_cols, drop_classes
    )
    return compute_spearman_like_paper_from_lists(
        all_ref_scores, all_ans_scores, round_digits=round_digits, scale_100=scale_100
    )

def main():
    models_paths = [
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train_FINAL2backup/SAVE_TEST/jsons/gen_mesh_opensora_256p_videos_embeddings_centroids_per_video_window_32_stride_8_valid_window_NO_ENT.json",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train_FINAL2backup/SAVE_TEST/jsons/gen_mesh_runway_gen4_videos_embeddings_centroids_per_video_window_32_stride_8_valid_window_NO_ENT.json",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train_FINAL2backup/SAVE_TEST/jsons/gen_mesh_wan21_videos_embeddings_centroids_per_video_window_32_stride_8_valid_window_NO_ENT.json",
        "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/src/train_FINAL2backup/SAVE_TEST/jsons/gen_mesh_hunyuan_360p_videos_embeddings_centroids_per_video_window_32_stride_8_valid_window_NO_ENT.json",
    ]
    
    model_names = ["Opensora_256p", "runway_gen_4", "wan21", "Hunyuan_360p"]
    
    human_json = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/form_creation/FINAL/human_scores_analysis_action_mos_centered.json"
    id_map_json = "/projectnb/ivc-ml/xthomas/RESEARCH/video_evals/video-gen-evals/YOUTUBE_DATA/id_map.json"
    
    # VBench evaluation dimensions (you can add more)
    evaluation_dimensions = ["mean_intra"]

    # drop classes
    drop = ["SoccerJuggling2"]
    
    # Load human scores for each model
    human_scores = {}
    for model_name in model_names:
        human_df = load_human_scores(human_json, id_map_json, model_name)
        human_df = human_df[~human_df['class'].isin(drop)]
        if not human_df.empty:
            human_scores[model_name] = human_df

    
    # # Load VBench model scores
    model_scores = {}
    for path, model_name in zip(models_paths, model_names):
        model_df = load_model_scores(path)
        model_df = model_df[~model_df['class'].isin(drop)]
        if not model_df.empty:
            model_scores[model_name] = model_df

    # Compute human win ratios (only once)
    human_win_ratios = compute_win_ratio(human_scores, 'human_score')
    
    print("Human Preference Win Ratios:")
    print("-" * 40)
    for model, ratio in sorted(human_win_ratios.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {ratio:.4f}")
    print()
    
    # For each VBench dimension, compute win ratios and correlations
    correlations = {}
    for dimension in evaluation_dimensions:
        print(f"\n{'='*50}")
        print(f"Evaluating dimension: {dimension}")
        print('='*50)
        
        # Compute VBench win ratios for this dimension
        vbench_win_ratios = compute_vbench_win_ratios(model_scores, dimension)
        
        print(f"\nVBench-2.0 Win Ratios ({dimension}):")
        print("-" * 40)
        for model, ratio in sorted(vbench_win_ratios.items(), key=lambda x: x[1], reverse=True):
            print(f"{model}: {ratio:.4f}")
        
        # Compute Spearman correlation and create plot
        rho, p_value = compute_spearman_correlation_with_plot(
            human_win_ratios, vbench_win_ratios, f"{dimension} Dimension"
        )
        
        correlations[dimension] = {'rho': rho, 'p_value': p_value}
        print(f"\nSpearman Correlation (ρ): {rho:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Significance: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")

    # Example usage inside main
    for dimension in evaluation_dimensions:
        # per-model
        pairwise_accs = compute_all_pairwise_accuracies(human_scores, model_scores, dimension)
        print(f"\nPairwise accuracies for {dimension}:")
        for model, acc in pairwise_accs.items():
            print(f"{model}: {acc:.4f}")

        # global
        global_acc = compute_global_pairwise_accuracy(human_scores, model_scores, dimension)
        print(f"Global VideoScore-style pairwise accuracy ({dimension}): {global_acc:.4f}")

    df = build_unified_df(human_scores, model_scores, metric_cols=evaluation_dimensions)
    print(f"\nUnified df shape: {df.shape}")

    for dimension in evaluation_dimensions:
        rho, pval = compute_spearman_like_paper(human_scores, model_scores, [dimension])
        print(f"=== VideoScore Correlation for mean_intra ===")
        print(f"Spearman ρ = {rho}, p = {pval}")

    # === Models × Actions average MOS matrix ===
    def build_mos_matrix(human_scores: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Returns a DataFrame with models as rows, action classes as columns,
        and the average MOS (human_score) for each (model, class).
        """
        frames = []
        for model, df in human_scores.items():
            # mean MOS per class for this model
            per_class_mean = (
                df.groupby('class', as_index=True)['human_score']
                .mean()
                .rename(model)
            )
            frames.append(per_class_mean)
        if not frames:
            return pd.DataFrame()

        # concat along columns, then put models as rows
        mos = pd.concat(frames, axis=1).T  # rows=models, cols=classes
        mos = mos.reindex(sorted(mos.columns), axis=1)  # optional: sort actions

        # transpose
        mos = mos.T
        return mos

    mos_matrix = build_mos_matrix(human_scores)

    print("\n📊 Average MOS by Model × Action (mean over videos per class):\n")
    if mos_matrix.empty:
        print("No human scores available to build the matrix.")
    else:
        # pretty print rounded
        print(mos_matrix.round(3).fillna(np.nan).to_string())

        # # save for later analysis
        # Path("SAVE_TEST/metrics").mkdir(parents=True, exist_ok=True)
        # mos_matrix.to_csv("SAVE_TEST/metrics/mos_model_by_action.csv")
        # print("\nSaved: SAVE_TEST/metrics/mos_model_by_action.csv")


    return correlations, human_win_ratios
    # return 0, 0

if __name__ == "__main__":
    correlations, human_win_ratios = main()
