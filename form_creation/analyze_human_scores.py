#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# ---------------------------
# IO helpers
# ---------------------------

def _corr_str(x: np.ndarray, y: np.ndarray) -> str:
    # guard for n<2
    if len(x) < 2:
        return "N<2"
    s_rho, _ = spearmanr(x, y)
    p_r, _ = pearsonr(x, y)
    return f"Spearman={s_rho:.3f}  Pearson={p_r:.3f}"

def _pair_common(df: pd.DataFrame, rater_a: str, rater_b: str, col: str) -> pd.DataFrame:
    """
    Return common videos and the two raters' columns as a tidy frame: ['video','a','b']
    """
    da = df[df.rater == rater_a][["video", col]].rename(columns={col: "a"})
    db = df[df.rater == rater_b][["video", col]].rename(columns={col: "b"})
    m = pd.merge(da, db, on="video", how="inner")
    return m.sort_values("video").reset_index(drop=True)

def plot_two_raters_line(all_raw: pd.DataFrame,
                         corrected: pd.DataFrame,
                         rater_a: str,
                         rater_b: str,
                         outpath: Path) -> Path:
    """
    Line plot of scores for two raters across video index.
    Shows RAW, Centered, and Z to illustrate same trends.
    """
    # Merge raw
    da = all_raw[all_raw.rater == rater_a][["video","raw_score"]].rename(columns={"raw_score":"raw_a"})
    db = all_raw[all_raw.rater == rater_b][["video","raw_score"]].rename(columns={"raw_score":"raw_b"})
    raw = pd.merge(da, db, on="video").sort_values("video").reset_index(drop=True)

    # Merge centered
    da = corrected[corrected.rater == rater_a][["video","centered","z"]].rename(columns={"centered":"cent_a","z":"z_a"})
    db = corrected[corrected.rater == rater_b][["video","centered","z"]].rename(columns={"centered":"cent_b","z":"z_b"})
    corr = pd.merge(da, db, on="video").sort_values("video").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10,5))

    # X = video index (not name, just order)
    x = np.arange(len(raw))

    # Raw
    ax.plot(x, raw["raw_a"], label=f"{rater_a} RAW", color="C0", alpha=0.6)
    ax.plot(x, raw["raw_b"], label=f"{rater_b} RAW", color="C1", alpha=0.6)

    # Centered
    ax.plot(x, corr["cent_a"], label=f"{rater_a} Centered", color="C0", linestyle="--")
    ax.plot(x, corr["cent_b"], label=f"{rater_b} Centered", color="C1", linestyle="--")

    # Z
    ax.plot(x, corr["z_a"], label=f"{rater_a} Z", color="C0", linestyle=":")
    ax.plot(x, corr["z_b"], label=f"{rater_b} Z", color="C1", linestyle=":")

    ax.set_xlabel("Video index")
    ax.set_ylabel("Score")
    ax.set_title(f"{rater_a} vs {rater_b}: RAW vs bias-corrected trends")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath

def load_scores(path: str, rater_name: str = None) -> pd.DataFrame:
    """
    JSON format:
      {
        "VideoName1.mp4": 3,
        "VideoName2.mp4": 5,
        ...
      }
    Returns DataFrame: ['video','raw_score','rater']
    """
    p = Path(path)
    data = json.loads(p.read_text())
    rows = [(k, float(v)) for k, v in data.items() if isinstance(v, (int, float))]
    name = rater_name or p.stem
    return pd.DataFrame(rows, columns=["video", "raw_score"]).assign(rater=name)

def per_rater_center_and_z(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: columns ['video','raw_score','rater']
    Adds ['centered','z'] per rater.
    """
    def _center_z(g: pd.DataFrame) -> pd.DataFrame:
        mean = g["raw_score"].mean()
        std = g["raw_score"].std(ddof=0)
        centered = g["raw_score"] - mean
        if std is None or std == 0 or np.isnan(std):
            z = np.zeros_like(centered, dtype=float)
        else:
            z = centered / std
        g = g.copy()
        g["centered"] = centered
        g["z"] = z
        return g
    return df.groupby("rater", group_keys=False).apply(_center_z)

# ---------------------------
# Correlations
# ---------------------------
def pairwise_raw_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Spearman/Pearson between every pair of raters
    on RAW scores, restricted to their common videos.
    Returns:
      ['rater_i','rater_j','N_common','spearman_rho','spearman_p','pearson_r','pearson_p']
    """
    raters = sorted(df["rater"].unique())
    rows = []
    for a, b in itertools.combinations(raters, 2):
        da = df[df.rater == a][["video", "raw_score"]].rename(columns={"raw_score": "raw_a"})
        db = df[df.rater == b][["video", "raw_score"]].rename(columns={"raw_score": "raw_b"})
        m = pd.merge(da, db, on="video", how="inner")
        n = len(m)
        if n >= 2:
            s_rho, s_p = spearmanr(m["raw_a"], m["raw_b"])
            p_r, p_p = pearsonr(m["raw_a"], m["raw_b"])
        else:
            s_rho = s_p = p_r = p_p = np.nan
        rows.append({
            "rater_i": a, "rater_j": b, "N_common": n,
            "spearman_rho": s_rho, "spearman_p": s_p,
            "pearson_r": p_r, "pearson_p": p_p
        })
    return pd.DataFrame(rows).sort_values(["rater_i","rater_j"]).reset_index(drop=True)

# ---------------------------
# MOS aggregation
# ---------------------------
def compute_mos_over_all_raters(corrected_df: pd.DataFrame) -> pd.DataFrame:
    """
    corrected_df: columns ['video','rater','centered','z']
    For each video, average across all available raters.
    Returns ['video','MOS_centered','MOS_z','N_raters']
    """
    agg = (
        corrected_df
        .groupby("video")
        .agg(
            MOS_centered=("centered", "mean"),
            MOS_z=("z", "mean"),
            N_raters=("rater", "nunique"),
        )
        .reset_index()
        .sort_values("video")
    )
    return agg

def summarize_min_max(mos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce min/max summary for MOS_centered and MOS_z.
    Returns:
        metric        min_video  min_val   max_video   max_val
    """
    rows: List[Dict] = []
    for col in ["MOS_centered", "MOS_z"]:
        if mos_df.empty:
            rows.append({"metric": col, "min_video": None, "min_val": np.nan,
                         "max_video": None, "max_val": np.nan})
            continue
        vmin = mos_df[col].min()
        vmax = mos_df[col].max()
        vmin_vid = mos_df.loc[mos_df[col].idxmin(), "video"]
        vmax_vid = mos_df.loc[mos_df[col].idxmax(), "video"]
        rows.append({
            "metric": col,
            "min_video": vmin_vid,
            "min_val": float(vmin),
            "max_video": vmax_vid,
            "max_val": float(vmax),
        })
    return pd.DataFrame(rows)

def save_mos_json(mos_df: pd.DataFrame, json_prefix: Path) -> Tuple[Path, Path]:
    """
    Save MOS_centered and MOS_z as flat JSONs: {"video.mp4": score, ...}
    """
    mos_centered_map = {row.video: float(row.MOS_centered) for row in mos_df.itertuples(index=False)}
    mos_z_map        = {row.video: float(row.MOS_z)        for row in mos_df.itertuples(index=False)}

    path_centered = json_prefix.parent / f"{json_prefix.name}_mos_centered.json"
    path_z        = json_prefix.parent / f"{json_prefix.name}_mos_z.json"

    path_centered.write_text(json.dumps(mos_centered_map, indent=2))
    path_z.write_text(json.dumps(mos_z_map, indent=2))
    return path_centered, path_z

def save_per_rater_corrected(corrected_df: pd.DataFrame, out_prefix: Path) -> List[Path]:
    """
    Save each rater's centered and z scores as JSON.
    One JSON per rater, two keys per video: {'video': score}.
    Filenames:
      <prefix>_<rater>_centered.json
      <prefix>_<rater>_z.json
    """
    paths = []
    for rater, g in corrected_df.groupby("rater"):
        centered_map = {row.video: float(row.centered) for row in g.itertuples(index=False)}
        z_map        = {row.video: float(row.z)        for row in g.itertuples(index=False)}

        # safe file-friendly name
        rater_fname = str(rater).replace(" ", "_")
        path_c = out_prefix.parent / f"{out_prefix.name}_{rater_fname}_centered.json"
        path_z = out_prefix.parent / f"{out_prefix.name}_{rater_fname}_z.json"

        path_c.write_text(json.dumps(centered_map, indent=2))
        path_z.write_text(json.dumps(z_map, indent=2))

        paths.extend([path_c, path_z])
    return paths

def summarize_min_max_per_rater(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: ['video','raw_score','rater','centered','z']
    Returns per-rater min/max for raw_score, centered, and z.
    """
    rows = []
    for rater, g in df.groupby("rater"):
        for col in ["raw_score", "centered", "z"]:
            vmin = g[col].min()
            vmax = g[col].max()
            vmin_vid = g.loc[g[col].idxmin(), "video"]
            vmax_vid = g.loc[g[col].idxmax(), "video"]
            rows.append({
                "rater": rater,
                "metric": col,
                # "min_video": vmin_vid,
                "min_val": float(vmin),
                # "max_video": vmax_vid,
                "max_val": float(vmax),
            })
    return pd.DataFrame(rows)

def pairwise_corrected_correlations(df: pd.DataFrame, col: str = "centered") -> pd.DataFrame:
    """
    Compute pairwise Spearman/Pearson between every pair of raters
    on corrected scores (e.g. 'centered' or 'z'), restricted to common videos.
    Returns:
      ['rater_i','rater_j','N_common','spearman_rho','spearman_p','pearson_r','pearson_p']
    """
    raters = sorted(df["rater"].unique())
    rows = []
    for a, b in itertools.combinations(raters, 2):
        da = df[df.rater == a][["video", col]].rename(columns={col: "a"})
        db = df[df.rater == b][["video", col]].rename(columns={col: "b"})
        m = pd.merge(da, db, on="video", how="inner")
        n = len(m)
        if n >= 2:
            s_rho, s_p = spearmanr(m["a"], m["b"])
            p_r, p_p = pearsonr(m["a"], m["b"])
        else:
            s_rho = s_p = p_r = p_p = np.nan
        rows.append({
            "rater_i": a, "rater_j": b, "N_common": n,
            "spearman_rho": s_rho, "spearman_p": s_p,
            "pearson_r": p_r, "pearson_p": p_p
        })
    return pd.DataFrame(rows).sort_values(["rater_i","rater_j"]).reset_index(drop=True)

def save_mos_raw_json(all_raw: pd.DataFrame, json_prefix: Path) -> Path:
    """
    Save plain average of raw scores per video (no bias correction).
    Output: {"video.mp4": avg_raw_score, ...}
    """
    mos_raw_map = (
        all_raw
        .groupby("video")["raw_score"]
        .mean()
        .apply(float)
        .to_dict()
    )
    path_raw = json_prefix.parent / f"{json_prefix.name}_mos_raw.json"
    path_raw.write_text(json.dumps(mos_raw_map, indent=2))
    return path_raw

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Inter-rater analysis and MOS over N raters with per-rater bias correction.")
    ap.add_argument("raters", nargs="+", type=str, help="Paths to rater JSONs (>=2).")
    ap.add_argument("--rater-names", nargs="*", default=None,
                    help="Optional names for raters (same order as files). Defaults to file stems.")
    ap.add_argument("--out", type=str, default="human_scores_analysis.csv",
                    help="Output CSV with per-video MOS and per-rater corrected scores.")
    ap.add_argument("--json-prefix", type=str, default=None,
                    help="Filename prefix (no extension) for MOS JSONs. Defaults to stem of --out.")
    ap.add_argument("--pairwise-out", type=str, default="pairwise_correlations.csv",
                    help="Optional CSV for pairwise inter-rater correlations.")
    args = ap.parse_args()

    if len(args.raters) < 2:
        raise SystemExit("Provide at least two rater JSON files.")

    # Load all raters
    if args.rater_names and (len(args.rater_names) != len(args.raters)):
        raise SystemExit("--rater-names length must match number of rater files.")
    dfs = []
    for i, path in enumerate(args.raters):
        name = (args.rater_names[i] if args.rater_names
                else Path(path).stem)
        dfs.append(load_scores(path, rater_name=name))
    all_raw = pd.concat(dfs, ignore_index=True)

    # Pairwise correlations on RAW scores
    pw = pairwise_raw_correlations(all_raw)
    if not pw.empty:
        print("\n[Pairwise inter-rater correlations on RAW scores]")
        print(pw.to_string(index=False))
    else:
        print("\n[Pairwise inter-rater correlations on RAW scores] No pairs with >=2 common videos.")

    # Per-rater bias correction
    corrected = per_rater_center_and_z(all_raw)

    # MOS across all available raters (per video)
    mos = compute_mos_over_all_raters(corrected)

    # Summary
    print("\n[Per-video MOS summary]")
    summ = summarize_min_max(mos)
    print(summ.to_string(index=False))

    # Build tidy table to export: each rater's corrected scores + MOS per video
    tidy = (
        corrected
        .merge(mos, on="video", how="left")
        .sort_values(["video", "rater"])
        .reset_index(drop=True)
    )

    # Save CSVs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_path, index=False)
    print(f"\nWrote per-rater & MOS CSV to: {out_path}")

    if pw is not None and not pw.empty and args.pairwise_out:
        pw_path = Path(args.pairwise_out)
        pw_path.parent.mkdir(parents=True, exist_ok=True)
        pw.to_csv(pw_path, index=False)
        print(f"Wrote pairwise correlations CSV to: {pw_path}")

    # Save MOS JSONs
    if args.json_prefix is None:
        json_prefix = out_path.with_suffix("")  # same folder, stem of CSV
    else:
        json_prefix = Path(args.json_prefix)
        json_prefix.parent.mkdir(parents=True, exist_ok=True)

    mos_centered_path, mos_z_path = save_mos_json(mos, json_prefix)
    print(f"Wrote MOS (centered) JSON to: {mos_centered_path}")
    print(f"Wrote MOS (z-scored) JSON to: {mos_z_path}")

    print("\n[Per-rater min/max summary]")
    per_rater_summ = summarize_min_max_per_rater(corrected)
    print(per_rater_summ.to_string(index=False))

    # Save per-rater corrected JSONs
    per_rater_paths = save_per_rater_corrected(corrected, json_prefix)
    print(f"Wrote per-rater centered/z JSONs for each user:")
    for p in per_rater_paths:
        print(f"  {p}")

    # Pairwise correlations on CENTERED scores
    pw_centered = pairwise_corrected_correlations(corrected, col="centered")
    if not pw_centered.empty:
        print("\n[Pairwise inter-rater correlations on CENTERED scores]")
        print(pw_centered.to_string(index=False))

    mos_raw_path = save_mos_raw_json(all_raw, json_prefix)

    # mos_raw = all_raw.groupby("video")["raw_score"].mean().reset_index(name="MOS_raw")
    # mos_compare = mos.merge(mos_raw, on="video")
    # print(mos_compare.head())

    plot_path = Path("scatter_two_raters_combined.png")
    plot_two_raters_line(all_raw, corrected, "asriniv1_scores", "tianle_scores", plot_path)
    print(f"Saved combined scatter: {plot_path}")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import argparse
# import json
# from pathlib import Path
# from typing import Dict, Tuple, List
# import itertools
# import numpy as np
# import pandas as pd
# from scipy.stats import spearmanr, pearsonr

# # ---------------------------
# # IO helpers
# # ---------------------------
# def load_scores(path: str, rater_name: str = None) -> pd.DataFrame:
#     """
#     JSON format:
#       {
#         "VideoName1.mp4": 3,
#         "VideoName2.mp4": 5,
#         ...
#       }
#     Returns DataFrame: ['video','raw_score','rater']
#     """
#     p = Path(path)
#     data = json.loads(p.read_text())
#     rows = [(k, float(v)) for k, v in data.items() if isinstance(v, (int, float))]
#     name = rater_name or p.stem
#     return pd.DataFrame(rows, columns=["video", "raw_score"]).assign(rater=name)

# def per_rater_center_and_z(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     df: columns ['video','raw_score','rater']
#     Adds ['centered','z'] per rater.
#     """
#     def _center_z(g: pd.DataFrame) -> pd.DataFrame:
#         mean = g["raw_score"].mean()
#         std = g["raw_score"].std(ddof=0)
#         centered = g["raw_score"] - mean
#         if std is None or std == 0 or np.isnan(std):
#             z = np.zeros_like(centered, dtype=float)
#         else:
#             z = centered / std
#         g = g.copy()
#         g["centered"] = centered
#         g["z"] = z
#         return g
#     return df.groupby("rater", group_keys=False).apply(_center_z)

# # ---------------------------
# # Correlations
# # ---------------------------
# def pairwise_raw_correlations(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute pairwise Spearman/Pearson between every pair of raters
#     on RAW scores, restricted to their common videos.
#     Returns:
#       ['rater_i','rater_j','N_common','spearman_rho','spearman_p','pearson_r','pearson_p']
#     """
#     raters = sorted(df["rater"].unique())
#     rows = []
#     for a, b in itertools.combinations(raters, 2):
#         da = df[df.rater == a][["video", "raw_score"]].rename(columns={"raw_score": "raw_a"})
#         db = df[df.rater == b][["video", "raw_score"]].rename(columns={"raw_score": "raw_b"})
#         m = pd.merge(da, db, on="video", how="inner")
#         n = len(m)
#         if n >= 2:
#             s_rho, s_p = spearmanr(m["raw_a"], m["raw_b"])
#             p_r, p_p = pearsonr(m["raw_a"], m["raw_b"])
#         else:
#             s_rho = s_p = p_r = p_p = np.nan
#         rows.append({
#             "rater_i": a, "rater_j": b, "N_common": n,
#             "spearman_rho": s_rho, "spearman_p": s_p,
#             "pearson_r": p_r, "pearson_p": p_p
#         })
#     return pd.DataFrame(rows).sort_values(["rater_i","rater_j"]).reset_index(drop=True)

# # ---------------------------
# # Categorical preparation
# # ---------------------------
# def categorize_scores(all_raw: pd.DataFrame, mode: str = "auto", bins: int = 5) -> pd.DataFrame:
#     """
#     Produce categorical labels in column 'cat' across ALL raters jointly.
#     Modes:
#       - 'auto': if all values are integer-like with <=20 unique, use 'int'; else 'bin'
#       - 'int' : round to nearest int, treat as categories
#       - 'round': same as 'int'
#       - 'bin' : equal-width bins over pooled raw scores into `bins` categories [0..bins-1]
#     """
#     vals = all_raw["raw_score"].to_numpy()
#     if mode == "auto":
#         all_int_like = np.all(np.isclose(vals, np.round(vals)))
#         uniq = np.unique(np.round(vals)).size
#         mode = "int" if (all_int_like and uniq <= 20) else "bin"

#     if mode in ("int", "round"):
#         cats = np.round(vals).astype(int)
#     elif mode == "bin":
#         vmin, vmax = float(np.min(vals)), float(np.max(vals))
#         if vmin == vmax:
#             cats = np.zeros_like(vals, dtype=int)
#         else:
#             edges = np.linspace(vmin, vmax, num=bins + 1)
#             cats = pd.cut(all_raw["raw_score"], bins=edges, include_lowest=True, labels=False).astype(int).to_numpy()
#     else:
#         raise SystemExit(f"Unknown --categorize mode: {mode}")

#     df_cat = all_raw.copy()
#     df_cat["cat"] = cats
#     return df_cat, mode

# # ---------------------------
# # IAA: Matching Ratio (pairwise & all-agree)
# # ---------------------------
# def matching_ratio_stats(cat_pivot: pd.DataFrame) -> Tuple[float, float]:
#     """
#     cat_pivot: rows=videos, cols=raters, integer categories, no NaNs.
#     Returns:
#       (avg_pairwise_agreement, all_raters_exact_agreement_rate)
#     """
#     n_raters = cat_pivot.shape[1]
#     if n_raters < 2 or cat_pivot.empty:
#         return np.nan, np.nan
#     total_pairs = n_raters * (n_raters - 1) // 2

#     pair_agreements = []
#     all_agree_flags = []
#     # Precompute max category id for bincount length
#     cat_max = int(np.nanmax(cat_pivot.to_numpy()))
#     for row in cat_pivot.to_numpy():
#         row = row.astype(int)
#         counts = np.bincount(row, minlength=cat_max + 1).astype(np.int64)
#         agree_pairs = int(((counts * (counts - 1)) // 2).sum())
#         pair_agreements.append(agree_pairs / total_pairs)
#         all_agree_flags.append(1.0 if counts.max() == n_raters else 0.0)

#     return float(np.mean(pair_agreements)), float(np.mean(all_agree_flags))

# # ---------------------------
# # IAA: Fleiss' kappa (nominal)
# # ---------------------------
# def fleiss_kappa(cat_pivot: pd.DataFrame) -> Tuple[float, float, float]:
#     """
#     Standard Fleiss' κ on items rated by ALL raters.
#     cat_pivot: rows=videos, cols=raters, integer categories, no NaNs.
#     Returns: (kappa, P_bar, P_e)
#     """
#     if cat_pivot.empty:
#         return np.nan, np.nan, np.nan
#     n_items, n_raters = cat_pivot.shape
#     cats = np.unique(cat_pivot.to_numpy())
#     cats = cats[~np.isnan(cats)].astype(int)
#     n_cats = len(cats)
#     if n_raters < 2 or n_cats < 2:
#         return np.nan, np.nan, np.nan

#     # counts[i, j] = count of category j on item i
#     cat_to_idx = {c: i for i, c in enumerate(cats)}
#     counts = np.zeros((n_items, n_cats), dtype=np.int64)
#     for i, row in enumerate(cat_pivot.to_numpy()):
#         for c in row.astype(int):
#             counts[i, cat_to_idx[c]] += 1

#     P_i = (counts * (counts - 1)).sum(axis=1) / (n_raters * (n_raters - 1))
#     P_bar = P_i.mean()
#     p_j = counts.sum(axis=0) / (n_items * n_raters)
#     P_e = (p_j ** 2).sum()
#     denom = (1.0 - P_e)
#     kappa = (P_bar - P_e) / denom if denom > 0 else np.nan
#     return float(kappa), float(P_bar), float(P_e)

# # ---------------------------
# # IAA: Krippendorff's alpha (interval & nominal)
# # ---------------------------
# def krippendorff_alpha_interval(raw_pivot: pd.DataFrame) -> Tuple[float, float, float]:
#     """
#     Interval alpha using squared differences; supports NaNs.
#     Returns: (alpha, Do, De)
#     """
#     Do_num = 0.0
#     Do_den = 0.0
#     for vals in raw_pivot.to_numpy():
#         v = vals[~np.isnan(vals)]
#         n = v.size
#         if n < 2:
#             continue
#         sum_x = float(v.sum())
#         sum_x2 = float((v * v).sum())
#         # sum_{i<j} (x_i - x_j)^2 = n * sum(x^2) - (sum x)^2
#         S = n * sum_x2 - (sum_x ** 2)
#         Do_num += S
#         Do_den += n * (n - 1) / 2.0
#     Do = Do_num / Do_den if Do_den > 0 else np.nan

#     a = raw_pivot.to_numpy().ravel()
#     a = a[~np.isnan(a)]
#     N = a.size
#     if N < 2:
#         return np.nan, Do, np.nan
#     sum_x = float(a.sum())
#     sum_x2 = float((a * a).sum())
#     De_num = N * sum_x2 - (sum_x ** 2)
#     De_den = N * (N - 1) / 2.0
#     De = De_num / De_den if De_den > 0 else np.nan

#     alpha = 1.0 - (Do / De) if (De is not np.nan and De > 0) else np.nan
#     return float(alpha), float(Do), float(De)

# def krippendorff_alpha_nominal(cat_pivot: pd.DataFrame) -> Tuple[float, float, float]:
#     """
#     Nominal alpha; supports NaNs (but for Fleiss we already intersect).
#     Returns: (alpha, Do, De)
#     """
#     # Determine category set
#     arr = cat_pivot.to_numpy()
#     arr = arr[~np.isnan(arr)]
#     if arr.size == 0:
#         return np.nan, np.nan, np.nan
#     cats = np.unique(arr).astype(int)
#     K = cats.size
#     cat_to_idx = {c: i for i, c in enumerate(cats)}
#     total_counts = np.zeros(K, dtype=np.float64)

#     Do_num = 0.0
#     Do_den = 0.0
#     for row in cat_pivot.to_numpy():
#         v = row[~np.isnan(row)]
#         n = v.size
#         if n < 2:
#             continue
#         counts = np.zeros(K, dtype=np.float64)
#         for c in v.astype(int):
#             counts[cat_to_idx[c]] += 1.0
#         pairs = n * (n - 1) / 2.0
#         agree_pairs = ((counts * (counts - 1)) / 2.0).sum()
#         disagree_pairs = pairs - agree_pairs
#         Do_num += disagree_pairs
#         Do_den += pairs
#         total_counts += counts

#     Do = Do_num / Do_den if Do_den > 0 else np.nan
#     N = total_counts.sum()
#     if N < 2:
#         return np.nan, Do, np.nan
#     De = 1.0 - ((total_counts * (total_counts - 1)).sum() / (N * (N - 1)))
#     alpha = 1.0 - (Do / De) if De > 0 else np.nan
#     return float(alpha), float(Do), float(De)

# # ---------------------------
# # MOS aggregation
# # ---------------------------
# def compute_mos_over_all_raters(corrected_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     corrected_df: columns ['video','rater','centered','z']
#     For each video, average across all available raters.
#     Returns ['video','MOS_centered','MOS_z','N_raters']
#     """
#     agg = (
#         corrected_df
#         .groupby("video")
#         .agg(
#             MOS_centered=("centered", "mean"),
#             MOS_z=("z", "mean"),
#             N_raters=("rater", "nunique"),
#         )
#         .reset_index()
#         .sort_values("video")
#     )
#     return agg

# def summarize_min_max(mos_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Produce min/max summary for MOS_centered and MOS_z.
#     Returns:
#         metric        min_video  min_val   max_video   max_val
#     """
#     rows: List[Dict] = []
#     for col in ["MOS_centered", "MOS_z"]:
#         if mos_df.empty:
#             rows.append({"metric": col, "min_video": None, "min_val": np.nan,
#                          "max_video": None, "max_val": np.nan})
#             continue
#         vmin = mos_df[col].min()
#         vmax = mos_df[col].max()
#         vmin_vid = mos_df.loc[mos_df[col].idxmin(), "video"]
#         vmax_vid = mos_df.loc[mos_df[col].idxmax(), "video"]
#         rows.append({
#             "metric": col,
#             "min_video": vmin_vid,
#             "min_val": float(vmin),
#             "max_video": vmax_vid,
#             "max_val": float(vmax),
#         })
#     return pd.DataFrame(rows)

# def save_mos_json(mos_df: pd.DataFrame, json_prefix: Path) -> Tuple[Path, Path]:
#     """
#     Save MOS_centered and MOS_z as flat JSONs: {"video.mp4": score, ...}
#     """
#     mos_centered_map = {row.video: float(row.MOS_centered) for row in mos_df.itertuples(index=False)}
#     mos_z_map        = {row.video: float(row.MOS_z)        for row in mos_df.itertuples(index=False)}

#     path_centered = json_prefix.parent / f"{json_prefix.name}_mos_centered.json"
#     path_z        = json_prefix.parent / f"{json_prefix.name}_mos_z.json"

#     path_centered.write_text(json.dumps(mos_centered_map, indent=2))
#     path_z.write_text(json.dumps(mos_z_map, indent=2))
#     return path_centered, path_z

# # ---------------------------
# # Main
# # ---------------------------
# def main():
#     ap = argparse.ArgumentParser(description="Inter-rater analysis and MOS over N raters with per-rater bias correction + IAA metrics.")
#     ap.add_argument("raters", nargs="+", type=str, help="Paths to rater JSONs (>=2).")
#     ap.add_argument("--rater-names", nargs="*", default=None,
#                     help="Optional names for raters (same order as files). Defaults to file stems.")
#     ap.add_argument("--out", type=str, default="human_scores_analysis.csv",
#                     help="Output CSV with per-video MOS and per-rater corrected scores.")
#     ap.add_argument("--json-prefix", type=str, default=None,
#                     help="Filename prefix (no extension) for MOS JSONs. Defaults to stem of --out.")
#     ap.add_argument("--pairwise-out", type=str, default="pairwise_correlations.csv",
#                     help="Optional CSV for pairwise inter-rater correlations.")
#     ap.add_argument("--categorize", type=str, default="auto", choices=["auto", "int", "round", "bin"],
#                     help="How to convert raw scores to categorical labels for Fleiss/Matching.")
#     ap.add_argument("--bins", type=int, default=5, help="Number of bins if --categorize bin/auto chooses bin.")
#     args = ap.parse_args()

#     if len(args.raters) < 2:
#         raise SystemExit("Provide at least two rater JSON files.")

#     # Load all raters
#     if args.rater_names and (len(args.rater_names) != len(args.raters)):
#         raise SystemExit("--rater-names length must match number of rater files.")
#     dfs = []
#     for i, path in enumerate(args.raters):
#         name = (args.rater_names[i] if args.rater_names else Path(path).stem)
#         dfs.append(load_scores(path, rater_name=name))
#     all_raw = pd.concat(dfs, ignore_index=True)

#     # Pairwise correlations on RAW scores
#     pw = pairwise_raw_correlations(all_raw)
#     if not pw.empty:
#         print("\n[Pairwise inter-rater correlations on RAW scores]")
#         # Pretty: drop p-values and put N_common last
#         pretty = pw[["rater_i","rater_j","spearman_rho","pearson_r","N_common"]].copy()
#         print(pretty.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
#     else:
#         print("\n[Pairwise inter-rater correlations on RAW scores] No pairs with >=2 common videos.")

#     # Per-rater bias correction
#     corrected = per_rater_center_and_z(all_raw)

#     # MOS across all available raters (per video)
#     mos = compute_mos_over_all_raters(corrected)

#     # Summary
#     print("\n[Per-video MOS summary]")
#     summ = summarize_min_max(mos)
#     print(summ.to_string(index=False))

#     # ---------- IAA: Matching Ratio, Fleiss' κ, Krippendorff's α ----------
#     # Categorical view for Fleiss / Matching
#     df_cat, cat_mode = categorize_scores(all_raw, mode=args.categorize, bins=args.bins)
#     cat_pivot = df_cat.pivot(index="video", columns="rater", values="cat")
#     # require all raters for Fleiss / Matching
#     cat_all = cat_pivot[cat_pivot.notna().all(axis=1)].astype(int)

#     print(f"\n[IAA | Categorical using mode='{cat_mode}'"
#           + (f", bins={args.bins}" if cat_mode == "bin" else "") + "]")
#     if cat_all.empty or cat_all.shape[1] < 2:
#         print("  Not enough fully-overlapped items for categorical IAA (need videos rated by all raters).")
#         mr_pair, mr_all = np.nan, np.nan
#         kappa, P_bar, P_e = np.nan, np.nan, np.nan
#     else:
#         mr_pair, mr_all = matching_ratio_stats(cat_all)
#         kappa, P_bar, P_e = fleiss_kappa(cat_all)
#         print(f"  Matching Ratio (avg pairwise agreement): {mr_pair:.3f}")
#         print(f"  Matching Ratio (all-raters exact agree): {mr_all:.3f}")
#         print(f"  Fleiss' kappa κ: {kappa:.3f} (P̄={P_bar:.3f}, Pe={P_e:.3f})")
#     # Krippendorff's alpha (nominal) on the same categorical table (allows NaNs; here we used all-rated)
#     alpha_nom, Do_nom, De_nom = krippendorff_alpha_nominal(cat_all if not cat_all.empty else cat_pivot)
#     print(f"  Krippendorff's α (nominal): {alpha_nom:.3f}" if not np.isnan(alpha_nom) else "  Krippendorff's α (nominal): NaN")

#     # Krippendorff's alpha (interval) on raw scores (can handle missing)
#     raw_pivot = all_raw.pivot(index="video", columns="rater", values="raw_score")
#     alpha_int, Do_int, De_int = krippendorff_alpha_interval(raw_pivot)
#     print("\n[IAA | Continuous (interval) on raw scores]")
#     print(f"  Krippendorff's α (interval): {alpha_int:.3f}" if not np.isnan(alpha_int) else "  Krippendorff's α (interval): NaN")

#     # Build tidy table to export: each rater's corrected scores + MOS per video
#     tidy = (
#         corrected
#         .merge(mos, on="video", how="left")
#         .sort_values(["video", "rater"])
#         .reset_index(drop=True)
#     )

#     # Save CSVs
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     tidy.to_csv(out_path, index=False)
#     print(f"\nWrote per-rater & MOS CSV to: {out_path}")

#     if pw is not None and not pw.empty and args.pairwise_out:
#         pw_path = Path(args.pairwise_out)
#         pw_path.parent.mkdir(parents=True, exist_ok=True)
#         # Save the pretty (p-values removed)
#         pretty = pw[["rater_i","rater_j","spearman_rho","pearson_r","N_common"]]
#         pretty.to_csv(pw_path, index=False)
#         print(f"Wrote pairwise correlations CSV to: {pw_path}")

#     # Save MOS JSONs
#     if args.json_prefix is None:
#         json_prefix = out_path.with_suffix("")  # same folder, stem of CSV
#     else:
#         json_prefix = Path(args.json_prefix)
#         json_prefix.parent.mkdir(parents=True, exist_ok=True)

#     mos_centered_path, mos_z_path = save_mos_json(mos, json_prefix)
#     print(f"Wrote MOS (centered) JSON to: {mos_centered_path}")
#     print(f"Wrote MOS (z-scored) JSON to: {mos_z_path}")

#     # Save IAA summary
#     iaa_rows = [{
#         "categorize_mode": cat_mode,
#         "bins": args.bins if cat_mode == "bin" else None,
#         "matching_ratio_pairwise": mr_pair,
#         "matching_ratio_all_agree": mr_all,
#         "fleiss_kappa": kappa,
#         "fleiss_P_bar": P_bar,
#         "fleiss_Pe": P_e,
#         "krippendorff_alpha_nominal": alpha_nom,
#         "krippendorff_alpha_interval": alpha_int,
#     }]
#     iaa_df = pd.DataFrame(iaa_rows)
#     iaa_path = out_path.with_name(out_path.stem + "_iaa_summary.csv")
#     iaa_df.to_csv(iaa_path, index=False)
#     print(f"Wrote IAA summary CSV to: {iaa_path}")

# if __name__ == "__main__":
#     main()