"""
Process raw human scores following the three-stage filtering procedure:
1. Repeated-video consistency filtering
2. Subject rejection (R1, R2 statistics)
3. Inter-rater reliability filtering (Spearman correlation)
4. Compute MOS and z-score normalize
"""

import csv
import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

def load_raw_data(filepath: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load CSV data and organize by participant."""
    rows = []
    participant_data = defaultdict(list)
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['action_consistency'] = float(row['action_consistency'])
            row['physical_plausibility'] = float(row['physical_plausibility'])
            rows.append(row)
            participant_data[row['participant_id']].append(row)
    
    return rows, dict(participant_data)

def find_repeated_videos(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Find videos that appear multiple times (repeated videos)."""
    video_counts = defaultdict(list)
    for row in rows:
        video_counts[row['video_id']].append(row)
    
    # Find videos that appear more than once
    repeated = {vid: entries for vid, entries in video_counts.items() if len(entries) > 1}
    return repeated

def stage1_repeated_video_consistency(participant_data: Dict[str, List[Dict]], 
                                     all_rows: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Stage 1: Repeated-video consistency filtering.
    Keep only participants whose scores on duplicated videos fall within the 95th percentile.
    For each participant, compute std of ratings across all repeated videos (both AC and TC).
    """
    participant_stds = {}
    for participant_id, rows in participant_data.items():
        # Find videos that appear multiple times for this participant
        video_counts = defaultdict(list)
        for row in rows:
            video_counts[row['video_id']].append(row)
        
        # Collect all stds for repeated videos
        repeated_stds = []
        for video_id, ratings in video_counts.items():
            if len(ratings) > 1:
                # Compute std for AC and TC separately
                ac_scores = [r['action_consistency'] for r in ratings]
                tc_scores = [r['physical_plausibility'] for r in ratings]
                if len(ac_scores) > 1:
                    repeated_stds.append(np.std(ac_scores))
                if len(tc_scores) > 1:
                    repeated_stds.append(np.std(tc_scores))
        
        if repeated_stds:
            # Mean std across all repeated videos for this participant
            participant_stds[participant_id] = np.mean(repeated_stds)
        else:
            # If no repeated videos rated, assign high std (will likely be rejected)
            participant_stds[participant_id] = 999.0
    
    # Keep participants within 95th percentile (lower std = more consistent)
    if participant_stds:
        threshold = np.percentile(list(participant_stds.values()), 95)
        filtered = {pid: rows for pid, rows in participant_data.items() 
                   if participant_stds[pid] <= threshold}
        print(f"Stage 1: Kept {len(filtered)}/{len(participant_data)} participants (95th percentile threshold: {threshold:.4f})")
        print(f"   Mean std of kept participants: {np.mean([participant_stds[pid] for pid in filtered.keys()]):.4f}")
        return filtered
    else:
        return participant_data

def stage2_subject_rejection(participant_data: Dict[str, List[Dict]], 
                            metric: str = 'action_consistency') -> Dict[str, List[Dict]]:
    """
    Stage 2: Subject rejection using R1 and R2 statistics.
    Reject participants with R1 > 0.05 and R2 < 0.3, or those who rated fewer than 10 videos.
    """
    # Compute population mean for this metric
    all_scores = []
    for rows in participant_data.values():
        all_scores.extend([r[metric] for r in rows])
    pop_mean = np.mean(all_scores)
    pop_std = np.std(all_scores)
    
    # Determine threshold based on kurtosis
    from scipy.stats import kurtosis
    kurt = kurtosis(all_scores)
    if kurt > 3:  # Leptokurtic
        threshold = np.sqrt(20) * pop_std
    else:
        threshold = 2 * pop_std
    
    valid_participants = {}
    for participant_id, rows in participant_data.items():
        if len(rows) < 10:
            continue
        
        scores = np.array([r[metric] for r in rows])
        Pi = np.sum(scores > pop_mean + threshold)
        Qi = np.sum(scores < pop_mean - threshold)
        Ni = len(scores)
        
        R1 = (Pi + Qi) / Ni if Ni > 0 else 1.0
        R2 = abs(Pi - Qi) / (Pi + Qi) if (Pi + Qi) > 0 else 0.0
        
        # Reject if R1 > 0.05 and R2 < 0.3
        if not (R1 > 0.05 and R2 < 0.3):
            valid_participants[participant_id] = rows
    
    print(f"Stage 2 ({metric}): Kept {len(valid_participants)}/{len(participant_data)} participants")
    return valid_participants

def stage3_inter_rater_reliability(participant_data: Dict[str, List[Dict]], 
                                   metric: str = 'action_consistency',
                                   correlation_threshold: float = 0.55) -> Dict[str, List[Dict]]:
    """
    Stage 3: Inter-rater reliability filtering.
    Compute Spearman correlation between each participant's ratings and aggregated mean ratings
    of all other participants. Exclude raters with ρ < threshold.
    """
    valid_participants = {}
    correlations = []
    
    for participant_id, rows in participant_data.items():
        # Get this participant's ratings
        participant_ratings = {r['video_id']: r[metric] for r in rows}
        
        if len(participant_ratings) < 3:  # Need at least 3 videos for correlation
            continue
        
        # Compute mean ratings excluding this participant
        other_ratings = defaultdict(list)
        for other_pid, other_rows in participant_data.items():
            if other_pid == participant_id:
                continue
            for row in other_rows:
                other_ratings[row['video_id']].append(row[metric])
        
        other_mean_ratings = {vid: np.mean(ratings) for vid, ratings in other_ratings.items() 
                             if len(ratings) > 0}
        
        # Align ratings for common videos
        this_scores = []
        other_scores = []
        for vid in participant_ratings.keys():
            if vid in other_mean_ratings:
                this_scores.append(participant_ratings[vid])
                other_scores.append(other_mean_ratings[vid])
        
        if len(this_scores) >= 3:
            corr, p_value = spearmanr(this_scores, other_scores)
            if not np.isnan(corr):
                correlations.append(corr)
                if corr >= correlation_threshold:
                    valid_participants[participant_id] = rows
    
    if correlations:
        print(f"Stage 3 ({metric}): Kept {len(valid_participants)}/{len(participant_data)} participants (threshold: {correlation_threshold})")
        print(f"   Mean correlation of kept participants: {np.mean([c for c in correlations if c >= correlation_threshold]):.4f}")
    else:
        print(f"Stage 3 ({metric}): No valid correlations computed")
    
    return valid_participants

def compute_mos_and_normalize(participant_data: Dict[str, List[Dict]], 
                              metric: str = 'action_consistency') -> Dict[str, float]:
    """
    Compute Mean Opinion Score (MOS) for each video and z-score normalize.
    """
    # Collect all ratings by video
    video_ratings = defaultdict(list)
    for rows in participant_data.values():
        for row in rows:
            video_ratings[row['video_id']].append(row[metric])
    
    # Compute MOS
    video_mos = {vid: np.mean(ratings) for vid, ratings in video_ratings.items()}
    
    # Z-score normalize
    mos_values = list(video_mos.values())
    mean_mos = np.mean(mos_values)
    std_mos = np.std(mos_values)
    
    video_mos_normalized = {
        vid: (mos - mean_mos) / std_mos if std_mos > 0 else 0.0
        for vid, mos in video_mos.items()
    }
    
    return video_mos_normalized

def normalize_video_id(video_id: str) -> str:
    """
    Normalize video ID to match expected format.
    This matches the _norm_name function in utils.py:
    - Extract basename without extension
    - Replace "_videos_" with "_"
    - Replace "videos_" with ""
    - Replace "_video_" with "_"
    - Keep .mp4 extension for output
    """
    import os
    # Extract just the filename, removing path
    if '/' in video_id:
        video_id = video_id.split('/')[-1]
    
    # Get stem (without extension)
    stem = os.path.splitext(video_id)[0]
    
    # Apply same normalization as _norm_name in utils.py
    stem = stem.replace("_videos_", "_")
    stem = stem.replace("videos_", "")
    stem = stem.replace("_video_", "_")
    
    # Add .mp4 extension back (output format expects .mp4)
    return stem + ".mp4"

def main():
    input_file = 'raw_humans.json'
    output_file = 'human_scores.json'
    
    print("Loading raw data...")
    all_rows, participant_data = load_raw_data(input_file)
    print(f"Loaded {len(all_rows)} ratings from {len(participant_data)} participants")
    
    # Stage 1: Repeated-video consistency filtering
    print("\n=== Stage 1: Repeated-video consistency filtering ===")
    filtered_stage1 = stage1_repeated_video_consistency(participant_data, all_rows)
    
    # Stage 2: Subject rejection (separate for AC and TC)
    print("\n=== Stage 2: Subject rejection ===")
    filtered_stage2_ac = stage2_subject_rejection(filtered_stage1, metric='action_consistency')
    filtered_stage2_tc = stage2_subject_rejection(filtered_stage1, metric='physical_plausibility')
    
    # Stage 3: Inter-rater reliability (separate for AC and TC)
    print("\n=== Stage 3: Inter-rater reliability filtering ===")
    filtered_stage3_ac = stage3_inter_rater_reliability(filtered_stage2_ac, metric='action_consistency', correlation_threshold=0.55)
    filtered_stage3_tc = stage3_inter_rater_reliability(filtered_stage2_tc, metric='physical_plausibility', correlation_threshold=0.55)
    
    # Compute MOS and normalize
    print("\n=== Computing MOS and z-score normalization ===")
    video_mos_ac = compute_mos_and_normalize(filtered_stage3_ac, metric='action_consistency')
    video_mos_tc = compute_mos_and_normalize(filtered_stage3_tc, metric='physical_plausibility')
    
    # Combine results
    all_videos = set(video_mos_ac.keys()) | set(video_mos_tc.keys())
    output = {}
    for video_id in all_videos:
        normalized_id = normalize_video_id(video_id)
        # Only include videos that have scores from valid participants
        ac_score = video_mos_ac.get(video_id)
        tc_score = video_mos_tc.get(video_id)
        if ac_score is not None or tc_score is not None:
            output[normalized_id] = {
                'ac': float(ac_score) if ac_score is not None else 0.0,
                'tc': float(tc_score) if tc_score is not None else 0.0
            }
    
    # Save output
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {len(output)} videos to {output_file}")
    print(f"   Action Consistency: {len(filtered_stage3_ac)} valid participants, {len(video_mos_ac)} videos")
    print(f"   Temporal Coherence: {len(filtered_stage3_tc)} valid participants, {len(video_mos_tc)} videos")
    
    # Print sample output for verification
    sample_keys = list(output.keys())[:3]
    print(f"\n   Sample output keys: {sample_keys}")
    for key in sample_keys:
        print(f"      {key}: ac={output[key]['ac']:.4f}, tc={output[key]['tc']:.4f}")

if __name__ == '__main__':
    main()

