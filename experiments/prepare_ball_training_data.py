"""Prepare ball trajectory training data from external SoccerNet-v3D."""

import argparse
from ast import literal_eval
import numpy as np
import pandas as pd
from pathlib import Path


def extract_soccernet_trajectories(csv_path, min_length=10):
    """Extract ball trajectories from SoccerNet-v3D CSV.

    Args:
        csv_path: Path to SNv3D.csv
        min_length: Minimum trajectory length to keep

    Returns:
        List of (N, 2) arrays of normalized 2D ball positions
    """
    print(f"Loading SoccerNet-v3D from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter to training set with valid 3D positions
    df = df[df['set'] == 'train']
    df = df[df['ball_3D'].notna()]

    print(f"Found {len(df)} training samples with 3D ball positions")

    # Extract 3D positions and convert to 2D (using x, y)
    trajectories = []
    current_match = None
    current_action = None
    current_traj = []

    for idx, row in df.iterrows():
        match_id = f"{row['league']}_{row['season']}_{row['match']}"
        action_id = f"{row['main_action']}_{row['action']}"

        # Start new trajectory on match/action change
        if match_id != current_match or action_id != current_action:
            if len(current_traj) >= min_length:
                trajectories.append(np.array(current_traj))
            current_traj = []
            current_match = match_id
            current_action = action_id

        # Parse 3D position
        try:
            ball_3d_str = row['ball_3D']
            if isinstance(ball_3d_str, str) and ball_3d_str.startswith('['):
                # Remove array() wrapper and parse
                ball_3d_str = ball_3d_str.replace('array(', '').replace(')', '')
                ball_3d = literal_eval(ball_3d_str)[0]
                # Normalize to [0, 1] range based on typical pitch dimensions
                # Assume standard pitch: 105m x 68m
                x_norm = (ball_3d[0] + 52.5) / 105.0
                y_norm = (ball_3d[1] + 34.0) / 68.0
                current_traj.append([x_norm, y_norm])
        except Exception as e:
            continue

    # Add final trajectory
    if len(current_traj) >= min_length:
        trajectories.append(np.array(current_traj))

    print(f"Extracted {len(trajectories)} trajectories")
    return trajectories


def create_synthetic_gaps(trajectories, gap_probabilities=[0.1, 0.2, 0.3]):
    """Create synthetic training data by randomly masking positions.

    Args:
        trajectories: List of complete trajectories
        gap_probabilities: List of gap ratios to create

    Returns:
        augmented_trajectories: List with synthetic gaps added
    """
    augmented = []

    for traj in trajectories:
        # Keep original
        augmented.append(traj.copy())

        # Create versions with synthetic gaps
        for gap_prob in gap_probabilities:
            traj_with_gaps = traj.copy()
            n_mask = int(len(traj) * gap_prob)
            if n_mask > 0:
                mask_indices = np.random.choice(len(traj), n_mask, replace=False)
                traj_with_gaps[mask_indices] = np.nan
                augmented.append(traj_with_gaps)

    print(f"Augmented {len(trajectories)} → {len(augmented)} trajectories")
    return augmented


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soccernet-csv', type=Path,
                        default='external_data/SNv3D.csv',
                        help='Path to SoccerNet-v3D CSV')
    parser.add_argument('--output', type=Path,
                        default='external_data/soccernet_ball_training_data.npy',
                        help='Output file for training data')
    parser.add_argument('--augment', action='store_true',
                        help='Add synthetic gaps for data augmentation')
    args = parser.parse_args()

    if not args.soccernet_csv.is_file():
        parser.error(f"SoccerNet CSV not found: {args.soccernet_csv}")
    trajectories = extract_soccernet_trajectories(args.soccernet_csv)
    if not trajectories:
        parser.error("no external trajectories found")

    # Augment with synthetic gaps
    if args.augment:
        trajectories = create_synthetic_gaps(trajectories)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch = np.empty(len(trajectories), dtype=object)
    batch[:] = trajectories
    np.save(args.output, batch, allow_pickle=True)

    print(f"\nSaved {len(trajectories)} trajectories to {args.output}")
    print(f"Total frames: {sum(len(t) for t in trajectories)}")


if __name__ == '__main__':
    main()
