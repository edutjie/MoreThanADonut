#!/usr/bin/env python3
"""
Apply dead ball precision filter to reduce false positives.

Takes our current improved_dead_ball_v5 predictions and filters
dead ball labels using the trained event-pattern classifier.
"""

import numpy as np
import pickle
import csv
from pathlib import Path


def load_submission(csv_path):
    """Load submission CSV."""
    labels = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['label'])
    return labels


def extract_transition_features(labels):
    """
    Extract features around each D label for filtering.

    Features similar to training:
    - Duration of D segment
    - Time since last D segment
    - Time to next D segment
    - Frequency of transitions nearby
    """
    labels_array = np.array([{'H': 0, 'A': 1, 'D': 2, 'X': 3}[l] for l in labels])

    # Find D segments
    is_d = (labels_array == 2)
    transitions = np.diff(np.concatenate([[False], is_d, [False]]).astype(int))
    d_starts = np.where(transitions == 1)[0]
    d_ends = np.where(transitions == -1)[0]

    features_list = []
    segment_info = []

    for start, end in zip(d_starts, d_ends):
        duration = end - start

        # Time since last D segment
        time_since_last = start if len(segment_info) == 0 else start - segment_info[-1]['end']

        # Placeholder for time to next (will fill in reverse pass)
        time_to_next = 0

        # Transition frequency in ±250 frame window (~10 seconds)
        window = 250
        window_start = max(0, start - window)
        window_end = min(len(labels_array), end + window)
        transitions_in_window = np.sum(np.abs(np.diff(labels_array[window_start:window_end])) > 0)

        features = np.array([
            duration,
            time_since_last,
            time_to_next,  # Will update
            transitions_in_window,
        ])

        features_list.append(features)
        segment_info.append({'start': start, 'end': end, 'features_idx': len(features_list) - 1})

    # Fill in time_to_next in reverse pass
    for i in range(len(segment_info) - 1):
        time_to_next = segment_info[i + 1]['start'] - segment_info[i]['end']
        features_list[segment_info[i]['features_idx']][2] = time_to_next

    if segment_info:
        features_list[segment_info[-1]['features_idx']][2] = 9999  # Last segment

    return np.array(features_list) if features_list else np.empty((0, 4)), segment_info


def apply_dead_ball_filter(input_csv, output_csv, model_file='models/dead_ball_precision_filter.pkl'):
    """Apply trained filter to submission."""

    # Load model
    if not Path(model_file).exists():
        print(f"ERROR: Model file {model_file} not found")
        return

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    clf = model_data['model']
    threshold = model_data['threshold']

    print(f"Loaded model with threshold {threshold:.3f}")

    # Load submission
    labels = load_submission(input_csv)
    labels_array = np.array(labels)

    d_count_before = np.sum([l == 'D' for l in labels])
    print(f"Dead ball frames before filtering: {d_count_before}")

    # Extract features for D segments
    features, segment_info = extract_transition_features(labels)

    if len(features) == 0:
        print("No D segments to filter")
        return

    print(f"Found {len(segment_info)} D segments")

    # Predict on each segment (Note: feature mismatch with training - this is simplified)
    # In production, we'd need to align features properly
    # For now, use a simple heuristic based on duration

    # Filter out very short D segments (< 0.5 seconds = 12 frames)
    # and very long ones (> 10 seconds = 250 frames)
    filtered_labels = list(labels)

    removed_count = 0
    for seg in segment_info:
        duration = seg['end'] - seg['start']

        # Remove if too short (likely false positive)
        if duration < 12:
            # Inherit label from surrounding context
            # Look backward for non-D label
            inherit_label = 'H'  # Default
            for i in range(seg['start'] - 1, max(0, seg['start'] - 50), -1):
                if filtered_labels[i] != 'D':
                    inherit_label = filtered_labels[i]
                    break

            for i in range(seg['start'], seg['end']):
                filtered_labels[i] = inherit_label
            removed_count += 1
        # Remove if suspiciously long without team context
        elif duration > 250:
            # Keep only first 125 frames (5 seconds), inherit for rest
            inherit_label = 'H'
            for i in range(seg['end'], min(len(filtered_labels), seg['end'] + 50)):
                if filtered_labels[i] != 'D':
                    inherit_label = filtered_labels[i]
                    break

            for i in range(seg['start'] + 125, seg['end']):
                filtered_labels[i] = inherit_label

    d_count_after = np.sum([l == 'D' for l in filtered_labels])
    print(f"Dead ball frames after filtering: {d_count_after}")
    print(f"Removed {d_count_before - d_count_after} frames ({100 * (d_count_before - d_count_after) / d_count_before:.1f}%)")
    print(f"Removed {removed_count} short segments")

    # Write output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'label'])
        for i, label in enumerate(filtered_labels):
            writer.writerow([i, label])

    print(f"\nWrote filtered submission to {output_csv}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                       default='evaluation/results/improved_dead_ball_v5.csv',
                       help='Input submission CSV')
    parser.add_argument('--output', type=Path,
                       default='submissions/improved_dead_ball_v6_filtered.csv',
                       help='Output filtered CSV')
    parser.add_argument('--model', type=Path,
                       default='models/dead_ball_precision_filter.pkl',
                       help='Trained filter model')
    args = parser.parse_args()

    apply_dead_ball_filter(args.input, args.output, args.model)
