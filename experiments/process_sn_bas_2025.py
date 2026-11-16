#!/usr/bin/env python3
"""
Extract event labels from SN-BAS-2025 and align with tactical tracking features.

SN-BAS-2025 provides frame-accurate event labels:
- Free Kick, Throw In, Goal, Clearance, etc.
- We'll use these to train a GBM model on external data (rule-compliant)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_sn_bas_events(data_dir='external_data/SN-BAS-2025'):
    """
    Load all event annotations from SN-BAS-2025.

    Returns:
        list of dicts: [{
            'match': str,
            'half': int,
            'frame': int,
            'event_type': str,
            'team': str,
        }, ...]
    """
    data_dir = Path(data_dir)

    events = []

    # Find all annotation JSON files
    annotation_files = list(data_dir.rglob('**/Labels*.json'))

    print(f"Found {len(annotation_files)} annotation files")

    for json_file in annotation_files:
        with open(json_file) as f:
            data = json.load(f)

        # Extract match and half info from path
        # Typical structure: league/season/match/half/Labels-v3.json
        parts = json_file.relative_to(data_dir).parts
        match_id = '/'.join(parts[:-2])  # e.g., "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"
        half = parts[-2]  # e.g., "1" or "2"

        # Parse annotations
        for anno in data.get('annotations', []):
            events.append({
                'match': match_id,
                'half': half,
                'frame': anno.get('gameTime', 0),  # Frame number
                'event_type': anno.get('label', 'unknown'),
                'team': anno.get('team', 'unknown'),
            })

    print(f"Loaded {len(events)} total events")
    return events


def analyze_event_distribution(events):
    """Print statistics about event types."""
    event_counts = defaultdict(int)

    for event in events:
        event_counts[event['event_type']] += 1

    print("\nEvent type distribution:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type:30s}: {count:6d}")

    return event_counts


def map_events_to_possession_labels(events):
    """
    Map SN-BAS-2025 events to possession state labels (H/A/D).

    Dead ball events: Throw In, Free Kick, Goal Kick, Corner, Penalty
    Live ball events: Ball possession changes, passes, shots

    Returns:
        dict: {(match, half, frame): 'H'/'A'/'D', ...}
    """

    # Events that clearly indicate dead ball
    DEAD_BALL_EVENTS = {
        'Throw-in',
        'Goal kick',
        'Corner',
        'Kick-off',
        'Free kick',
        'Penalty',
        'Offside',
    }

    labels = {}

    for event in events:
        key = (event['match'], event['half'], event['frame'])

        if event['event_type'] in DEAD_BALL_EVENTS:
            labels[key] = 'D'
        # Could also map team-specific events to H/A if we had home/away info

    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                       default='external_data/SN-BAS-2025',
                       help='SN-BAS-2025 data directory')
    args = parser.parse_args()

    # Load and analyze events
    events = load_sn_bas_events(args.data_dir)
    event_counts = analyze_event_distribution(events)

    # Map to possession labels
    labels = map_events_to_possession_labels(events)
    print(f"\nMapped {len(labels)} event frames to possession labels")

    # Count label distribution
    label_counts = defaultdict(int)
    for label in labels.values():
        label_counts[label] += 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count:6d} ({100*count/len(labels):.1f}%)")


if __name__ == '__main__':
    import argparse
    main()
