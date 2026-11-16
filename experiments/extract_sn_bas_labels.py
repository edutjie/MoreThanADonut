#!/usr/bin/env python3
"""
Extract SN-BAS-2025 events as training labels for possession prediction.

Strategy:
1. Use dead ball events (Throw-in, Free kick, Corner, etc.) to create D labels
2. Use team information (left/right) around live events for H/A labels
3. Build temporal features around these events for GBM training
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import pickle


# Events that clearly indicate dead ball periods
DEAD_BALL_EVENTS = {
    'Throw-in',
    'Direct free-kick',
    'Indirect free-kick',
    'Corner',
    'Goal kick',
    'Kick-off',
    'Penalty',
    'Offside',
    'Ball out of play',
}

# Events that indicate live ball
LIVE_BALL_EVENTS = {
    'Shots on target',
    'Shots off target',
    'Clearance',
    'Goal',
}


def extract_event_labels(label_dir='external_data/SN-BAS-2025/ExtraLabelsActionSpotting500games'):
    """
    Extract frame-level possession labels from SN-BAS-2025 events.

    Returns:
        list of dicts: [{
            'match': str,
            'half': int,
            'time_ms': int,
            'label': 'H'/'A'/'D',
            'event_type': str,
            'confidence': float,
        }, ...]
    """
    label_dir = Path(label_dir)
    json_files = list(label_dir.rglob('Labels-v2.json'))

    print(f'Processing {len(json_files)} matches...')

    labels = []
    event_counts = Counter()

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        match_path = json_file.parent
        match_id = str(match_path.relative_to(label_dir))

        for anno in data.get('annotations', []):
            event_type = anno['label']
            event_counts[event_type] += 1

            # Parse game time "1 - 00:19" -> half=1, time_ms=19000
            game_time = anno['gameTime']
            parts = game_time.split(' - ')
            if len(parts) != 2:
                continue

            half = int(parts[0])
            time_str = parts[1]  # "00:19" or "01:23"
            time_parts = time_str.split(':')
            if len(time_parts) != 2:
                continue

            minutes, seconds = map(int, time_parts)
            time_ms = (minutes * 60 + seconds) * 1000

            # Position is also in milliseconds
            position_ms = int(anno['position'])

            team = anno['team']  # 'left', 'right', 'not applicable'

            # Determine possession label
            if event_type in DEAD_BALL_EVENTS:
                label = 'D'
                confidence = 1.0
            elif event_type in LIVE_BALL_EVENTS:
                # Live ball with team info
                if team == 'left':
                    label = 'H'  # Assume left = home
                elif team == 'right':
                    label = 'A'  # Assume right = away
                else:
                    continue  # Skip if no team info
                confidence = 0.8  # Lower confidence for inferred possession
            else:
                # Events like Substitution, Yellow card, etc.
                continue

            labels.append({
                'match': match_id,
                'half': half,
                'time_ms': position_ms,
                'label': label,
                'event_type': event_type,
                'team': team,
                'confidence': confidence,
            })

    print(f'\nExtracted {len(labels)} possession labels from {len(json_files)} matches')
    print(f'\nEvent distribution:')
    for event, count in event_counts.most_common(10):
        print(f'  {event:30s}: {count:6d}')

    # Count label distribution
    label_counts = Counter(l['label'] for l in labels)
    print(f'\nLabel distribution:')
    for label in ['H', 'A', 'D']:
        count = label_counts[label]
        pct = 100 * count / len(labels) if labels else 0
        print(f'  {label}: {count:6d} ({pct:5.1f}%)')

    return labels


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-dir', type=Path,
                       default='external_data/SN-BAS-2025/ExtraLabelsActionSpotting500games',
                       help='SN-BAS-2025 label directory')
    parser.add_argument('--output', type=Path,
                       default='external_data/sn_bas_2025_labels.pkl',
                       help='Output pickle file')
    args = parser.parse_args()

    labels = extract_event_labels(args.label_dir)

    # Save to pickle
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(labels, f)

    print(f'\nSaved {len(labels)} labels to {args.output}')
    print(f'File size: {args.output.stat().st_size / 1024:.1f} KB')


if __name__ == '__main__':
    main()
