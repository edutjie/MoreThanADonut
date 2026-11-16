#!/usr/bin/env python3
"""
Build possession training dataset from SN-BAS-2025.

Combines:
1. Labels-ball.json: ball action events with team possession (H/A)
2. Labels-v2.json: dead ball events (D)

Creates frame-level labels at 2 FPS for training.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import pickle


def load_match_labels(match_dir):
    """
    Load both ball actions and dead ball events for a match.

    Returns:
        dict: {
            'ball_actions': list of dicts,
            'dead_ball_events': list of dicts,
            'match_id': str,
        }
    """
    match_dir = Path(match_dir)

    # Load ball possession actions
    ball_json = match_dir / 'Labels-ball.json'
    if not ball_json.exists():
        return None

    with open(ball_json) as f:
        ball_data = json.load(f)

    ball_actions = []
    for anno in ball_data.get('annotations', []):
        position_ms = int(anno['position'])
        team = anno['team']

        if team in ['left', 'right']:
            ball_actions.append({
                'time_ms': position_ms,
                'team': team,
                'action': anno['label'],
            })

    # Load dead ball events from action spotting labels
    # Path pattern: ExtraLabelsActionSpotting500games/<league>/<season>/<match>/Labels-v2.json
    match_name = match_dir.name
    season = match_dir.parent.name
    league = match_dir.parent.parent.name

    labels_v2_path = Path('external_data/SN-BAS-2025/ExtraLabelsActionSpotting500games') / league / season / match_name / 'Labels-v2.json'

    dead_ball_events = []
    if labels_v2_path.exists():
        with open(labels_v2_path) as f:
            event_data = json.load(f)

        DEAD_BALL_EVENT_TYPES = {
            'Throw-in', 'Direct free-kick', 'Indirect free-kick',
            'Corner', 'Goal kick', 'Kick-off', 'Penalty', 'Offside',
            'Ball out of play',
        }

        for anno in event_data.get('annotations', []):
            if anno['label'] in DEAD_BALL_EVENT_TYPES:
                position_ms = int(anno['position'])
                dead_ball_events.append({
                    'time_ms': position_ms,
                    'event_type': anno['label'],
                })

    return {
        'ball_actions': ball_actions,
        'dead_ball_events': dead_ball_events,
        'match_id': f"{league}/{season}/{match_name}",
    }


def create_frame_level_labels(match_labels, fps=2, match_duration_ms=6000000):
    """
    Create frame-level H/A/D labels from event sequences.

    Strategy:
    - Start with all frames as X (unknown)
    - Label frames near ball actions with team possession (H/A)
    - Override with D during dead ball events
    - Interpolate between consecutive actions of same team

    Args:
        fps: frames per second (2 for SoccerNet features)
        match_duration_ms: ~100 minutes = 6000000 ms

    Returns:
        numpy array of shape (n_frames,) with labels 0=H, 1=A, 2=D, 3=X
    """
    frame_ms = 1000 / fps  # milliseconds per frame at 2 FPS
    n_frames = int(match_duration_ms / frame_ms)

    labels = np.full(n_frames, 3, dtype=np.int8)  # Start with X (unknown)

    # First pass: label ball actions as H/A
    ball_actions = sorted(match_labels['ball_actions'], key=lambda x: x['time_ms'])

    for i, action in enumerate(ball_actions):
        frame = int(action['time_ms'] / frame_ms)
        if frame >= n_frames:
            continue

        # Map team to H/A
        label = 0 if action['team'] == 'left' else 1  # 0=H(left), 1=A(right)

        # Label this frame and nearby frames (~2 seconds = 4 frames)
        start = max(0, frame - 2)
        end = min(n_frames, frame + 3)
        labels[start:end] = label

        # If next action is same team within 5 seconds, interpolate
        if i + 1 < len(ball_actions):
            next_action = ball_actions[i + 1]
            next_frame = int(next_action['time_ms'] / frame_ms)

            if (next_action['team'] == action['team'] and
                next_frame - frame < 5 * fps and
                next_frame < n_frames):
                # Fill the gap
                labels[frame:next_frame+1] = label

    # Second pass: override with dead ball events
    for event in match_labels['dead_ball_events']:
        frame = int(event['time_ms'] / frame_ms)
        if frame >= n_frames:
            continue

        # Dead ball period: event frame ± 3 seconds
        start = max(0, frame - int(3 * fps))
        end = min(n_frames, frame + int(3 * fps))
        labels[start:end] = 2  # 2 = D (dead ball)

    return labels


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-matches', type=int, default=50,
                       help='Maximum matches to process')
    parser.add_argument('--output', type=Path,
                       default='external_data/sn_bas_2025_training_labels.pkl',
                       help='Output pickle file')
    args = parser.parse_args()

    # Find all matches with videos
    video_root = Path('external_data/SN-BAS-2025')
    match_dirs = []

    for league_dir in video_root.glob('england_*/'):
        if not league_dir.is_dir() or 'ActionSpotting' in league_dir.name:
            continue
        for season_dir in league_dir.glob('*/'):
            if not season_dir.is_dir():
                continue
            for match_dir in season_dir.glob('*/'):
                if (match_dir / 'Labels-ball.json').exists():
                    match_dirs.append(match_dir)
                    if len(match_dirs) >= args.max_matches:
                        break
            if len(match_dirs) >= args.max_matches:
                break
        if len(match_dirs) >= args.max_matches:
            break

    print(f"Processing {len(match_dirs)} matches...")

    training_data = []
    label_counts = Counter()

    for match_dir in match_dirs:
        match_labels = load_match_labels(match_dir)
        if not match_labels:
            continue

        frame_labels = create_frame_level_labels(match_labels)

        # Count labels
        unique, counts = np.unique(frame_labels, return_counts=True)
        for label, count in zip(unique, counts):
            label_counts[label] += count

        training_data.append({
            'match_id': match_labels['match_id'],
            'labels': frame_labels,
            'n_ball_actions': len(match_labels['ball_actions']),
            'n_dead_ball_events': len(match_labels['dead_ball_events']),
        })

        if len(training_data) % 10 == 0:
            print(f"  Processed {len(training_data)} matches...")

    print(f"\nCreated training labels for {len(training_data)} matches")
    print(f"\nLabel distribution:")
    label_names = ['H (left team)', 'A (right team)', 'D (dead ball)', 'X (unknown)']
    total = sum(label_counts.values())
    for i, name in enumerate(label_names):
        count = label_counts[i]
        pct = 100 * count / total if total > 0 else 0
        print(f"  {name:20s}: {count:10d} frames ({pct:5.1f}%)")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"\nSaved to {args.output}")
    print(f"File size: {args.output.stat().st_size / (1024*1024):.1f} MB")


if __name__ == '__main__':
    main()
