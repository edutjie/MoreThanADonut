#!/usr/bin/env python3
"""
Train dead ball precision filter using SN-BAS-2025 event patterns.

Problem: Our current heuristic labels 73K frames as D, but GT has only 1.5K (99% FP rate)
Solution: Learn patterns around 43,959 real dead ball events to filter false positives

Approach:
1. Extract temporal features around each SN-BAS-2025 dead ball event (±5s window)
2. Create negative samples from non-dead-ball periods
3. Train binary classifier: real_dead_ball vs false_positive
4. Apply to our BAR-ATH predictions to filter spurious D labels
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt


def build_event_pattern_features(events, window_frames=10):
    """
    Build features around dead ball events.

    Features:
    - Event type (one-hot)
    - Time since last event
    - Time to next event
    - Event frequency in window
    - Sequence patterns (event before, event after)

    Args:
        events: list of {time_ms, event_type, team}
        window_frames: frames to look before/after (at 2 FPS, 10 frames = 5 seconds)

    Returns:
        features: numpy array (n_events, n_features)
        labels: numpy array (n_events,) - 1 for dead ball, 0 for live
    """
    # Dead ball event types
    DEAD_BALL_EVENTS = {
        'Throw-in', 'Direct free-kick', 'Indirect free-kick',
        'Corner', 'Goal kick', 'Kick-off', 'Penalty', 'Offside',
        'Ball out of play',
    }

    # Sort events by time
    events = sorted(events, key=lambda x: x['time_ms'])

    features_list = []
    labels_list = []

    for i, event in enumerate(events):
        is_dead_ball = event['event_type'] in DEAD_BALL_EVENTS

        # Feature 1-9: Event type one-hot (top 9 types)
        event_onehot = np.zeros(9)
        event_types_ordered = ['Throw-in', 'Ball out of play', 'Foul',
                               'Indirect free-kick', 'Clearance', 'Shots on target',
                               'Shots off target', 'Corner', 'Goal kick']
        if event['event_type'] in event_types_ordered:
            event_onehot[event_types_ordered.index(event['event_type'])] = 1

        # Feature 10: Time since last event (seconds)
        time_since_last = 999.0
        if i > 0:
            time_since_last = min((event['time_ms'] - events[i-1]['time_ms']) / 1000.0, 999.0)

        # Feature 11: Time to next event (seconds)
        time_to_next = 999.0
        if i < len(events) - 1:
            time_to_next = min((events[i+1]['time_ms'] - event['time_ms']) / 1000.0, 999.0)

        # Feature 12: Event frequency in ±5 second window
        window_ms = 5000
        events_in_window = sum(1 for e in events
                              if abs(e['time_ms'] - event['time_ms']) <= window_ms)

        # Feature 13-14: Previous event type (binary: dead_ball, live_ball)
        prev_is_dead = 0.0
        prev_is_live = 0.0
        if i > 0:
            prev_is_dead = 1.0 if events[i-1]['event_type'] in DEAD_BALL_EVENTS else 0.0
            prev_is_live = 1.0 - prev_is_dead

        # Feature 15-16: Next event type (binary)
        next_is_dead = 0.0
        next_is_live = 0.0
        if i < len(events) - 1:
            next_is_dead = 1.0 if events[i+1]['event_type'] in DEAD_BALL_EVENTS else 0.0
            next_is_live = 1.0 - next_is_dead

        # Combine features
        features = np.concatenate([
            event_onehot,
            [time_since_last, time_to_next, events_in_window],
            [prev_is_dead, prev_is_live, next_is_dead, next_is_live],
        ])

        features_list.append(features)
        labels_list.append(1 if is_dead_ball else 0)

    return np.array(features_list), np.array(labels_list)


def train_dead_ball_filter():
    """Train classifier to distinguish real dead balls from false positives."""

    # Load SN-BAS-2025 extracted labels
    labels_file = Path('external_data/sn_bas_2025_labels.pkl')
    if not labels_file.exists():
        print(f"ERROR: {labels_file} not found. Run extract_sn_bas_labels.py first.")
        return None

    with open(labels_file, 'rb') as f:
        sn_labels = pickle.load(f)

    print(f"Loaded {len(sn_labels)} event labels from SN-BAS-2025")

    # Build features
    print("Building event pattern features...")
    X, y = build_event_pattern_features(sn_labels)

    print(f"Feature matrix: {X.shape}")
    print(f"Class distribution: {Counter(y)}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train GBM
    print("\nTraining gradient boosting classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\nTest set performance:")
    print(classification_report(y_test, y_pred,
                               target_names=['Live ball', 'Dead ball']))

    # Find optimal threshold for high precision
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Target 90% precision
    target_precision = 0.90
    idx = np.where(precision >= target_precision)[0]
    if len(idx) > 0:
        optimal_threshold = thresholds[idx[0]]
        optimal_recall = recall[idx[0]]
        print(f"\nOptimal threshold for {target_precision:.0%} precision: {optimal_threshold:.3f}")
        print(f"  Recall at this threshold: {optimal_recall:.3f}")
    else:
        optimal_threshold = 0.5
        print(f"\nUsing default threshold: 0.5")

    # Save model
    model_file = Path('models/dead_ball_precision_filter.pkl')
    model_file.parent.mkdir(parents=True, exist_ok=True)

    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': clf,
            'threshold': optimal_threshold,
            'feature_names': ['event_type_' + str(i) for i in range(9)] +
                           ['time_since_last', 'time_to_next', 'events_in_window',
                            'prev_is_dead', 'prev_is_live', 'next_is_dead', 'next_is_live'],
        }, f)

    print(f"\nSaved model to {model_file}")

    return clf, optimal_threshold


if __name__ == '__main__':
    train_dead_ball_filter()
