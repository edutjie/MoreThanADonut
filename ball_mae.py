"""Masked Autoencoder for ball trajectory prediction and interpolation.

Uses a transformer-based MAE to learn temporal patterns in ball motion and predict
missing ball positions with better accuracy than linear interpolation.
"""

import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BallTrajectoryDataset(Dataset):
    """Dataset of ball trajectory windows for MAE training."""

    def __init__(self, trajectories, window_size=50, mask_ratio=0.3):
        """
        Args:
            trajectories: List of (T, 2) arrays of ball positions
            window_size: Number of frames in each training window
            mask_ratio: Fraction of positions to mask for training
        """
        self.trajectories = trajectories
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.windows = self._extract_windows()

    def _extract_windows(self):
        """Extract sliding windows from trajectories."""
        windows = []
        for traj in self.trajectories:
            if len(traj) < self.window_size:
                continue
            for i in range(len(traj) - self.window_size + 1):
                window = traj[i:i + self.window_size]
                # Only keep windows with some valid positions
                if np.isfinite(window).any():
                    windows.append(window)
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx].copy()

        # Identify valid positions
        valid = np.isfinite(window).all(axis=1)

        # Create random mask for training
        n_mask = max(1, int(valid.sum() * self.mask_ratio))
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > n_mask:
            mask_indices = np.random.choice(valid_indices, n_mask, replace=False)
        else:
            mask_indices = valid_indices

        # Create input with masked positions
        input_window = window.copy()
        input_window[mask_indices] = np.nan

        # Target is the original window
        target = window.copy()

        return {
            'input': torch.from_numpy(input_window).float(),
            'target': torch.from_numpy(target).float(),
            'mask': torch.from_numpy(mask_indices).long(),
            'valid': torch.from_numpy(valid).bool()
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1)]


class BallMAE(nn.Module):
    """Masked Autoencoder for ball trajectory prediction."""

    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()
        self.d_model = d_model

        # Embed ball positions to d_model dimensions
        self.input_proj = nn.Linear(2, d_model)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, 2) ball positions (may contain NaN for masked positions)
            mask: (batch, seq_len) boolean mask (True for valid, False for masked)

        Returns:
            predictions: (batch, seq_len, 2) predicted ball positions
        """
        batch_size, seq_len = x.shape[:2]

        # Replace NaN with mask tokens
        valid_mask = torch.isfinite(x).all(dim=-1)  # (batch, seq_len)
        x_filled = torch.where(
            valid_mask.unsqueeze(-1).expand_as(x),
            x,
            torch.zeros_like(x)
        )

        # Project to d_model
        x_embed = self.input_proj(x_filled)  # (batch, seq_len, d_model)

        # Replace masked positions with mask token
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        x_embed = torch.where(
            valid_mask.unsqueeze(-1).expand_as(x_embed),
            x_embed,
            mask_token
        )

        # Add positional encoding
        x_embed = self.pos_encoder(x_embed)

        # Apply transformer
        x_encoded = self.transformer(x_embed)

        # Project back to 2D positions
        predictions = self.output_proj(x_encoded)

        return predictions


def train_mae(model, train_loader, val_loader, epochs=50, lr=1e-3, device='cpu'):
    """Train the MAE model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_pos = batch['input'].to(device)
            target_pos = batch['target'].to(device)
            valid = batch['valid'].to(device)

            # Forward pass
            pred = model(input_pos)

            # Loss only on valid positions
            valid_mask = valid.unsqueeze(-1).expand_as(pred)
            if valid_mask.any():
                loss = F.mse_loss(pred[valid_mask], target_pos[valid_mask])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

        train_loss /= max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_pos = batch['input'].to(device)
                target_pos = batch['target'].to(device)
                valid = batch['valid'].to(device)

                pred = model(input_pos)
                valid_mask = valid.unsqueeze(-1).expand_as(pred)
                if valid_mask.any():
                    loss = F.mse_loss(pred[valid_mask], target_pos[valid_mask])
                    val_loss += loss.item()
                    n_val_batches += 1

        val_loss /= max(1, n_val_batches)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/ball_mae_best.pt')

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}', flush=True)

    return model


def interpolate_trajectory(model, trajectory, device='cpu'):
    """Use trained MAE to interpolate missing positions in a trajectory.

    Args:
        model: Trained BallMAE model
        trajectory: (T, 2) array with NaN for missing positions
        device: torch device

    Returns:
        interpolated: (T, 2) array with missing positions filled
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Convert to tensor
        traj_tensor = torch.from_numpy(trajectory).float().unsqueeze(0).to(device)

        # Predict
        pred = model(traj_tensor)

        # Fill only the missing positions
        result = trajectory.copy()
        missing = ~np.isfinite(trajectory).all(axis=1)
        result[missing] = pred[0, missing].cpu().numpy()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-data', type=Path, required=True,
                        help='Path to training trajectories (npy file)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Load training data
    trajectories = np.load(args.train_data, allow_pickle=True)

    # Split train/val
    n_train = int(0.8 * len(trajectories))
    train_trajs = trajectories[:n_train]
    val_trajs = trajectories[n_train:]

    # Create datasets
    train_dataset = BallTrajectoryDataset(train_trajs)
    val_dataset = BallTrajectoryDataset(val_trajs, mask_ratio=0.5)  # Higher mask for validation

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = BallMAE()

    # Train
    print(f"Training MAE on {len(train_dataset)} windows")
    model = train_mae(model, train_loader, val_loader, args.epochs, device=args.device)

    print("Training complete. Best model saved to models/ball_mae_best.pt")


if __name__ == '__main__':
    main()
