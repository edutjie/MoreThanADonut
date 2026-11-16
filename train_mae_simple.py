"""Simple MAE training on ball trajectories - fixed window size for fast training."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load training data
print("Loading training data...")
trajectories = np.load('external_data/ball_training_data.npy', allow_pickle=True)
print(f"Loaded {len(trajectories)} trajectories")

# Use only the longest trajectory (tactical tracking) for quick training
traj = trajectories[0]  # 199,319 frames
print(f"Training trajectory length: {len(traj)}")

# Create fixed-size windows
WINDOW_SIZE = 50
MASK_RATIO = 0.3

windows = []
for i in range(0, len(traj) - WINDOW_SIZE, WINDOW_SIZE // 2):  # 50% overlap
    window = traj[i:i + WINDOW_SIZE]
    if np.isfinite(window).any():
        windows.append(window)

windows = np.array(windows)
print(f"Created {len(windows)} windows of size {WINDOW_SIZE}")

# Split train/val
n_train = int(0.8 * len(windows))
train_windows = windows[:n_train]
val_windows = windows[n_train:]

print(f"Train: {len(train_windows)}, Val: {len(val_windows)}")

# Simple transformer model
class SimpleBallMAE(nn.Module):
    def __init__(self, d_model=32, nhead=2, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(2, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        pos = torch.arange(WINDOW_SIZE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(WINDOW_SIZE, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, x, mask):
        # x: (batch, seq_len, 2), mask: (batch, seq_len) bool
        x_filled = torch.where(mask.unsqueeze(-1).expand_as(x), x, torch.zeros_like(x))
        x_embed = self.input_proj(x_filled)

        # Replace masked with mask token
        mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
        x_embed = torch.where(mask.unsqueeze(-1).expand_as(x_embed), x_embed, mask_token)

        # Add positional encoding
        x_embed = x_embed + self.pe[:x.size(1)]

        # Transform
        x_encoded = self.transformer(x_embed)
        pred = self.output_proj(x_encoded)
        return pred

# Training
device = 'cpu'
model = SimpleBallMAE().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\nTraining MAE model...")
EPOCHS = 10
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    n_batches = 0

    # Shuffle training data
    indices = np.random.permutation(len(train_windows))

    for i in range(0, len(train_windows), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        batch = train_windows[batch_idx]

        # Create random masks
        batch_tensor = torch.from_numpy(batch).float().to(device)
        valid = torch.from_numpy(np.isfinite(batch).all(axis=2)).bool().to(device)

        # Random masking
        mask = valid.clone()
        for j in range(len(mask)):
            valid_pos = torch.where(valid[j])[0]
            if len(valid_pos) > 1:
                n_mask = max(1, int(len(valid_pos) * MASK_RATIO))
                mask_pos = valid_pos[torch.randperm(len(valid_pos))[:n_mask]]
                mask[j, mask_pos] = False

        # Forward
        pred = model(batch_tensor, mask)

        # Loss only on valid positions
        valid_mask = valid.unsqueeze(-1).expand_as(pred)
        if valid_mask.any():
            loss = F.mse_loss(pred[valid_mask], batch_tensor[valid_mask])

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
        for i in range(0, len(val_windows), BATCH_SIZE):
            batch = val_windows[i:i+BATCH_SIZE]
            batch_tensor = torch.from_numpy(batch).float().to(device)
            valid = torch.from_numpy(np.isfinite(batch).all(axis=2)).bool().to(device)

            mask = valid.clone()
            for j in range(len(mask)):
                valid_pos = torch.where(valid[j])[0]
                if len(valid_pos) > 1:
                    n_mask = max(1, int(len(valid_pos) * 0.5))  # More aggressive masking for val
                    mask_pos = valid_pos[torch.randperm(len(valid_pos))[:n_mask]]
                    mask[j, mask_pos] = False

            pred = model(batch_tensor, mask)
            valid_mask = valid.unsqueeze(-1).expand_as(pred)
            if valid_mask.any():
                loss = F.mse_loss(pred[valid_mask], batch_tensor[valid_mask])
                val_loss += loss.item()
                n_val_batches += 1

    val_loss /= max(1, n_val_batches)
    print(f'Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}')

# Save model
torch.save(model.state_dict(), 'models/ball_mae_trained.pt')
print("\nMAE model training completed!")
print("Model saved to models/ball_mae_trained.pt")

# Quick evaluation
print("\n=== Evaluation ===")
model.eval()
with torch.no_grad():
    # Test on a validation window
    test_window = val_windows[0:1]
    test_tensor = torch.from_numpy(test_window).float().to(device)
    valid = torch.from_numpy(np.isfinite(test_window).all(axis=2)).bool().to(device)

    # Mask 50% of positions
    mask = valid.clone()
    valid_pos = torch.where(valid[0])[0]
    n_mask = len(valid_pos) // 2
    mask_pos = valid_pos[torch.randperm(len(valid_pos))[:n_mask]]
    mask[0, mask_pos] = False

    pred = model(test_tensor, mask)

    # Calculate error on masked positions
    masked_positions = valid[0] & ~mask[0]
    if masked_positions.any():
        true_pos = test_tensor[0, masked_positions]
        pred_pos = pred[0, masked_positions]
        error = torch.mean(torch.norm(pred_pos - true_pos, dim=1))
        print(f"Mean prediction error on masked positions: {error.item():.4f} (normalized coordinates)")
        print(f"Masked {masked_positions.sum()} positions, predicted with avg error {error.item():.4f}")

print("\nMAE training experiment successfully completed!")
