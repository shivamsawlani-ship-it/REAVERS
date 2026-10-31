"""
CAVShield PyTorch LSTM Autoencoder for CAN Sequence Anomaly Detection.

Architecture:
- Input shape: (Batch, 64, 17) where 64 = window_size, 17 = per-message features.
- Encoder: 2-layer LSTM (input_dim -> hidden_dim=64), bottleneck projection (hidden_dim -> bottleneck_dim=24).
- Decoder: Bottleneck repetition across sequence length (64), 2-layer LSTM (bottleneck_dim -> hidden_dim=64),
  linear projection (hidden_dim -> 17).
- Loss: Mean Squared Error (MSE).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """2-layer LSTM encoder compressing sequence into a compact bottleneck representation."""

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 64,
        bottleneck_dim: int = 24,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence (B, seq_len, input_dim) -> (B, bottleneck_dim)."""
        # out: (B, seq_len, hidden_dim), (h_n, c_n)
        _, (h_n, _) = self.lstm(x)
        # Take the top layer's final hidden state: h_n[-1] has shape (B, hidden_dim)
        bottleneck = self.fc(h_n[-1])
        return bottleneck


class Decoder(nn.Module):
    """2-layer LSTM decoder reconstructing sequence from bottleneck representation."""

    def __init__(
        self,
        bottleneck_dim: int = 24,
        hidden_dim: int = 64,
        output_dim: int = 17,
        seq_len: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck (B, bottleneck_dim) -> (B, seq_len, output_dim)."""
        # Repeat bottleneck vector across seq_len
        # bottleneck: (B, bottleneck_dim) -> (B, 1, bottleneck_dim) -> (B, seq_len, bottleneck_dim)
        repeated = bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(repeated)
        reconstruction = self.fc(lstm_out)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """Complete LSTM Autoencoder for multivariate temporal anomaly detection."""

    def __init__(
        self,
        input_dim: int = 17,
        seq_len: int = 64,
        hidden_dim: int = 64,
        bottleneck_dim: int = 24,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (B, seq_len, input_dim).
        """
        bottleneck = self.encoder(x)
        return self.decoder(bottleneck)

    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute MSE reconstruction error per sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, seq_len, input_dim).
        reduction : str
            'none': returns (B,) MSE per sample.
            'mean': returns scalar mean across batch.

        Returns
        -------
        torch.Tensor
            Reconstruction error.
        """
        reconstructed = self.forward(x)
        diff_sq = (x - reconstructed) ** 2
        # Mean across seq_len and input_dim
        sample_mse = torch.mean(diff_sq, dim=(1, 2))
        if reduction == "none":
            return sample_mse
        return torch.mean(sample_mse)

