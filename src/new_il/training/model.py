"""Small action-chunk policy for LIBERO BC / PA-TCS training.

ActionMLPPolicy is a deliberately minimal fallback that keeps the training
infrastructure working without SmolVLA. The interface (obs → action_chunk)
is stable so SmolVLA or any other backbone can be swapped in later without
changing the training loop.

Architecture:
  obs [B, obs_dim]
    → LayerNorm → Linear(obs_dim, hidden) → GELU
    → (num_layers - 1) × [Linear(hidden, hidden) → LayerNorm → GELU → Dropout]
    → Linear(hidden, horizon * action_dim)
    → reshape [B, horizon, action_dim]

Parameter count at defaults (obs=3, hidden=256, layers=3, H=8, D=7): ~200K.
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit("torch is required. Run: uv sync --extra train") from exc


class ActionMLPPolicy(nn.Module):
    """MLP that maps a single observation to a fixed-length action chunk.

    Args:
        obs_dim: dimension of the input observation (e.g. 3 for ee_pos).
        action_dim: dimension of each action step (default 7 for LIBERO).
        horizon: number of action steps to predict per forward pass.
        hidden_dim: width of hidden layers.
        num_layers: total number of linear layers (including input and output).
        dropout: dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 7,
        horizon: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 (input + output).")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon

        layers: list[nn.Module] = [
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_dim, horizon * action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            obs: [B, obs_dim] float32 tensor.

        Returns:
            action_chunk: [B, horizon, action_dim] float32 tensor.
        """
        out = self.net(obs)
        return out.view(obs.shape[0], self.horizon, self.action_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
