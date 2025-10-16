import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, device):
        super().__init__()
        self.device = device

        # ---- Shared Body ----
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # ---- Actor Head ----
        self.actor_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # ---- Critic Head ----
        self.critic_head = nn.Linear(64, 1)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, obs):
        """
        obs: [B, F] (F = obs_dim)
        Output:
          action: [1], ∈ [0,1]
          value : [1]
        """
        x = self.body(obs)                    # [B, 64]
        x_mean = x.mean(dim=0, keepdim=True)  # [1, 64]
        action = self.actor_head(x_mean).view(-1)  # [1]
        value = self.critic_head(x_mean).view(-1)  # [1]
        return action, value
