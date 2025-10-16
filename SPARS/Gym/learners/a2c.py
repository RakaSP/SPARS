# SPARS/Gym/learners/a2c.py
# NOTE: Logic kept as-is; comments mark known issues/TODOs.
from torch import nn
import torch as T


def discounted_returns(rewards, gamma, time_dim=-1):
    """
    rewards: tensor with time along `time_dim`
    gamma: float or 0-D tensor (can require_grad)
    returns: discounted-to-go along `time_dim` (same shape as rewards)
    """
    assert T.is_tensor(rewards), "rewards must be a tensor"
    dtype, device = rewards.dtype, rewards.device

    if time_dim != -1:
        rewards = rewards.transpose(time_dim, -1)  # shape: [..., seq_len]

    seq_len = rewards.size(-1)

    if T.is_tensor(gamma):
        gamma = gamma.to(device=device, dtype=dtype)
    else:
        gamma = T.tensor(gamma, device=device, dtype=dtype)

    weighted = rewards.view(-1, 1) * gamma
    flipped = T.flip(weighted, dims=[-1])
    csum = T.cumsum(flipped, dim=-1)
    disc = T.flip(csum, dims=[-1]) / gamma

    if time_dim != -1:
        disc = disc.transpose(-1, time_dim)

    return disc


def learn(model, model_opt, done, saved_experiences, next_observation,
          gamma: float = 0.99, entropy_coef: float = 0.0, eps: float = 1e-12):
    """
    Batched A2C-style update.
    Expects saved_experiences as lists of Tensors with matching shapes per step:
      memory_features[t] : [N, D]
      memory_masks[t]    : [N] or [N,1]  (1=valid, 0=invalid)  (used for logprob reduction)
      memory_actions[t]  : [N] / [N,1] / [N,2] (see original notes)
      memory_rewards[t]  : scalar or tensor reducible to scalar (mean)
    next_observation = (next_features, next_masks) with next_features: [N,D]
    Agent forward: logits, values = model(features)
    """
    # NOTE: Logic kept identical to your original function.
    memory_actions, memory_features, memory_masks, memory_rewards = saved_experiences
    memory_features = T.stack(memory_features, dim=0)
    memory_actions = T.stack(memory_actions, dim=0)
    # NOTE: kept typo as-is (unused)
    memory_rewads = T.stack(memory_rewards, dim=0)
    next_features, _next_masks = next_observation

    # Device handling (two lines as in original)
    device = model.device
    rews = T.stack([r.to(device).float().view(-1).mean() if isinstance(r, T.Tensor)
                    else T.tensor(float(r), device=device)
                    for r in memory_rewards])
    device = next(model.parameters()).device

    Tlen = len(memory_rewards)

    logits, values = model(memory_features)

    # --- SPARS ---
    next_logits, next_values = model(next_features)

    # --- Thomas Reshape ---
    # num_nodes = 128
    # next_features_reshaped = next_features.reshape(1, num_nodes, 11)
    # next_logits, next_values = model(next_features_reshaped)

    loc = logits.mean()

    # use std only when we have >1 element, else fallback to 1.0
    if logits.numel() > 1:
        std = logits.float().std(unbiased=False)   # avoid NaN
    else:
        std = T.tensor(1.0, device=logits.device, dtype=logits.dtype)

    scale = std.clamp_min(1e-6)  # avoid 0 or NaN
    dist = T.distributions.Normal(loc=loc, scale=scale)
    log_probs = dist.log_prob(memory_actions)
    entropy = dist.entropy().mean()

    bootstrap = T.zeros(
        (), device=device) if done else next_values.view(-1).mean()
    returns = T.empty_like(rews)
    R = bootstrap
    for t in range(Tlen - 1, -1, -1):
        R = rews[t] + gamma * R
        returns[t] = R

    advantages = gamma * returns - values
    if len(advantages.shape) == 1:
        advantages = advantages.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    print("log_probs.shape =", log_probs.shape)
    print("advantages.shape =", advantages.shape)
    policy_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).e**0.5 if False else (returns -
                                                                 values).pow(2).mean()  # NOTE: keep original mean
    loss = policy_loss + 0.5 * value_loss - entropy_coef * \
        entropy

    model_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model_opt.step()
