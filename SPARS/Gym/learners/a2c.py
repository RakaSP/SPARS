# SPARS/Gym/learners/a2c.py
# NOTE: Logic kept as-is; comments mark known issues/TODOs.
from torch import nn
import torch as T
from SPARS.Utils import get_global_logger

logger = get_global_logger()


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
    memory_rewards = T.stack(memory_rewards, dim=0)
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

    delta = T.zeros(Tlen, dtype=values.dtype, device=values.device)
    # first option
    print(values.shape)
    # delta[Tlen-1] = rews[Tlen-1] - values[Tlen-1]

    # Sec option
    delta[Tlen-1] = rews[Tlen-1]

    for t in range(Tlen-2, -1, -1):
        delta[t] = rews[t] + gamma * values[t + 1] - values[t]

    advantages = T.zeros(Tlen, dtype=values.dtype, device=values.device)

    curr_advantage = delta[Tlen-1]

    for t in range(Tlen-2, -1, -1):
        curr_advantage = delta[t] + gamma * curr_advantage
        advantages[t] = curr_advantage

    logger.info(f"log_probs.shape = {log_probs.shape}")
    logger.info(f"advantages.shape = {advantages.shape}")
    policy_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).e**0.5 if False else (returns -
                                                                 values).pow(2).mean()  # NOTE: keep original mean
    logger.info(f'Policy Loss: {policy_loss}')
    logger.info(f'Value Loss: {value_loss}')

    loss = policy_loss + 0.5 * value_loss - entropy_coef * \
        entropy

    model_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model_opt.step()
