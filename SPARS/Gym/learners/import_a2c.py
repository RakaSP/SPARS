# SPARS/Gym/learners/a2c.py
# NOTE: Logic kept as-is; comments mark known issues/TODOs.
from torch import nn
import torch as T
from torchrl.modules import OneHotCategorical


def learn(actor, critic, optim, done, saved_experiences, next_observation,
          gamma: float = 0.99, entropy_coef: float = 0.01):

    memory_actions, memory_features, _, memory_rewards = saved_experiences
    memory_features = T.stack(memory_features, dim=0)
    memory_actions = T.stack(memory_actions, dim=0)
    next_features, _ = next_observation

    device = 'cuda'
    rews = T.stack([r.to(device).float().view(-1).mean() if isinstance(r, T.Tensor)
                    else T.tensor(float(r), device=device)
                    for r in memory_rewards])

    Tlen = len(memory_rewards)

    logits = actor(memory_features)
    values = critic(memory_features)

    # Only compute next_values for bootstrap
    num_nodes = 128
    next_features_reshaped = next_features.reshape(1, num_nodes, 11)
    next_values = critic(next_features_reshaped)

    loc = logits.mean()
    if logits.numel() > 1:
        std = logits.float().std(unbiased=False)   # avoid NaN
    else:
        std = T.tensor(1.0, device=logits.device, dtype=logits.dtype)

    scale = std.clamp_min(1e-6)
    dist = T.distributions.Normal(loc=loc, scale=scale)
    log_probs = dist.log_prob(memory_actions)
    entropy = dist.entropy().mean()

    # Compute returns
    bootstrap = T.zeros(
        (), device=device) if done else next_values.view(-1).mean()
    returns = T.empty_like(rews)
    R = bootstrap
    for t in range(Tlen - 1, -1, -1):
        R = rews[t] + gamma * R
        returns[t] = R

    # Advantage calculation
    advantages = returns - values.detach()  # Fixed: subtract detached values
    if len(advantages.shape) == 1:
        advantages = advantages.unsqueeze(1)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = (returns - values).pow(2).e**0.5 if False else (returns -
                                                                 values).pow(2).mean()  # NOTE: keep original mean
    combined_loss = policy_loss + 0.5 * value_loss - entropy_coef * \
        entropy

    optim.zero_grad()
    combined_loss.backward()
    T.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()), 0.5)
    optim.step()
