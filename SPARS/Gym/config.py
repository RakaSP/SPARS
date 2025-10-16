# config.py
# Define everything here; utils.py has NO defaults/registries.
# Import this module BEFORE importing your env so the monkey-patch takes effect.

from SPARS.Gym import utils as G

# Local name -> dotted path (pure-config; edit freely here)
FEATURE_EXTRACTORS = {
    "global_node_11d": "SPARS.Gym.features.global_node_11d:feature_extraction",
    "thomas": "SPARS.Gym.features.thomas_11d:feature_extraction",
    # "global_node_12d": "mypkg.features:global_node_12d",
}

TRANSLATORS = {
    "scalar_active_target": "SPARS.Gym.translators.scalar_active_target:action_translator",
    "thomas": "SPARS.Gym.translators.thomas:action_translator",
    "import": "SPARS.Gym.translators.import_translator:action_translator",
}

FEASIBLE_MASKS = {
    "default": "SPARS.Gym.features.feasible_mask:get_feasible_mask",
}

REWARDS = {
    "energy_wait_time": "SPARS.Gym.rewards.energy_wait_time:Reward",
    "thomas": "SPARS.Gym.rewards.thomas:Reward",
    # "my_reward": "mypkg.rewards:MyReward",
}

LEARNERS = {
    "a2c": "SPARS.Gym.learners.a2c:learn",
    'import_a2c': "SPARS.Gym.learners.import_a2c:learn",
    # "my_a2c": "mypkg.learners:my_a2c",
}

RETURNS = {
    "a2c": "SPARS.Gym.learners.a2c:discounted_returns",
    # "my_a2c": "mypkg.learners:my_discounted_returns",
}

# Choose by local names (or use dotted strings directly)
CFG = {
    "feature_extractor": "global_node_11d",
    # "feature_extractor": "thomas",
    "translator": "scalar_active_target",
    # "translator": "thomas",
    "feasible_mask": "default",
    # "reward": {"name": "energy_wait_time", "params": {"alpha": 0.5, "beta": 0.5, "device": "cuda"}},
    "reward": {"name": "thomas", "params": {"alpha": 0.5, "beta": 0.5, "device": "cuda"}},
    "learner": "a2c",
    "discounted_returns": "a2c",
}

# ---- Resolve using ONLY this config's maps (no library defaults) ------------


def _resolve_from_map(mapping: dict, key_or_obj):
    if callable(key_or_obj) and not isinstance(key_or_obj, str):
        return key_or_obj
    if isinstance(key_or_obj, str):
        # allow dotted path directly
        target = mapping.get(key_or_obj, key_or_obj)
        return G._load_object(target) if isinstance(target, str) else target
    raise TypeError(
        f"Expected callable or string key/path, got {type(key_or_obj)}")


def _resolve_reward(spec):
    # spec can be: local name, dotted path, class, instance, or dict {"name":..., "params":...}
    if isinstance(spec, dict):
        name = spec.get("name")
        params = dict(spec.get("params", {}))
        if isinstance(name, str):
            # map to dotted path if it's a local name
            name = REWARDS.get(name, name)
        return G.make_reward({"name": name, "params": params})
    if isinstance(spec, str):
        spec = REWARDS.get(spec, spec)
    return G.make_reward(spec)


# Build selected implementations
feature_extractor = _resolve_from_map(
    FEATURE_EXTRACTORS, CFG["feature_extractor"])
translator = _resolve_from_map(TRANSLATORS,       CFG["translator"])
feasible_mask = _resolve_from_map(FEASIBLE_MASKS,    CFG["feasible_mask"])
learner = _resolve_from_map(LEARNERS,          CFG["learner"])
discounted_returns = _resolve_from_map(
    RETURNS,           CFG["discounted_returns"])
reward_instance = _resolve_reward(CFG["reward"])

# ---- Monkey-patch utils BEFORE env import so old imports keep working -------
G.feature_extraction = feature_extractor
G.action_translator = translator
G.get_feasible_mask = feasible_mask
G.learn = learner
G.discounted_returns = discounted_returns


def _reward_factory():
    # fresh instance each time (matches env calling Reward())
    return _resolve_reward(CFG["reward"])


G.Reward = _reward_factory

SELECTED = CFG  # optional: export chosen config
