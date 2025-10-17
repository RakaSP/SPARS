# runner.py
import json
import os
from datetime import datetime
from typing import Any, Dict

import torch as T

from SPARS.Utils import setup_global_logger, get_global_logger, log_output
from SPARS.Simulator.Simulator import Simulator, run_simulation

# IMPORTANT: load Gym config BEFORE importing the env so monkey-patches apply
from SPARS.Gym import config  # monkey patching the gym config
from SPARS.Gym import utils as G
from SPARS.Gym.gym import HPCGymEnv

DEFAULT_CFG_PATH = "simulator_config.yaml"


def _load_config(path: str = DEFAULT_CFG_PATH) -> Dict[str, Any]:
    """
    Load YAML or JSON config from a fixed path.
    If the file doesn't exist, raise (no silent fallback).
    """
    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    if p.suffix.lower() in {".yml", ".yaml"}:
        import yaml  # requires PyYAML
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _choose_device(pref: str) -> str:
    if pref == "auto":
        return "cuda" if T.cuda.is_available() else "cpu"
    return pref


def _parse_start_time(value) -> int:
    """
    Accepts:
      - int/float epoch
      - "now"
      - "YYYY-MM-DD HH:MM:SS"
    Returns epoch seconds (int).
    """
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.lower() == "now":
            return int(datetime.now().timestamp())
        try:
            t = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return int(t.timestamp())
        except ValueError:
            raise ValueError(
                "run.start_time must be epoch int, 'now', or 'YYYY-MM-DD HH:MM:SS'"
            )
    raise TypeError("Unsupported start_time type")


# ---------------------------
# Helpers for flexible agent construction (ONLY addition)
# ---------------------------
def _load_object(spec: str):
    """Load 'pkg.mod:Obj' or 'pkg.mod.Obj' into a Python object."""
    import importlib
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        mod, _, name = spec.rpartition(".")
        if not mod:
            raise ValueError(f"Bad import path: {spec}")
    return getattr(importlib.import_module(mod), name)


def _instantiate_with_flexible_kwargs(cls, params: dict, *, positional_first: str | None = None):
    """
    Instantiate `cls` with kwargs in `params`. If the constructor needs a first positional
    argument (e.g., optimizer 'params'), set positional_first='params'.
    Filters unknown kwargs automatically when possible.
    """
    import inspect
    params = dict(params or {})

    def _call(p: dict):
        if positional_first and positional_first in p:
            pf = p.pop(positional_first)
            try:
                return cls(pf, **p)
            finally:
                p[positional_first] = pf
        return cls(**p)

    try:
        return _call(params)
    except TypeError:
        # Filter unknown kwargs unless ctor accepts **kwargs
        sig = None
        try:
            sig = inspect.signature(cls.__init__)
            has_varkw = any(
                a.kind == inspect.Parameter.VAR_KEYWORD for a in sig.parameters.values())
            if has_varkw:
                raise
            allowed = {k for k in sig.parameters if k != "self"}
            filtered = {k: v for k, v in params.items() if k in allowed}
            return _call(filtered)
        except Exception:
            raise


def _build_agent(rl_cfg: dict, device: str):
    """
    Build agent and optimizer ENTIRELY from cfg['rl']['agent'] with flexible params.
    - No hard-coded keys like obs_dim/act_dim are injected.
    - 'device' handling:
        * if agent.params.device == "auto" -> resolve with _choose_device
        * if agent.params.device missing   -> set to resolved device
        * if agent ctor doesn't accept 'device', it's filtered; if it's an nn.Module,
          we still move it to the device afterward.
    """
    agent_cfg = rl_cfg.get("agent") or {}

    # ----- Agent class -----
    AgentClass = _load_object(agent_cfg.get(
        "class", "RL_Agent.SPARS.agent:ActorCriticMLP"))
    params = dict(agent_cfg.get("params") or {})

    cfg_device = params.get("device", rl_cfg.get("device", "auto"))
    final_device = _choose_device(
        cfg_device if cfg_device is not None else "auto")

    if "device" not in params or str(params.get("device")).lower() == "auto":
        params["device"] = final_device

    model = _instantiate_with_flexible_kwargs(AgentClass, params)

    # Ensure nn.Module is moved even if ctor ignored 'device'
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            model.to(final_device)
    except Exception:
        pass

    # ----- Optimizer -----
    opt_cfg = agent_cfg.get("optimizer") or {}
    OptClass = _load_object(opt_cfg.get("class", "torch.optim:Adam"))

    opt_params = dict(opt_cfg.get("params") or {})
    if "lr" not in opt_params and "learning_rate" in rl_cfg:
        opt_params["lr"] = float(rl_cfg["learning_rate"])

    optimizer = _instantiate_with_flexible_kwargs(
        OptClass,
        {"params": model.parameters() if hasattr(model, "parameters")
         else model, **opt_params},
        positional_first="params",
    )

    return model, optimizer
# ---------------------------

def get_action(model, obs):
    logits, V = model(obs)
    dist = T.distributions.Normal(logits, 0.02)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    return action, log_prob

def main():
    cfg = _load_config(DEFAULT_CFG_PATH)

    # --- Logging ---
    setup_global_logger(
        "runner",
        level=cfg["logging"]["level"],
        log_file=cfg["logging"]["file"],
    )

    logger = get_global_logger()

    # --- Config Unpack (only the pieces you still use) ---
    output_path = cfg["paths"]["output"]

    rl_enabled = bool(cfg["rl"]["enabled"])
    rl_type = cfg["rl"]["type"] if rl_enabled else None
    rl_dt = cfg["rl"]["dt"] if rl_type == "discrete" else None
    device = _choose_device(cfg["rl"]["device"])

    # === RL parameters ===
    epochs = int(cfg["rl"]["epochs"])
    num_nodes = int(cfg["rl"]["num_nodes"])

    if rl_enabled and rl_type == "discrete" and rl_dt is None:
        raise RuntimeError("Discrete RL requires rl.dt in the config file.")

    if rl_enabled:
        # Select the agent per new config structure: rl.agents + rl.assign
        # e.g., "thomas" or "spars"
        assigned_name = cfg["rl"]["assign"]
        # dict with both agents
        agents_dict = cfg["rl"]["agents"]
        # pick the requested one
        agent_cfg = agents_dict[assigned_name]

        # Build simulator from config (no CLI/args)
        simulator = Simulator.from_config(
            cfg,
            rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt},
        )
        env = HPCGymEnv(simulator, device)

        # Build the agent using your existing builder by passing a tiny shim:
        # we keep its expected shape: {'agent': <agent_cfg>, 'device': <rl.device>}
        model, model_opt = _build_agent(
            {"agent": agent_cfg, "device": cfg["rl"]["device"]}, device)

        for _ in range(epochs):
            # reset per epoch
            simulator = Simulator.from_config(
                cfg,
                rl_kwargs={"rl_type": rl_type, "rl_dt": rl_dt},
            )
            env.reset(simulator)
            env.simulator.start_simulator()
            observation = env.get_observation()

            while env.simulator.is_running:
                batch_timesteps_size = 4
                memory_features = []
                memory_logprob = []
                memory_actions = []
                memory_rewards = []
                
                # roll out
                for i in range(batch_timesteps_size):
                    features_, mask_ = observation
                    features_ = features_.to(device)

                    # your policy/value forward

                    # --- SPARS ---
                    
                    
                    # Actions is considered as mean
                    action, logprob = get_action(model, features_)

                    # --- Thomas Reshape ---
                    # features_reshaped = features_.reshape(1, num_nodes, 11)
                    # logits, values = model(features_reshaped)

                    next_observation, reward, done = env.step(action)

                    logger.info(f"Step reward: {reward}")

                    # store experience (detach from graph)
                    memory_actions.append(action.detach())
                    memory_logprob.append(logprob.detach())
                    memory_features.append(features_.detach())
                    memory_rewards.append(reward.detach() if isinstance(reward, T.Tensor)
                                        else T.tensor(float(reward)))

                    saved_experiences = (
                        memory_actions, memory_features, memory_logprob, memory_rewards
                    )

                    observation = next_observation
                
                G.learn(model, model_opt, done,
                            saved_experiences)
        
        log_output(env.simulator, output_path)

        # --- Save agent checkpoint ---
        os.makedirs(output_path, exist_ok=True)
        ckpt = {
            "agent_class": f"{model.__class__.__module__}:{model.__class__.__name__}",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_opt.state_dict(),
            "rl_config": cfg.get("rl", {}),  # left as-is
            "epochs_trained": epochs,
        }
        ckpt_path = os.path.join(output_path, "agent_checkpoint.pt")
        T.save(ckpt, ckpt_path)
        logger.info(f"Saved agent checkpoint to: {ckpt_path}")

    else:
        simulator = Simulator.from_config(cfg)
        run_simulation(simulator, output_path)


if __name__ == "__main__":
    main()
