# This file is used to try 100 different case to find for a potential bug
# 100-case generator & simulator runner (NEW_RUNNER-like, simple config rewrite per case)
# - Always algorithm="easy_auto"
# - Always uses timeout
# - Per case: rewrite a copy of the (initialized) config and run
# ---------------------------------------------------------------------------

import os
import sys
import json
import math
import random
import traceback
from datetime import datetime
from typing import Dict, Any, Tuple
import copy

import numpy as np

# ---------------------------------------------------------------------------
# Try SPARS logging/output; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from SPARS.Utils import get_logger as spars_get_logger, log_output as spars_log_output
except Exception:
    spars_get_logger = None
    spars_log_output = None

import logging


def _fallback_get_logger(name: str, level: str = "INFO", log_file: str | None = None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def get_logger(name: str, level: str = "INFO", log_file: str | None = None):
    if spars_get_logger is not None:
        try:
            return spars_get_logger(name, level=level, log_file=log_file)
        except Exception:
            pass
    return _fallback_get_logger(name, level, log_file)


def log_output(simulator, output_path: str):
    """Prefer SPARS.Utils.log_output; else leave a breadcrumb file."""
    if spars_log_output is not None:
        try:
            return spars_log_output(simulator, output_path)
        except Exception:
            pass
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "_debug_fallback_log_output.txt"), "a", encoding="utf-8") as f:
        f.write(f"Run completed at {datetime.now().isoformat()}\n")

# ---------------------------------------------------------------------------
# Config helpers (NEW_RUNNER-like)
# ---------------------------------------------------------------------------


DEFAULT_CFG_PATH = "simulator_config.yaml"


def _load_config(path: str = DEFAULT_CFG_PATH) -> Dict[str, Any]:
    """Load YAML/JSON config, else return defaults with debug_spammer initialized."""
    import pathlib
    p = pathlib.Path(path)
    if p.exists():
        if p.suffix.lower() in {".yml", ".yaml"}:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
    else:
        cfg = {
            "paths": {
                "base_dir": "debug_runs",
            },
            "run": {
                "algorithm": "easy_auto",    # will be enforced anyway
                "overrun_policy": "terminate",
                "start_time": 0,
            },
            "logging": {
                "level": "INFO",
                "file": "debug_runs/debug_spam.log",
            },
        }
    _ensure_debug_spammer(cfg)
    return cfg


def _ensure_debug_spammer(cfg: Dict[str, Any]) -> None:
    """Make sure cfg['debug_spammer'] exists with sane defaults."""
    ds = cfg.get("debug_spammer")
    if not isinstance(ds, dict):
        ds = {}
        cfg["debug_spammer"] = ds
    ds.setdefault("cases", 100)
    ds.setdefault("jobs_per_case", 100)
    ds.setdefault("platform_nodes_min", 2)
    ds.setdefault("platform_nodes_max", 32)
    # Timeout range moved here per your request
    ds.setdefault("timeout_range", [10, 30])


def _parse_start_time(value) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.lower() == "now":
            return int(datetime.now().timestamp())
        from datetime import datetime as _dt
        t = _dt.strptime(value, "%Y-%m-%d %H:%M:%S")
        return int(t.timestamp())
    raise TypeError("Unsupported start_time type")

# ---------------------------------------------------------------------------
# Workload & platform generators
# ---------------------------------------------------------------------------


class ProblemGenerator:
    def __init__(
        self,
        lambda_arrival: float = 0.2,
        mu_execution: float = 80,
        sigma_execution: float = 30,
        mu_noise: float = 0,
        sigma_noise: float = 1,
        num_jobs: int | None = None,
        max_node: int = 8,
    ):
        self.lambda_arrival = lambda_arrival
        self.mu_execution = mu_execution
        self.sigma_execution = sigma_execution
        self.mu_noise = mu_noise
        self.sigma_noise = sigma_noise
        self.max_node = max_node
        self.num_jobs = num_jobs if num_jobs is not None else max(
            1, int(np.random.normal(10, 2)))

    def generate(self):
        # ----- arrivals: Poisson vs bursty mix -----
        arrival_mode = random.choice(["poisson", "bursty"])
        if arrival_mode == "poisson":
            interarrival_times = np.random.exponential(
                1 / max(self.lambda_arrival, 1e-6), self.num_jobs)
        else:
            # mixture of fast & slow rates to create bursts
            fast = max(self.lambda_arrival * random.uniform(2.0, 5.0), 1e-6)
            slow = max(self.lambda_arrival * random.uniform(0.1, 0.5), 1e-6)
            rates = np.where(np.random.rand(self.num_jobs) < 0.7, fast, slow)
            interarrival_times = np.random.exponential(1 / rates)
        arrival_times = np.cumsum(interarrival_times)

        # ----- requested exec time: pick a distribution per case -----
        exec_mode = random.choice(["normal", "lognormal", "gamma"])
        if exec_mode == "normal":
            requested_execution_times = np.random.normal(
                self.mu_execution, max(self.sigma_execution, 1e-6), self.num_jobs)
        elif exec_mode == "lognormal":
            sigma_l = random.uniform(0.3, 0.8)
            mu_l = math.log(max(self.mu_execution, 1.0)) - 0.5 * sigma_l**2
            requested_execution_times = np.random.lognormal(
                mu_l, sigma_l, self.num_jobs)
        else:  # gamma
            k = random.uniform(2.0, 7.0)
            theta = max(self.mu_execution / k, 1e-6)
            requested_execution_times = np.random.gamma(
                k, theta, self.num_jobs)

        requested_execution_times = np.maximum(requested_execution_times, 1.0)

        # noise (still additive, but with random scale)
        noise_sigma = max(self.sigma_noise, 1e-6) * random.uniform(0.5, 1.5)
        noise = np.random.normal(self.mu_noise, noise_sigma, self.num_jobs)
        actual_execution_times = np.maximum(
            requested_execution_times + noise, 1.0)

        # ----- nodes required: mix of patterns -----
        vals = np.empty(self.num_jobs, dtype=float)
        r = np.random.rand(self.num_jobs)
        mid = math.ceil(self.max_node / 2)
        for i in range(self.num_jobs):
            if r[i] < 0.30:
                # uniform across full range
                vals[i] = np.random.randint(1, self.max_node + 1)
            elif r[i] < 0.70:
                # concentrated near mid with some spread
                vals[i] = np.random.normal(mid, max(self.max_node / 5, 1))
            else:
                # extremes: either tiny or maxed-out jobs
                vals[i] = 1 if np.random.rand() < 0.5 else self.max_node
        num_nodes_required = np.clip(
            np.round(vals), 1, self.max_node).astype(int)

        # ----- assemble -----
        workloads = []
        for i in range(self.num_jobs):
            workloads.append({
                "job_id": i + 1,
                "res": int(num_nodes_required[i]),
                "subtime": round(float(arrival_times[i])),
                "reqtime": round(float(requested_execution_times[i])),
                "runtime": round(float(actual_execution_times[i])),
                "profile": "100",
                "user_id": 0
            })
        return workloads


def build_workload_json(max_node_for_jobs: int, workloads: list) -> Dict[str, Any]:
    return {
        "nb_res": max_node_for_jobs,
        "jobs": workloads,
        "profiles": {
            "100": {
                "cpu": 10**22,
                "com": 0,
                "type": "parallel_homogeneous"
            }
        }
    }


def generate_machine(
    machine_id: int,
    base_power: float | None = None,
    switching_off_power: float | None = None,
    switching_on_power: float | None = None,
    sleeping_power: float | None = None,
    switching_on_time: int | None = None,
    switching_off_time: int | None = None
) -> Dict[str, Any]:
    # ---- randomized specs per machine ----
    # Base perf/power
    base_power = base_power if base_power is not None else random.uniform(
        150.0, 300.0)
    base_speed = random.uniform(0.8, 1.3)  # vary compute capability

    # DVFS multipliers
    u_factor = random.uniform(0.5, 0.9)    # underclock scale
    o_factor = random.uniform(1.1, 1.6)    # overclock scale

    underclock_power = base_power * random.uniform(0.55, 0.9)
    overclock_power = base_power * random.uniform(1.1, 1.6)
    underclock_speed = base_speed * u_factor
    overclock_speed = base_speed * o_factor

    # Idle / transition powers
    sleeping_power = sleeping_power if sleeping_power is not None else random.uniform(
        3.0, 20.0)
    switching_off_power = switching_off_power if switching_off_power is not None else random.uniform(
        5.0, 22.0)
    switching_on_power = switching_on_power if switching_on_power is not None else random.uniform(
        5.0, 22.0)

    # Transition times (allow zeros and varied latencies)
    switching_off_time = switching_off_time if switching_off_time is not None else random.randint(
        1, 20)
    switching_on_time = switching_on_time if switching_on_time is not None else random.randint(
        1, 20)

    return {
        "id": machine_id,
        "dvfs_profiles": {
            "underclock_1": {"power": underclock_power, "compute_speed": underclock_speed},
            "base":         {"power": base_power,       "compute_speed": base_speed},
            "overclock_1":  {"power": overclock_power,  "compute_speed": overclock_speed},
        },
        "dvfs_mode": "base",
        "states": {
            "active": {
                "power": "from_dvfs",
                "compute_speed": "from_dvfs",
                "transitions": [{"state": "switching_off", "transition_time": 0}],
            },
            "switching_off": {
                "power": switching_off_power,
                "compute_speed": "from_dvfs",
                "transitions": [{"state": "sleeping", "transition_time": switching_off_time}],
            },
            "switching_on": {
                "power": switching_on_power,
                "compute_speed": "from_dvfs",
                "transitions": [{"state": "active", "transition_time": switching_on_time}],
            },
            "sleeping": {
                "power": sleeping_power,
                "compute_speed": "from_dvfs",
                "transitions": [{"state": "switching_on", "transition_time": 0}],
            },
        },
    }


def generate_cluster(num_node: int) -> Dict[str, Any]:
    # each machine gets its own randomized characteristics
    machines = [generate_machine(i) for i in range(num_node)]
    return {"machines": machines}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def unique_case_paths(base_dir: str, case_id: int) -> Tuple[str, str, str]:
    case_dir = os.path.join(base_dir, f"run_{case_id:04d}")
    workload_path = os.path.join(case_dir, "workloads", "generated.json")
    platform_path = os.path.join(case_dir, "platforms", "generated.json")
    result_dir = os.path.join(case_dir, "results")
    return workload_path, platform_path, result_dir

# ---------------------------------------------------------------------------
# Single-case runner: rewrite a copy of the existing config per case
# ---------------------------------------------------------------------------


def run_one_case(case_id: int, base_cfg: Dict[str, Any], logger: logging.Logger) -> tuple[bool, str]:
    # Per-case randomness
    seed = random.randint(1, 10_000_000)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # Start with a deep copy of the *existing* config and initialize debug_spammer
    cfg = copy.deepcopy(base_cfg)
    _ensure_debug_spammer(cfg)
    ds = cfg["debug_spammer"]

    # Resolve base paths
    paths = cfg.get("paths") or {}
    base_dir = paths.get("base_dir", "debug_runs")

    # Per-case IO paths
    workload_path, platform_path, result_dir = unique_case_paths(
        base_dir, case_id)
    ensure_dir(os.path.dirname(workload_path))
    ensure_dir(os.path.dirname(platform_path))
    ensure_dir(result_dir)

    # Derive per-case selections (you can replace with fixed values if desired)
    nodes_lo = int(ds.get("platform_nodes_min", 2))
    nodes_hi = int(ds.get("platform_nodes_max", 32))
    platform_nodes = random.randint(nodes_lo, nodes_hi)
    job_max_nodes = max(1, platform_nodes // 2)
    jobs = int(ds.get("jobs_per_case", 100))
    t_lo, t_hi = ds.get("timeout_range", [10, 30])
    timeout = random.randint(int(t_lo), int(t_hi))

    # Simple workload params (can be made configurable similarly)
    lambda_arrival = max(0.05, np.random.uniform(0.05, 0.5))
    mu_exec = np.random.uniform(40, 120)
    sigma_exec = np.random.uniform(10, 50)
    mu_noise = np.random.uniform(-5, 5)
    sigma_noise = np.random.uniform(0.5, 5.0)

    # Generate per-case workload/platform files
    pg = ProblemGenerator(lambda_arrival, mu_exec, sigma_exec, mu_noise, sigma_noise,
                          num_jobs=jobs, max_node=job_max_nodes)
    workloads = pg.generate()
    write_json(workload_path, build_workload_json(job_max_nodes, workloads))
    write_json(platform_path, generate_cluster(platform_nodes))

    # ---- Rewrite the config copy for THIS case ----
    cfg["paths"] = {
        "workload": workload_path,
        "platform": platform_path,
        "output":   result_dir,
        "base_dir": base_dir,  # keep for reference
    }
    run_cfg = cfg.get("run") or {}
    run_cfg.update({
        "algorithm": "easy_auto",                 # always easy_auto
        "timeout": timeout,                       # always set
        "overrun_policy": run_cfg.get("overrun_policy", "terminate"),
        "start_time": _parse_start_time(run_cfg.get("start_time", 0)),
    })
    cfg["run"] = run_cfg

    # Record useful per-case selections back into debug_spammer (for traceability)
    cfg["debug_spammer"].update({
        "platform_nodes": platform_nodes,
        "job_max_nodes": job_max_nodes,
        "jobs_per_case": jobs,
        "timeout_used": timeout,
        "seed": seed,
    })

    logger.info(
        f"[CASE {case_id}] seed={seed} alg=easy_auto timeout={timeout}s "
        f"platform_nodes={platform_nodes} job_max_nodes={job_max_nodes} jobs={jobs} -> start"
    )

    try:
        from SPARS.Simulator.Simulator import Simulator, run_simulation
        simulator = Simulator.from_config(cfg)
        run_simulation(simulator, result_dir)
        log_output(simulator, result_dir)
        logger.info(f"[CASE {case_id}] SUCCESS")
        return True, ""
    except Exception as e:
        err_txt = f"[CASE {case_id}] ERROR: {type(e).__name__}: {e}"
        logger.error(err_txt)
        tb_path = os.path.join(result_dir, "exception_traceback.txt")
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(err_txt + "\n\n")
            f.write(traceback.format_exc())
        return False, err_txt

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = _load_config(DEFAULT_CFG_PATH)
    _ensure_debug_spammer(cfg)  # make sure it's initialized *before* we start

    base_dir = (cfg.get("paths") or {}).get("base_dir", "debug_runs")
    ensure_dir(base_dir)

    logger = get_logger(
        "debug_spammer",
        level=(cfg.get("logging") or {}).get("level", "INFO"),
        log_file=(cfg.get("logging") or {}).get(
            "file", os.path.join(base_dir, "debug_spam.log")),
    )

    total_cases = int(cfg["debug_spammer"].get("cases", 100))
    success = 0
    failures = 0
    summary_csv = os.path.join(base_dir, "summary.csv")

    with open(summary_csv, "w", encoding="utf-8") as fcsv:
        fcsv.write("case_id,success,error\n")
        for case_id in range(1, total_cases + 1):
            ok, err = run_one_case(case_id, cfg, logger)
            if ok:
                success += 1
                fcsv.write(f"{case_id},1,\n")
            else:
                failures += 1
                err_clean = err.replace("\n", " ").replace(",", ";")
                fcsv.write(f"{case_id},0,{err_clean}\n")

            if case_id % 10 == 0:
                logger.info(
                    f"Progress: {case_id}/{total_cases} | OK={success} FAIL={failures}")

    logger.info(
        f"ALL DONE. total={total_cases} success={success} failures={failures}")
    logger.info(f"Summary CSV: {summary_csv}")
    logger.info(
        f"Logs: {(cfg.get('logging') or {}).get('file', os.path.join(base_dir, 'debug_spam.log'))}")


if __name__ == "__main__":
    main()
