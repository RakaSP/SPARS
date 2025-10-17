from ast import literal_eval  # if used elsewhere
import math
import warnings
import pandas as pd
import ast
import os
# logger_setup.py
import logging
import os
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

TRACE = 5
logging.addLevelName(TRACE, "TRACE")
logging.Logger.trace = lambda self, msg, * \
    args, **kwargs: self.log(TRACE, msg, *args, **kwargs)

_global_logger = None
_logger_config = None


def setup_global_logger(
    name: str = "runner",
    level: str = "INFO",
    log_file: Optional[str] = None,
    propagate: bool = False
):
    global _logger_config
    _logger_config = {
        "name": name,
        "level": level,
        "log_file": log_file,
        "propagate": propagate
    }


def get_global_logger() -> logging.Logger:
    global _global_logger, _logger_config

    if _global_logger is None:
        if _logger_config is None:
            # Default config if not setup
            _logger_config = {"name": "runner", "level": "INFO",
                              "log_file": None, "propagate": False}

        logger = logging.getLogger(_logger_config["name"])

        level_value = {
            "TRACE": TRACE,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(_logger_config["level"].upper(), logging.INFO)
        logger.setLevel(level_value)
        logger.propagate = _logger_config["propagate"]

        fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        ch = logging.StreamHandler()
        ch.setLevel(level_value)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if _logger_config["log_file"]:
            fh = logging.FileHandler(
                _logger_config["log_file"], encoding="utf-8")
            fh.setLevel(level_value)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        _global_logger = logger

    return _global_logger


def _to_int_series(s):
    # handle lists, strings like "3", floats like 3.0, and None
    return pd.to_numeric(s, errors="coerce").astype("Int64")  # nullable int


def _to_float_series(s):
    # unify time columns to float (seconds). If you use datetime, convert both sides to datetime64[ns] instead.
    return pd.to_numeric(s, errors="coerce").astype("float64")


def parse_nodes(x):
    # handle NaN/None/empty strings
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return list(v) if isinstance(v, (list, tuple)) else v
        except Exception:
            return []
    return x  # as-is


def process_node_job_data(nodes_data, jobs):
    """Build intervals per node and attach job subtime as submission_time (floats kept)."""

    mapping_non_active = {
        'switching_off': -2,
        'switching_on': -3,
        'sleeping': -4,
    }

    # --- node intervals ---
    node_intervals = []
    for node in (nodes_data or []):
        nid = node['id']
        current_dvfs = None
        for itv in node.get('state_history', []):
            if 'dvfs_mode' in itv:
                current_dvfs = itv['dvfs_mode']
            if itv['start_time'] < itv['finish_time']:
                node_intervals.append({
                    'node_id':    nid,
                    'state':      itv['state'],
                    'dvfs_mode':  current_dvfs,
                    'start_time': float(itv['start_time']),
                    'finish_time': float(itv['finish_time']),
                })

    node_intervals_df = pd.DataFrame(
        node_intervals,
        columns=['node_id', 'state', 'dvfs_mode', 'start_time', 'finish_time']
    )

    if node_intervals_df.empty:
        return pd.DataFrame(columns=[
            'dvfs_mode', 'state', 'submission_time', 'start_time', 'finish_time', 'nodes', 'job_id', 'terminated'
        ])

    # --- jobs exploded by node ---
    jobs_exploded = jobs.copy()

    # nodes "1 2 3" -> [1,2,3], then explode
    jobs_exploded['nodes'] = jobs_exploded['nodes'].map(parse_nodes)
    jobs_exploded = jobs_exploded.explode(
        'nodes').rename(columns={'nodes': 'node_id'})

    # keep times as float
    for c in ('start_time', 'finish_time', 'subtime'):
        if c in jobs_exploded.columns:
            jobs_exploded[c] = pd.to_numeric(
                jobs_exploded[c], errors='coerce').astype(float)

    # ensure essential cols exist minimally
    if 'terminated' not in jobs_exploded.columns:
        jobs_exploded['terminated'] = pd.NA
    if 'job_id' not in jobs_exploded.columns:
        jobs_exploded['job_id'] = -1

    # join ACTIVE intervals with jobs on (node_id, start_time, finish_time)
    active_df = node_intervals_df[node_intervals_df['state'] == 'active'].copy(
    )
    merged = pd.merge(
        active_df,
        jobs_exploded[['node_id', 'start_time', 'finish_time',
                       'subtime', 'job_id', 'terminated']],
        on=['node_id', 'start_time', 'finish_time'],
        how='left'
    )
    merged['submission_time'] = merged['subtime']  # carry from jobs
    merged.drop(columns=['subtime'], inplace=True)
    merged['job_id'] = merged['job_id'].fillna(-1)

    # non-active intervals: fill placeholders
    non_active_df = node_intervals_df[node_intervals_df['state'] != 'active'].copy(
    )
    non_active_df['submission_time'] = pd.NA
    non_active_df['job_id'] = non_active_df['state'].map(
        mapping_non_active).fillna(-1)
    non_active_df['terminated'] = pd.NA

    combined = pd.concat([merged, non_active_df], ignore_index=True)

    # group nodes that share the same interval tuple
    grouped = combined.groupby(
        ['state', 'dvfs_mode', 'submission_time',
            'start_time', 'finish_time', 'job_id'],
        dropna=False
    ).agg(
        nodes=('node_id', lambda s: ' '.join(
            map(str, sorted(int(i) for i in s.dropna().tolist())))),
        terminated=('terminated', lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())
                    if s.notna().any() else pd.NA)
    ).reset_index()

    grouped = grouped.sort_values(['start_time', 'finish_time'])

    return grouped[['dvfs_mode', 'state', 'submission_time', 'start_time', 'finish_time', 'nodes', 'job_id', 'terminated']]


def build_waiting_time_df(jobs_execution_log: list) -> pd.DataFrame:
    """
    Convert jobs_execution_log (list of dict) into a DataFrame with:
    job_id, subtime, start_time, finish_time, waiting_time (start_time - subtime).

    Handles both numeric timestamps and datetime-like strings.
    """
    df = pd.DataFrame(jobs_execution_log)
    required = {'job_id', 'subtime', 'start_time', 'finish_time'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    sub = df['subtime']
    start = df['start_time']

    if not (pd.api.types.is_numeric_dtype(sub) and pd.api.types.is_numeric_dtype(start)):
        sub_dt = pd.to_datetime(sub, errors='coerce')
        start_dt = pd.to_datetime(start, errors='coerce')
        waiting = (start_dt - sub_dt).dt.total_seconds()
    else:
        waiting = start - sub

    out = df.loc[:, ['job_id', 'subtime', 'start_time', 'finish_time']].copy()
    out['waiting_time'] = waiting
    return out


def write_waiting_time_log(simulator, output_folder: str, filename: str = "waiting_time_log.csv") -> str:
    """
    Build waiting-time DataFrame from simulator.Monitor.jobs_execution_log
    and write it to <output_folder>/<filename>. Returns the file path.
    """
    os.makedirs(output_folder, exist_ok=True)
    wt_df = build_waiting_time_df(simulator.Monitor.jobs_execution_log)
    path = os.path.join(output_folder, filename)
    wt_df.to_csv(path, index=False)
    return path


def build_energy_df(energy_log: list) -> pd.DataFrame:
    """
    Convert simulator.Monitor.energy (list[dict]) into a DataFrame with columns:
    id, energy_consumption, energy_effective, energy_waste.
    """
    df = pd.DataFrame(energy_log)
    required = {'id', 'energy_consumption', 'energy_effective', 'energy_waste'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing required columns in energy log: {sorted(missing)}")

    out = df.loc[:, ['id', 'energy_consumption',
                     'energy_effective', 'energy_waste']].copy()

    # (Optional) coerce to numeric in case inputs are strings
    for col in ['energy_consumption', 'energy_effective', 'energy_waste']:
        out[col] = pd.to_numeric(out[col], errors='coerce')

    return out


def write_energy_log(simulator, output_folder: str, filename: str = "energy_log.csv") -> str:
    """
    Build energy DataFrame from simulator.Monitor.energy and write it to CSV.
    Returns the file path.
    """
    os.makedirs(output_folder, exist_ok=True)
    energy_df = build_energy_df(simulator.Monitor.energy)
    path = os.path.join(output_folder, filename)
    energy_df.to_csv(path, index=False)
    return path


def _sum_states_dur(states_dur: list) -> dict:
    """
    Sum durations across all nodes and all DVFS modes for each state bucket.
    Expected keys per node: active_idle, active_compute, switching_off, switching_on, sleeping.
    Returns a dict with totals for each bucket (float seconds).
    """
    totals = {
        "total_active_idle": 0.0,
        "total_active_compute": 0.0,
        "total_switching_off": 0.0,
        "total_switching_on": 0.0,
        "total_sleeping": 0.0,
    }
    if not states_dur:
        totals["total_time_all_states"] = 0.0
        return totals

    for entry in states_dur:
        # each value is a dict of dvfs_mode -> duration
        for key, out_key in [
            ("active_idle", "total_active_idle"),
            ("active_compute", "total_active_compute"),
            ("switching_off", "total_switching_off"),
            ("switching_on", "total_switching_on"),
            ("sleeping", "total_sleeping"),
        ]:
            bucket = entry.get(key, {})
            if isinstance(bucket, dict):
                totals[out_key] += float(pd.to_numeric(pd.Series(bucket),
                                         errors="coerce").sum())

    totals["total_time_all_states"] = sum(totals.values())
    return totals


def build_metrics_df(jobs_execution_log: list, energy_log: list, states_dur: list | None = None) -> pd.DataFrame:
    """
    Return a 1-row DataFrame with:
      - total_waiting_time
      - mean_waiting_time
      - total_energy_waste
      - total_energy_consumption
      - energy_effective (= total_energy_consumption - total_energy_waste)
      - totals of node state durations aggregated over all nodes & dvfs:
        total_active_idle, total_active_compute, total_switching_off, total_switching_on,
        total_sleeping, total_time_all_states

    waiting_time is computed as start_time - subtime (seconds if datetimes).
    """
    # reuse existing builders
    wt_df = build_waiting_time_df(
        jobs_execution_log) if jobs_execution_log else pd.DataFrame(columns=["waiting_time"])
    en_df = build_energy_df(energy_log) if energy_log else pd.DataFrame(
        columns=["energy_waste"])

    # Waiting-time aggregates
    wt_series = pd.to_numeric(
        wt_df.get("waiting_time", pd.Series(dtype=float)), errors="coerce")
    total_waiting = wt_series.sum(min_count=1)
    mean_waiting = wt_series.mean() if not wt_series.empty else float("nan")

    # Energy aggregates
    waste_series = pd.to_numeric(
        en_df.get("energy_waste", pd.Series(dtype=float)), errors="coerce")
    total_waste = waste_series.sum(min_count=1)

    # Try several common column names for total consumption
    cons_col_candidates = ["energy_consumption",
                           "energy_total", "consumed_energy", "energy"]
    cons_series = None
    for col in cons_col_candidates:
        if col in en_df.columns:
            cons_series = pd.to_numeric(en_df[col], errors="coerce")
            break

    if cons_series is not None:
        total_consumption = cons_series.sum(min_count=1)
        energy_effective = total_consumption - \
            (total_waste if pd.notna(total_waste) else 0.0)
    else:
        total_consumption = float("nan")
        energy_effective = float("nan")

    # NaN-safe defaults to 0.0 for totals; keep mean as NaN if unavailable
    if pd.isna(total_waiting):
        total_waiting = 0.0
    if pd.isna(total_waste):
        total_waste = 0.0
    if pd.isna(total_consumption):
        total_consumption = 0.0
    if pd.isna(energy_effective):
        energy_effective = 0.0

    state_totals = _sum_states_dur(states_dur or [])

    row = {
        "total_waiting_time": float(total_waiting),
        "mean_waiting_time": float(mean_waiting) if pd.notna(mean_waiting) else 0.0,
        "total_energy_waste": float(total_waste),
        "total_energy_consumption": float(total_consumption),
        "energy_effective": float(energy_effective),
        **state_totals,
    }
    return pd.DataFrame([row])


def write_metrics_log(simulator, output_folder: str, filename: str = "metrics.csv") -> str:
    """
    Build metrics DataFrame and write it to <output_folder>/<filename>.
    """
    os.makedirs(output_folder, exist_ok=True)
    metrics_df = build_metrics_df(
        simulator.Monitor.jobs_execution_log,
        simulator.Monitor.energy,
        simulator.Monitor.states_dur,
    )
    path = os.path.join(output_folder, filename)
    metrics_df.to_csv(path, index=False)
    return path


def write_state_switch_csv(simulator, output_folder: str, filename: str = "state_switch.csv") -> str:
    """
    Save `state_switch` (list of dicts) to CSV with ordered columns:
    time, nb_sleeping, nb_switching_on, nb_switching_off, nb_idle, nb_computing.

    Returns the written filepath.
    """
    state_switch = simulator.Monitor.state_switch

    os.makedirs(output_folder, exist_ok=True)

    cols = ["time", "nb_sleeping", "nb_switching_on",
            "nb_switching_off", "nb_idle", "nb_computing"]
    df = pd.DataFrame(state_switch)

    # ensure all expected columns exist (missing -> NaN)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # order columns
    df = df[cols]

    # optional: coerce numeric columns (except 'time' if it's datetime-like strings)
    for c in cols:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    path = os.path.join(output_folder, filename)
    df.to_csv(path, index=False)


def log_output(simulator, output_folder):
    os.makedirs(f'{output_folder}', exist_ok=True)

    raw_node_log = pd.DataFrame(simulator.Monitor.states_hist)
    raw_node_log.to_csv(f'{output_folder}/raw_node_log.csv', index=False)

    raw_job_log = pd.DataFrame(simulator.Monitor.jobs_execution_log)
    raw_job_log.to_csv(f'{output_folder}/raw_job_log.csv', index=False)

    write_waiting_time_log(simulator, output_folder)
    write_energy_log(simulator, output_folder)
    write_metrics_log(simulator, output_folder)
    write_state_switch_csv(simulator, output_folder)

    node_log = process_node_job_data(
        simulator.Monitor.states_hist, raw_job_log)
    node_log.to_csv(f'{output_folder}/node_log.csv', index=False)
