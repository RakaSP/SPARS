import pandas as pd
import math
import re
from ast import literal_eval

BATSIM_CSV = "_batsim.csv"
SPARS_CSV = "_SPARS.csv"

# --- helpers ---


def parse_nodes_generic(s) -> list[int]:
    """
    Robust parser for node lists:
    - Accepts Python list strings like "[0, 1, 2, 3]"
    - Accepts ranges like "0-3,6,8-10" or "0-3 6 8-10" (spaces/semicolons ok)
    - Normalizes en/em dashes to '-'
    - Treats empty or '-1' as no allocation
    - Returns sorted unique ints
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []

    s = str(s).strip()
    if not s or s == "-1":
        return []

    # Try literal list first (e.g., "[0, 1, 2]")
    try:
        val = literal_eval(s)
        if isinstance(val, list):
            return sorted({int(x) for x in val})
    except Exception:
        pass

    # Normalize dashes and split on any non digit/hyphen
    s = s.replace("–", "-").replace("—", "-")
    tokens = re.split(r"[^\d\-]+", s)
    tokens = [t for t in tokens if t]

    nodes = []
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-", 1)
            if a and b:
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                nodes.extend(range(lo, hi + 1))
        else:
            nodes.append(int(tok))

    return sorted(set(nodes))


def nearly_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return a == b
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)
    except Exception:
        return a == b


# --- load & clean ---
b = pd.read_csv(BATSIM_CSV)
s = pd.read_csv(SPARS_CSV)

# Keep only COMPLETED_WALLTIME_REACHED from Batsim
b = b[b["final_state"] == "COMPLETED_WALLTIME_REACHED"].copy()

# Minimal columns we need
b = b[["job_id", "starting_time", "finish_time", "allocated_resources"]].rename(
    columns={"starting_time": "start_time", "allocated_resources": "nodes"}
)
s = s[["job_id", "start_time", "finish_time", "nodes"]].copy()

# Normalize nodes using the robust parser
b["nodes"] = b["nodes"].map(parse_nodes_generic)
s["nodes"] = s["nodes"].map(parse_nodes_generic)

# Build dicts by job_id
b_by_id = {
    int(row.job_id): {
        "start_time": row.start_time,
        "finish_time": row.finish_time,
        "nodes": sorted(row.nodes),
    }
    for _, row in b.iterrows()
}
s_by_id = {
    int(row.job_id): {
        "start_time": row.start_time,
        "finish_time": row.finish_time,
        "nodes": sorted(row.nodes),
    }
    for _, row in s.iterrows()
}

# Compare
# only compare jobs present in both
all_ids = sorted(set(b_by_id) & set(s_by_id))
differences = 0
for jid in all_ids:
    bb = b_by_id[jid]
    ss = s_by_id[jid]

    diff_fields = []
    if not nearly_equal(bb["start_time"], ss["start_time"]):
        diff_fields.append(("start_time", bb["start_time"], ss["start_time"]))
    if not nearly_equal(bb["finish_time"], ss["finish_time"]):
        diff_fields.append(
            ("finish_time", bb["finish_time"], ss["finish_time"]))
    if bb["nodes"] != ss["nodes"]:
        diff_fields.append(("nodes", bb["nodes"], ss["nodes"]))

    if diff_fields:
        differences += 1
        print(f"\n⚠️  Mismatch for job_id={jid}:")
        for name, bval, sval in diff_fields:
            print(f"  - {name}: BATSIM={bval}  vs  SPARS={sval}")
        input("\nPaused — press Enter to continue checking...")

# Optional: report jobs present only in one side
only_b = sorted(set(b_by_id) - set(s_by_id))
only_s = sorted(set(s_by_id) - set(b_by_id))
if only_b:
    print(f"\nNote: job_ids only in BATSIM: {only_b}")
if only_s:
    print(f"\nNote: job_ids only in SPARS: {only_s}")

if differences == 0 and not only_b and not only_s:
    print("✅ All compared jobs match on start_time, finish_time, and nodes.")
