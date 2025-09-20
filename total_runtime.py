#!/usr/bin/env python3
"""
check_job_log.py

Validate a jobs CSV:
  - compute_time = finish_time - start_time must equal runtime (within EPS)
  - len(nodes) must equal res

Usage:
  python check_job_log.py path/to/log.csv
  # optional: set tolerance
  python check_job_log.py path/to/log.csv --eps 1e-6
"""

import sys
import csv
import math
import argparse
import ast


def parse_nodes(s):
    """
    Parse nodes column which is expected to look like "[0, 1, 2]".
    Returns a list; on failure returns None.
    """
    if s is None:
        return None
    s = s.strip()
    if s == "" or s.lower() == "none":
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
        # sometimes it's a single int or str; coerce to list
        return [val]
    except Exception:
        # last resort: try comma-split
        try:
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            # coerce to ints if possible
            out = []
            for p in parts:
                try:
                    out.append(int(p))
                except Exception:
                    out.append(p)
            return out
        except Exception:
            return None


def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        # handle floats like "4.0" safely as int
        try:
            return int(float(x))
        except Exception:
            return default


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check job CSV for compute_time/runtime and nodes/res consistency.")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Tolerance for float comparison")
    args = parser.parse_args(argv)

    issues = []
    total = 0
    ok = 0

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # normalize fieldnames (strip spaces)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]

        # start=2 to account for header at line 1
        for idx, row in enumerate(reader, start=2):
            total += 1
            job_id = row.get("job_id")
            job_id_disp = job_id if job_id not in (None, "") else f"row{idx}"

            start_time = to_float(row.get("start_time"))
            finish_time = to_float(row.get("finish_time"))
            runtime = to_float(row.get("runtime"))
            res = to_int(row.get("res"))
            nodes = parse_nodes(row.get("nodes"))

            row_issues = []

            # Check required numeric fields
            if start_time is None or finish_time is None or runtime is None:
                row_issues.append(
                    "missing/invalid numeric time (start_time, finish_time, runtime)")
            else:
                compute_time = finish_time - start_time
                if not (math.isfinite(compute_time) and math.isfinite(runtime)):
                    row_issues.append("non-finite compute_time/runtime")
                else:
                    if abs(compute_time - runtime) > args.eps:
                        row_issues.append(
                            f"compute_time ({compute_time}) != runtime ({runtime}) beyond eps={args.eps}")

            # Check nodes vs res
            if res is None:
                row_issues.append("missing/invalid res")
            if nodes is None:
                row_issues.append("nodes column unparsable")
            elif res is not None and len(nodes) != res:
                row_issues.append(f"len(nodes) ({len(nodes)}) != res ({res})")

            if row_issues:
                issues.append((job_id_disp, row_issues))
            else:
                ok += 1

    # Report
    if not issues:
        print(f"OK: {ok}/{total} rows passed. No issues found.")
        return 0

    print(f"Issues found in {len(issues)}/{total} rows:")
    for jid, msgs in issues:
        print(f" - job_id={jid}:")
        for m in msgs:
            print(f"     * {m}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
