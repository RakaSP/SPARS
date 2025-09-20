#!/usr/bin/env python3
"""
total_runtime.py

Compute total runtime from a workload JSON:
total_runtime = sum(job["runtime"] * job["res"] for job in jobs)

Usage:
  python total_runtime.py path/to/workload.json
  # or read from stdin:
  cat workload.json | python total_runtime.py -
"""

import sys
import json


def load_json(path: str):
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_total_runtime(data: dict) -> float:
    jobs = data.get("jobs", [])
    total = 0.0
    for j in jobs:
        try:
            res = float(j.get("res", 0))
            runtime = float(j.get("runtime", 0))
            total += res * runtime
        except (TypeError, ValueError):
            # skip malformed job entries
            continue
    return total


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python total_runtime.py <workload.json|->", file=sys.stderr)
        return 2

    path = argv[0]
    try:
        data = load_json(path)
    except Exception as e:
        print(f"Failed to read JSON: {e}", file=sys.stderr)
        return 1

    total = compute_total_runtime(data)
    # print as integer if whole number, else as float
    if total.is_integer():
        print(int(total))
    else:
        print(total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
