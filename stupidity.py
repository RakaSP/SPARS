#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any

# === set your paths here ===
path = Path("/home/raka/SPARS/workloads/training/sample-real-synthetic.json")     # <- change me
out_path = Path("/home/raka/SPARS/workloads/training/sample-real-synthetic-fixed.json")  # <- change me
# ===========================

def transform(obj: Any) -> Any:
    """Recursively transform dicts/lists:
       - 'id' -> 'job_id' (non-destructive if 'job_id' already exists)
       - 'walltime' -> both 'reqtime' and 'runtime' with the same value
    """
    if isinstance(obj, dict):
        # make a shallow copy to avoid mutating during iteration
        o = dict(obj)

        # walltime -> reqtime + runtime
        if "walltime" in o:
            v = o.pop("walltime")
            o["reqtime"] = v
            o["runtime"] = v

        # id -> job_id
        if "id" in o:
            if "job_id" not in o:
                o["job_id"] = o["id"]
            o.pop("id", None)

        # recurse
        for k, v in list(o.items()):
            o[k] = transform(v)
        return o

    if isinstance(obj, list):
        return [transform(x) for x in obj]

    return obj

def try_parse_json_all(text: str):
    return json.loads(text)

def process_json_file(inp: Path, outp: Path):
    data = json.loads(inp.read_text())
    data = transform(data)
    outp.write_text(json.dumps(data, indent=2))
    print(f"Wrote JSON -> {outp}")

def process_jsonl_file(inp: Path, outp: Path):
    with inp.open() as fin, outp.open("w") as fout:
        for line_no, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            obj = transform(obj)
            json.dump(obj, fout)
            fout.write("\n")
    print(f"Wrote JSONL -> {outp}")

if __name__ == "__main__":
    text = path.read_text()
    try:
        # Try full JSON
        try_parse_json_all(text)
        process_json_file(path, out_path)
    except Exception:
        # Fallback to JSONL (line-delimited JSON)
        process_jsonl_file(path, out_path)
