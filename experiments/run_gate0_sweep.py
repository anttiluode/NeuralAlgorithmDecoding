#!/usr/bin/env python3
"""Run Gate 0 across several seeds and write a compact aggregate receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gate0_linear_demixer import run


METRICS = [
    "network_nmse",
    "ls_vs_exact_rel_fro",
    "ls_surrogate_vs_network_nmse",
    "jacobian_vs_exact_rel_fro",
    "ood_2x_network_nmse",
    "ood_2x_extracted_math_nmse",
    "circuit_units_for_1pct_fidelity",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", default="results/gate0_summary.json")
    args = p.parse_args()

    receipts = [run(seed, args.steps) for seed in range(args.seeds)]

    summary = {
        "seeds": [r["seed"] for r in receipts],
        "steps": args.steps,
        "architecture": receipts[0]["architecture"],
        "task": receipts[0]["task"],
        "exact_inverse": receipts[0]["final"]["exact_inverse"],
        "metrics": {},
        "seed0_extracted_matrix": receipts[0]["final"]["ls_matrix"],
        "seed0_avg_jacobian": receipts[0]["final"]["avg_jacobian"],
    }

    for key in METRICS:
        values = [r["final"][key] for r in receipts]
        summary["metrics"][key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
