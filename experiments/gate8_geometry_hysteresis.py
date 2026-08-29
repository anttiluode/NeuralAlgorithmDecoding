#!/usr/bin/env python3
"""
Gate 8: once the algorithmic geometry has formed, does a new world reshape it?

Gate 7 showed:
    same task + different training statistics from scratch
        -> same decoded base-10 algorithm
        -> different causal robustness geometry.

That leaves a confound: perhaps the different geometries are merely different solutions
chosen during initial learning.

Gate 8 turns that into a temporal experiment.

1. Train ONE uniform-world addition GRU to convergence.
2. Clone the exact trained weights into two branches.
3. Continue training one clone in the sticky world and one in the toggle world.
4. Keep task, architecture, labels, optimizer family, and decoded algorithm unchanged.
5. Measure the causal flip radius before and after the world switch.

If the geometry rapidly tracks current statistics, the flip radius should move toward the
from-scratch Gate-7 sticky/toggle values.
If it hardly moves, the realization has hysteresis: the world shaped geometry during
formation, and later statistics do not cheaply rewrite that geometry while the algorithm
continues to work.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gate5_addition_causal_decode import (
    build_transducer,
    discover_states,
    infer_program,
)
from gate7_world_shapes_geometry import (
    geometry_audit,
    make_world_batch,
    train_world,
)


def continue_training(model, world: str, steps: int = 800):
    model = copy.deepcopy(model)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-5,
    )

    for _ in range(steps):
        x, y, _a, _b = make_world_batch(
            256,
            8,
            world,
        )
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, 10),
            y.reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        opt.step()

    return model


def decode(model, seed: int):
    disc = discover_states(model, seed)
    machine = build_transducer(model, disc)
    program = infer_program(machine)
    return {
        "state_count": int(machine["k"]),
        "program_found": bool(program["found"]),
        "base": int(program["base"]) if program["found"] else None,
        "mismatches": int(program["mismatches"]),
    }


def uniform_accuracy(model, seed: int):
    x, y, _a, _b = make_world_batch(
        2048,
        8,
        "uniform",
        seed=seed + 88888,
    )
    with torch.no_grad():
        return float(
            (model(x).argmax(-1) == y)
            .float()
            .mean()
        )


def compact_geometry(model, seed: int):
    g = geometry_audit(model, seed)
    return {
        "predicted_margin_rms": g[
            "median_predicted_boundary_margin_in_rms_units"
        ],
        "actual_flip_radius_rms": g[
            "median_actual_causal_flip_radius_in_rms_units"
        ],
        "margin_radius_corr": g[
            "boundary_margin_vs_flip_radius_correlation"
        ],
    }


def run(seed: int = 0, initial_steps: int = 2000, switch_steps: int = 800):
    uniform_model, _ = train_world(
        seed,
        "uniform",
        steps=initial_steps,
        length=8,
    )

    before = {
        "uniform_accuracy": uniform_accuracy(
            uniform_model, seed
        ),
        "decode": decode(uniform_model, seed),
        "geometry": compact_geometry(
            uniform_model, seed
        ),
    }

    branches = {}
    for world in ("sticky", "toggle"):
        model = continue_training(
            uniform_model,
            world,
            steps=switch_steps,
        )
        branches[world] = {
            "uniform_accuracy": uniform_accuracy(
                model, seed
            ),
            "decode": decode(model, seed),
            "geometry": compact_geometry(
                model, seed
            ),
        }

    return {
        "seed": seed,
        "initial_world": "uniform",
        "initial_steps": initial_steps,
        "switch_steps": switch_steps,
        "before_switch": before,
        "branches": branches,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--initial_steps", type=int, default=2000)
    p.add_argument("--switch_steps", type=int, default=800)
    p.add_argument(
        "--out",
        default="results/gate8_seed0.json",
    )
    args = p.parse_args()

    receipt = run(
        args.seed,
        args.initial_steps,
        args.switch_steps,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
