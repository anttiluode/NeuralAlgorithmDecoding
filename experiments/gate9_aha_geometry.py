#!/usr/bin/env python3
"""
Gate 9: an artificial "Aha microscope" for algorithm formation.

Question
--------
When a neural network appears to "click" into a robust algorithm, does the internal
causal geometry also change suddenly, or does useful structure form earlier and more
gradually?

Model organism
--------------
Reuse Gate 5's 16-D GRU trained on decimal column addition:
    input  : two decimal digits per step, least-significant column first
    output : one sum digit per step
    hidden : GRU(16)
    train horizon : 8 columns

Gate 5 already showed that the converged network can be decoded into the exact
two-state base-10 carry program. Gate 9 therefore uses that known decoded mechanism
as a *progress-measure target* during training, analogous in spirit to mechanistic
progress measures in the grokking literature.

Important boundary
------------------
This is NOT a blind algorithm-discovery gate. The carry truth table is used only as a
post-hoc microscope to measure how close checkpoint hidden states are to the already
earned Gate-5 causal abstraction.

At checkpoints we measure:
1. short-horizon digit accuracy (length 8);
2. long-horizon digit accuracy (length 128);
3. exact 128-column sequence success (all 128 output digits correct);
4. causal-response error:
      inject a real hidden state,
      query all 100 possible next digit pairs,
      compare the resulting response signature to the nearest exact carry=0/1 table;
5. exact-signature fraction:
      fraction of sampled hidden states whose complete 100-query response signature
      exactly matches one carry state;
6. hidden-geometry separability of the two nearest carry-response classes.

The central test is whether the visible behavioral "click" precedes, coincides with,
or follows the emergence of the causal representation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gate5_addition_causal_decode import AddGRU, encode_pairs, make_batch


BASE = 10
PAIR_ARRAY = np.array(
    [(a, b) for a in range(BASE) for b in range(BASE)],
    dtype=np.int64,
)
PAIR_TENSOR = torch.tensor(PAIR_ARRAY, dtype=torch.long)
PAIR_INPUT = encode_pairs(
    PAIR_TENSOR[:, 0, None],
    PAIR_TENSOR[:, 1, None],
)

TRUE_RESPONSE = np.zeros((2, BASE * BASE), dtype=np.int64)
for carry in (0, 1):
    for j, (a, b) in enumerate(PAIR_ARRAY):
        TRUE_RESPONSE[carry, j] = (a + b + carry) % BASE


def query_states(model: AddGRU, states: torch.Tensor) -> torch.Tensor:
    """Return logits for every hidden-state x next-digit-pair intervention."""
    n = len(states)
    m = BASE * BASE

    x = PAIR_INPUT.repeat(n, 1, 1)
    h0 = (
        states[:, None, :]
        .repeat(1, m, 1)
        .reshape(n * m, -1)
        .unsqueeze(0)
    )

    with torch.no_grad():
        logits, _hidden, _hn = model(
            x,
            h0,
            return_hidden=True,
        )

    return logits[:, 0].reshape(n, m, BASE)


def causal_signature_metrics(
    model: AddGRU,
    states: torch.Tensor,
) -> dict[str, float]:
    response = query_states(model, states).argmax(-1).cpu().numpy()

    # Post-hoc progress scoring against the Gate-5 decoded causal machine.
    distances = np.stack(
        [
            (response != TRUE_RESPONSE[carry]).mean(axis=1)
            for carry in (0, 1)
        ],
        axis=1,
    )
    labels = distances.argmin(axis=1)

    X = states.cpu().numpy()
    geometry = float("nan")

    if np.sum(labels == 0) >= 3 and np.sum(labels == 1) >= 3:
        mu0 = X[labels == 0].mean(axis=0)
        mu1 = X[labels == 1].mean(axis=0)
        axis = mu1 - mu0
        midpoint = 0.5 * (mu0 + mu1)

        geometric_label = (
            (X - midpoint) @ axis > 0
        ).astype(np.int64)

        direct = np.mean(geometric_label == labels)
        flipped = np.mean((1 - geometric_label) == labels)
        geometry = float(max(direct, flipped))

    return {
        "causal_signature_error": float(
            distances.min(axis=1).mean()
        ),
        "exact_signature_fraction": float(
            np.mean(distances.min(axis=1) == 0.0)
        ),
        "causal_geometry_separability": geometry,
    }


def audit_checkpoint(
    model: AddGRU,
    *,
    seed: int,
    step: int,
    local_batch: int = 256,
    long_batch: int = 128,
    long_horizon: int = 128,
    state_sample: int = 128,
) -> dict[str, float | int]:
    g = torch.Generator().manual_seed(
        seed * 100_000 + step + 777
    )

    x8, y8, _a8, _b8 = make_batch(
        local_batch,
        8,
        seed=None,
    )
    # make_batch does not currently accept a generator. Re-seed torch immediately
    # before each audit so checkpoint probes are deterministic.
    torch.manual_seed(
        seed * 100_000 + step + 777
    )
    x8, y8, _a8, _b8 = make_batch(local_batch, 8)

    torch.manual_seed(
        seed * 100_000 + step + 778
    )
    x_long, y_long, _a_long, _b_long = make_batch(
        long_batch,
        long_horizon,
    )

    with torch.no_grad():
        out8, h8, _hn = model(
            x8,
            return_hidden=True,
        )
        out_long = model(x_long)

    pred8 = out8.argmax(-1)
    pred_long = out_long.argmax(-1)

    short_accuracy = float(
        (pred8 == y8).float().mean()
    )
    long_digit_accuracy = float(
        (pred_long == y_long).float().mean()
    )
    long_sequence_success = float(
        (pred_long == y_long)
        .all(dim=1)
        .float()
        .mean()
    )

    previous = torch.zeros_like(h8)
    previous[:, 1:] = h8[:, :-1]
    H = previous.reshape(-1, model.hidden)

    ids = torch.randperm(
        len(H),
        generator=g,
    )[:state_sample]
    states = H[ids].clone()

    causal = causal_signature_metrics(
        model,
        states,
    )

    return {
        "step": int(step),
        "short_digit_accuracy": short_accuracy,
        "long_digit_accuracy": long_digit_accuracy,
        "long_sequence_success": long_sequence_success,
        **causal,
    }


def train_seed(
    seed: int,
    *,
    steps: int = 1200,
    interval: int = 50,
    batch_size: int = 384,
) -> list[dict[str, float | int]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AddGRU(hidden=16)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=1e-5,
    )

    trajectory = []

    for step in range(steps + 1):
        if step % interval == 0:
            trajectory.append(
                audit_checkpoint(
                    model,
                    seed=seed,
                    step=step,
                )
            )

        if step == steps:
            break

        x, y, _a, _b = make_batch(
            batch_size,
            8,
        )
        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, BASE),
            y.reshape(-1),
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        opt.step()

    return trajectory


def first_step(
    rows: list[dict[str, float | int]],
    key: str,
    *,
    at_least: float | None = None,
    at_most: float | None = None,
) -> int | None:
    for row in rows:
        value = float(row[key])
        if not np.isfinite(value):
            continue
        if at_least is not None and value < at_least:
            continue
        if at_most is not None and value > at_most:
            continue
        return int(row["step"])
    return None


def summarize(
    trajectories: list[
        list[dict[str, float | int]]
    ],
) -> dict[str, object]:
    per_seed = []

    for seed, rows in enumerate(trajectories):
        per_seed.append(
            {
                "seed": seed,
                "geometry_90_step": first_step(
                    rows,
                    "causal_geometry_separability",
                    at_least=0.90,
                ),
                "short_90_step": first_step(
                    rows,
                    "short_digit_accuracy",
                    at_least=0.90,
                ),
                "signature_error_5pct_step": first_step(
                    rows,
                    "causal_signature_error",
                    at_most=0.05,
                ),
                "signature_error_1pct_step": first_step(
                    rows,
                    "causal_signature_error",
                    at_most=0.01,
                ),
                "exact_signature_10pct_step": first_step(
                    rows,
                    "exact_signature_fraction",
                    at_least=0.10,
                ),
                "exact_signature_50pct_step": first_step(
                    rows,
                    "exact_signature_fraction",
                    at_least=0.50,
                ),
                "long_sequence_10pct_step": first_step(
                    rows,
                    "long_sequence_success",
                    at_least=0.10,
                ),
                "long_sequence_50pct_step": first_step(
                    rows,
                    "long_sequence_success",
                    at_least=0.50,
                ),
                "long_sequence_90pct_step": first_step(
                    rows,
                    "long_sequence_success",
                    at_least=0.90,
                ),
            }
        )

    def aggregate(key: str) -> dict[str, object]:
        values = [row[key] for row in per_seed]
        finite = [
            float(value)
            for value in values
            if value is not None
        ]
        return {
            "values": values,
            "median": (
                float(np.median(finite))
                if finite
                else None
            ),
            "mean": (
                float(np.mean(finite))
                if finite
                else None
            ),
            "n_reached": len(finite),
        }

    flat = [
        row
        for rows in trajectories
        for row in rows
        if int(row["step"]) >= 100
    ]

    signature_error = np.array(
        [
            float(row["causal_signature_error"])
            for row in flat
        ]
    )
    sequence_success = np.array(
        [
            float(row["long_sequence_success"])
            for row in flat
        ]
    )

    geometry = np.array(
        [
            float(
                row[
                    "causal_geometry_separability"
                ]
            )
            for row in flat
        ]
    )
    finite_geom = np.isfinite(geometry)

    rho_signature = spearmanr(
        signature_error,
        sequence_success,
    ).statistic

    rho_geometry = spearmanr(
        geometry[finite_geom],
        sequence_success[finite_geom],
    ).statistic

    return {
        "per_seed_thresholds": per_seed,
        "threshold_summary": {
            key: aggregate(key)
            for key in per_seed[0]
            if key != "seed"
        },
        "pooled_checkpoint_correlations": {
            "signature_error_vs_long_sequence_success_spearman": float(
                rho_signature
            ),
            "geometry_separability_vs_long_sequence_success_spearman": float(
                rho_geometry
            ),
            "warning": (
                "These pooled correlations share a strong training-time trend. "
                "They are progress associations, not causal estimates."
            ),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--interval", type=int, default=50)
    p.add_argument(
        "--out",
        default="results/gate9_summary.json",
    )
    args = p.parse_args()

    trajectories = [
        train_seed(
            seed,
            steps=args.steps,
            interval=args.interval,
        )
        for seed in range(args.seeds)
    ]

    receipt = {
        "experiment": "Gate 9 artificial Aha microscope",
        "architecture": "decimal-addition GRU, hidden=16",
        "train_horizon": 8,
        "long_test_horizon": 128,
        "steps": args.steps,
        "checkpoint_interval": args.interval,
        "seeds": list(range(args.seeds)),
        "trajectories": trajectories,
        "summary": summarize(trajectories),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))

    print(
        json.dumps(
            receipt["summary"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
