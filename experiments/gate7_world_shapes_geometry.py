#!/usr/bin/env python3
"""
Gate 7: same algorithm, different worlds, different neural geometry.

SplatField's old "two worlds" lesson was:

    data world
        -> learned geometry / basis
        -> freeze
        -> geometry constrains later dynamics

Gate 7 asks for the neural-algorithm analogue.

Three GRUs have:
- identical architecture;
- identical addition task;
- identical loss;
- identical initialization within each seed;
- different TRAINING-WORLD transition statistics.

Worlds:
  uniform : ordinary random decimal digit pairs.
  sticky  : 65% of samples are chosen so next carry tends to equal current carry.
  toggle  : 65% are chosen so next carry tends to differ from current carry.
            (35% remains uniform, so all digit pairs retain support.)

After training, every model is decoded WITHOUT carry labels using Gate 5's
counterfactual-response machinery. We require the same base-10 executable program.

Then we ask whether the different worlds shaped different realization geometry:
- causal-state separability in raw hidden space;
- normalized distance to a learned causal boundary;
- actual intervention distance required to flip causal response class;
- correlation between geometric boundary margin and causal flip threshold;
- matched random vs boundary-normal perturbations.

This is not a claim about all neural networks. It is a controlled test of:

    world statistics -> neural realization geometry -> causal robustness

while holding the decoded algorithm fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gate5_addition_causal_decode import (
    AddGRU,
    build_transducer,
    discover_states,
    encode_pairs,
    infer_program,
    query_many,
)


BASE = 10
PAIR_LIST = [(a, b) for a in range(BASE) for b in range(BASE)]


def _allowed_pairs():
    allowed = {}
    for carry in (0, 1):
        for next_carry in (0, 1):
            idx = []
            for j, (a, b) in enumerate(PAIR_LIST):
                if int(a + b + carry >= BASE) == next_carry:
                    idx.append(j)
            allowed[(carry, next_carry)] = torch.tensor(idx, dtype=torch.long)
    return allowed


ALLOWED = _allowed_pairs()


def make_world_batch(
    batch: int,
    length: int,
    world: str,
    *,
    uniform_fraction: float = 0.35,
    seed: int | None = None,
):
    if world not in {"uniform", "sticky", "toggle"}:
        raise ValueError("world must be uniform, sticky, or toggle")

    g = torch.Generator().manual_seed(seed) if seed is not None else None
    a = torch.empty((batch, length), dtype=torch.long)
    b = torch.empty_like(a)
    y = torch.empty_like(a)
    carry = torch.zeros(batch, dtype=torch.long)

    for t in range(length):
        aa = torch.randint(0, BASE, (batch,), generator=g)
        bb = torch.randint(0, BASE, (batch,), generator=g)

        if world != "uniform":
            use_bias = torch.rand(batch, generator=g) > uniform_fraction
            for c in (0, 1):
                mask = (carry == c) & use_bias
                n = int(mask.sum())
                if not n:
                    continue

                desired = c if world == "sticky" else 1 - c
                candidates = ALLOWED[(c, desired)]
                chosen = candidates[
                    torch.randint(0, len(candidates), (n,), generator=g)
                ]
                aa[mask] = chosen // BASE
                bb[mask] = chosen % BASE

        total = aa + bb + carry
        y[:, t] = total % BASE
        carry = total // BASE
        a[:, t] = aa
        b[:, t] = bb

    return encode_pairs(a, b), y, a, b


def world_transition_stats(world: str, seed: int = 12345):
    _x, _y, a, b = make_world_batch(
        20000, 8, world, seed=seed
    )
    carry = torch.zeros(len(a), dtype=torch.long)
    same = 0
    total_count = 0

    for t in range(a.shape[1]):
        next_carry = ((a[:, t] + b[:, t] + carry) >= BASE).long()
        same += int((next_carry == carry).sum())
        total_count += len(a)
        carry = next_carry

    return {
        "carry_persistence_fraction": same / total_count,
        "carry_flip_fraction": 1.0 - same / total_count,
    }


def train_world(seed: int, world: str, steps: int = 2000, length: int = 8):
    # Resetting the same seed for each world gives identical initial weights
    # within each paired seed trio.
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AddGRU(hidden=16)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=1e-5,
    )

    for _ in range(steps):
        x, y, _a, _b = make_world_batch(256, length, world)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, BASE),
            y.reshape(-1),
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    x, y, _a, _b = make_world_batch(
        2048,
        length,
        "uniform",
        seed=seed + 9999,
    )
    with torch.no_grad():
        accuracy = float((model(x).argmax(-1) == y).float().mean())

    return model, accuracy


def collect_uniform_prev_states(model: AddGRU, seed: int):
    x, _y, _a, _b = make_world_batch(
        512,
        12,
        "uniform",
        seed=seed + 20000,
    )
    with torch.no_grad():
        _logits, hseq, _hn = model(x, return_hidden=True)

    prev = torch.zeros_like(hseq)
    prev[:, 1:] = hseq[:, :-1]
    return prev.reshape(-1, model.hidden)


def dominant_response_classes(model: AddGRU, states: torch.Tensor):
    logits, _next_h, _symbols = query_many(model, states)
    response = logits.argmax(-1).cpu().numpy()

    unique, counts = np.unique(
        response,
        axis=0,
        return_counts=True,
    )
    if len(unique) < 2:
        raise RuntimeError("fewer than two response signatures found")

    major_idx = np.argsort(counts)[-2:]
    major = unique[major_idx]

    distance = np.stack(
        [(response != signature).mean(axis=1) for signature in major],
        axis=1,
    )
    labels = distance.argmin(axis=1)

    return {
        "major": major,
        "labels": labels,
        "exact_fraction": float(np.mean(distance.min(axis=1) == 0.0)),
    }


def classify_full_response(
    model: AddGRU,
    states: torch.Tensor,
    major: np.ndarray,
):
    logits, _next_h, _symbols = query_many(model, states)
    response = logits.argmax(-1).cpu().numpy()

    distance = np.stack(
        [(response != signature).mean(axis=1) for signature in major],
        axis=1,
    )
    return distance.argmin(axis=1), distance.min(axis=1)


def geometry_audit(model: AddGRU, seed: int):
    H = collect_uniform_prev_states(model, seed)
    g = torch.Generator().manual_seed(seed + 25000)
    ids = torch.randperm(len(H), generator=g)[:1024]
    H = H[ids].clone()

    discovered = dominant_response_classes(model, H)
    labels = discovered["labels"]
    major = discovered["major"]
    X = H.cpu().numpy()

    rng = np.random.default_rng(seed + 30000)
    order = rng.permutation(len(H))
    train_idx = order[:768]
    test_idx = order[768:]

    mu0 = X[train_idx][labels[train_idx] == 0].mean(axis=0)
    mu1 = X[train_idx][labels[train_idx] == 1].mean(axis=0)

    centered = np.concatenate(
        [
            X[train_idx][labels[train_idx] == 0] - mu0,
            X[train_idx][labels[train_idx] == 1] - mu1,
        ],
        axis=0,
    )
    covariance = np.cov(centered.T) + 1e-3 * np.eye(X.shape[1])
    inv_cov = np.linalg.inv(covariance)

    delta = mu1 - mu0
    mahalanobis_separation = float(
        np.sqrt(delta @ inv_cov @ delta)
    )

    w = inv_cov @ delta
    wnorm = np.linalg.norm(w) + 1e-12
    axis = w / wnorm
    midpoint = 0.5 * (mu0 + mu1)

    score = (X[test_idx] - midpoint) @ w
    geometric_label = (score > 0).astype(int)
    lda_accuracy = float(
        np.mean(geometric_label == labels[test_idx])
    )

    valid = test_idx[geometric_label == labels[test_idx]][:48]
    base = H[valid].clone()
    base_label = labels[valid]

    rms_state_norm = float(
        np.sqrt(np.mean(np.sum(X[test_idx] ** 2, axis=1)))
    )

    raw_score = (base.cpu().numpy() - midpoint) @ w
    predicted_margin = (
        np.abs(raw_score) / wnorm / rms_state_norm
    )
    toward_other = (
        -np.sign(raw_score)[:, None] * axis[None, :]
    )

    # Binary search the actual response-class flip radius along the learned
    # causal-boundary normal.
    lo = np.zeros(len(base))
    hi = np.full(len(base), 0.8)

    for _ in range(8):
        midpoint_frac = 0.5 * (lo + hi)
        perturbed = base + torch.tensor(
            toward_other
            * (midpoint_frac * rms_state_norm)[:, None],
            dtype=torch.float32,
        )
        new_label, _distance = classify_full_response(
            model,
            perturbed,
            major,
        )
        flipped = new_label != base_label
        hi = np.where(flipped, midpoint_frac, hi)
        lo = np.where(flipped, lo, midpoint_frac)

    final_probe = base + torch.tensor(
        toward_other * (hi * rms_state_norm)[:, None],
        dtype=torch.float32,
    )
    final_label, _ = classify_full_response(
        model,
        final_probe,
        major,
    )
    successful = final_label != base_label

    if int(successful.sum()) >= 3:
        margin_radius_corr = float(
            np.corrcoef(
                predicted_margin[successful],
                hi[successful],
            )[0, 1]
        )
    else:
        margin_radius_corr = None

    # Fixed normalized perturbations: causal-axis vs matched random direction.
    fixed = {}
    rng2 = np.random.default_rng(seed + 40000)
    random_direction = rng2.normal(
        size=(len(base), X.shape[1])
    )
    random_direction /= (
        np.linalg.norm(
            random_direction,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )

    for fraction in (0.2, 0.3):
        axis_probe = base + torch.tensor(
            toward_other * (fraction * rms_state_norm),
            dtype=torch.float32,
        )
        random_probe = base + torch.tensor(
            random_direction * (fraction * rms_state_norm),
            dtype=torch.float32,
        )

        axis_label, _ = classify_full_response(
            model, axis_probe, major
        )
        random_label, _ = classify_full_response(
            model, random_probe, major
        )

        fixed[str(fraction)] = {
            "boundary_normal_flip_rate": float(
                np.mean(axis_label != base_label)
            ),
            "matched_random_flip_rate": float(
                np.mean(random_label != base_label)
            ),
        }

    return {
        "signature_exact_fraction": discovered["exact_fraction"],
        "lda_accuracy": lda_accuracy,
        "mahalanobis_class_separation": mahalanobis_separation,
        "median_predicted_boundary_margin_in_rms_units": float(
            np.median(predicted_margin)
        ),
        "median_actual_causal_flip_radius_in_rms_units": (
            float(np.median(hi[successful]))
            if np.any(successful)
            else None
        ),
        "boundary_margin_vs_flip_radius_correlation": (
            margin_radius_corr
        ),
        "flip_radius_success_fraction": float(
            np.mean(successful)
        ),
        "fixed_perturbations": fixed,
    }


def decode_program(model: AddGRU, seed: int):
    discovered = discover_states(model, seed)
    machine = build_transducer(model, discovered)
    program = infer_program(machine)

    return {
        "causal_state_count": int(machine["k"]),
        "output_consistency": float(
            machine["output_consistency"]
        ),
        "transition_consistency": float(
            machine["transition_consistency"]
        ),
        "program_found": bool(program["found"]),
        "program_base": (
            int(program["base"])
            if program["found"]
            else None
        ),
        "program_mismatches": int(
            program["mismatches"]
        ),
        "state_to_carry": program.get(
            "state_to_carry"
        ),
    }


def run(seed: int = 0, steps: int = 2000):
    receipt = {
        "seed": seed,
        "steps": steps,
        "paired_initialization": True,
        "uniform_fraction_in_biased_worlds": 0.35,
        "world_transition_stats": {
            world: world_transition_stats(world)
            for world in ("uniform", "sticky", "toggle")
        },
        "worlds": {},
    }

    for world in ("uniform", "sticky", "toggle"):
        model, uniform_accuracy = train_world(
            seed,
            world,
            steps=steps,
            length=8,
        )
        receipt["worlds"][world] = {
            "uniform_eval_accuracy": uniform_accuracy,
            "decoded": decode_program(model, seed),
            "geometry": geometry_audit(model, seed),
        }

    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument(
        "--out",
        default="results/gate7_seed0.json",
    )
    args = p.parse_args()

    receipt = run(args.seed, args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
