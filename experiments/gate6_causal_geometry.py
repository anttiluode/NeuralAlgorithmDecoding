#!/usr/bin/env python3
"""
Gate 6: causal geometry — the world shaped a hidden manifold, and that geometry
constrains which perturbations change the decoded algorithm.

Lineage:
SplatField's "two worlds" observation was:
    training data chooses packet geometry
        -> freeze geometry
        -> Gram/operator geometry constrains later dynamics.

Here the organism is Gate 5's trained decimal-addition GRU.  We already know how to
discover two causal response-equivalence classes without carry labels.

This gate asks a different question:
    Once those computational classes have formed, is there a geometry in raw hidden
    state space that constrains which perturbations can change the causal computation?

Protocol:
1. Train the GRU normally. Freeze it.
2. Label hidden states ONLY by their 100-input counterfactual response signatures.
3. Fit a held-out LDA axis separating the two discovered causal classes.
4. For held-out states, perturb by the same Euclidean norm either:
      a) across the learned causal boundary normal,
      b) in a random direction orthogonal to that normal.
5. Re-query all 100 interventions and ask whether the causal response class changed.

This does not claim a universal neural geometry. It tests one controlled model organism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gate5_addition_causal_decode import (
    make_batch,
    query_many,
    train,
)


def dominant_response_classes(model, states):
    logits, _, _ = query_many(model, states)
    responses = logits.argmax(-1).cpu().numpy()

    unique, _inv, counts = np.unique(
        responses, axis=0, return_inverse=True, return_counts=True
    )
    if len(unique) < 2:
        raise RuntimeError("fewer than two response signatures discovered")

    major_idx = np.argsort(counts)[-2:]
    major = unique[major_idx]

    distance = np.stack(
        [(responses != signature).mean(axis=1) for signature in major],
        axis=1,
    )
    labels = distance.argmin(1)

    return {
        "major": major,
        "labels": labels,
        "exact_fraction": float(np.mean(distance.min(1) == 0.0)),
        "counts": [int(counts[i]) for i in major_idx],
    }


def classify_by_full_response(model, states, major):
    logits, _, _ = query_many(model, states)
    responses = logits.argmax(-1).cpu().numpy()
    distance = np.stack(
        [(responses != signature).mean(axis=1) for signature in major],
        axis=1,
    )
    return distance.argmin(1), distance.min(1)


def run(seed=0, steps=2000):
    model, checkpoints = train(seed=seed, steps=steps, T=8)

    # Real trajectory states, but no true carry labels.
    x, _y, _a, _b = make_batch(1024, 12, seed + 20000)
    with torch.no_grad():
        _, hseq, _ = model(x, return_hidden=True)

    prev = torch.zeros_like(hseq)
    prev[:, 1:] = hseq[:, :-1]
    all_states = prev.reshape(-1, model.hidden)

    g = torch.Generator().manual_seed(seed + 25000)
    ids = torch.randperm(len(all_states), generator=g)[:2048]
    states = all_states[ids].clone()

    discovered = dominant_response_classes(model, states)
    labels = discovered["labels"]
    X = states.cpu().numpy()

    rng = np.random.default_rng(seed + 30000)
    perm = rng.permutation(len(states))
    train_idx = perm[:1536]
    test_idx = perm[1536:]

    # LDA is used only as a geometry microscope after causal classes have been
    # discovered by intervention.  It is not given carry labels.
    mu0 = X[train_idx][labels[train_idx] == 0].mean(0)
    mu1 = X[train_idx][labels[train_idx] == 1].mean(0)
    covariance = np.cov(X[train_idx].T) + 1e-3 * np.eye(X.shape[1])
    w = np.linalg.solve(covariance, mu1 - mu0)
    wnorm = np.linalg.norm(w) + 1e-12
    axis = w / wnorm
    midpoint = 0.5 * (mu0 + mu1)

    score = (X[test_idx] - midpoint) @ w
    geometric_label = (score > 0).astype(int)
    probe_accuracy = float(np.mean(geometric_label == labels[test_idx]))

    # Only use correctly classified held-out states for the controlled boundary test.
    valid = test_idx[geometric_label == labels[test_idx]][:128]
    base = states[valid].clone()
    base_label = labels[valid]

    score = (base.cpu().numpy() - midpoint) @ w
    margin = np.abs(score) / wnorm
    toward_other = -np.sign(score)[:, None] * axis[None, :]

    # Halfway to the fitted boundary: negative control.
    half = base + torch.tensor(
        (0.50 * margin)[:, None] * toward_other,
        dtype=torch.float32,
    )

    # Cross the fitted causal boundary by 25%.
    displacement = (1.25 * margin)[:, None]
    across = base + torch.tensor(
        displacement * toward_other,
        dtype=torch.float32,
    )

    # Same displacement norm, but explicitly remove the causal-axis component.
    random_direction = rng.normal(size=(len(base), X.shape[1]))
    random_direction -= (
        (random_direction @ axis)[:, None] * axis[None, :]
    )
    random_direction /= (
        np.linalg.norm(random_direction, axis=1, keepdims=True) + 1e-12
    )
    orthogonal = base + torch.tensor(
        displacement * random_direction,
        dtype=torch.float32,
    )

    major = discovered["major"]
    half_label, half_distance = classify_by_full_response(model, half, major)
    across_label, across_distance = classify_by_full_response(model, across, major)
    orth_label, orth_distance = classify_by_full_response(model, orthogonal, major)

    return {
        "seed": seed,
        "steps": steps,
        "organism": "Gate-5 decimal-addition GRU, hidden=16",
        "state_labels": "discovered from full 100-input counterfactual response signatures",
        "signature_exact_fraction": discovered["exact_fraction"],
        "dominant_signature_counts": discovered["counts"],
        "linear_geometry_probe_accuracy": probe_accuracy,
        "median_boundary_margin": float(np.median(margin)),
        "half_margin_axis_flip_rate": float(np.mean(half_label != base_label)),
        "cross_boundary_axis_flip_rate": float(np.mean(across_label != base_label)),
        "matched_orthogonal_flip_rate": float(np.mean(orth_label != base_label)),
        "cross_boundary_signature_hamming_to_nearest_class": float(
            np.mean(across_distance)
        ),
        "matched_orthogonal_signature_hamming_to_nearest_class": float(
            np.mean(orth_distance)
        ),
        "checkpoints": checkpoints,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--out", default="results/gate6_seed0.json")
    args = p.parse_args()

    receipt = run(args.seed, args.steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
