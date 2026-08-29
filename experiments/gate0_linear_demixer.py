#!/usr/bin/env python3
"""
Gate 0: decode a fuzzy neural demixer back into a compact matrix.

Calibration task:
    s -> A -> x -> tiny tanh MLP -> s_hat

The hidden neural implementation is nonlinear/distributed, but the true task law is
linear: s = A^-1 x.

The decoder measures:
- effective affine operator from network I/O,
- mean input-output Jacobian from learned weights/activations,
- causal hidden-unit ablation importance,
- smallest hidden-unit subset reproducing the network to 1% NMSE,
- OOD extrapolation of neural network vs extracted math.

This is intentionally NOT blind source separation and NOT a novelty claim.
It is the simplest model organism for:
    neural approximation -> decoded mathematical operation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


A = torch.tensor([[1.0, 0.8], [0.4, 1.2]], dtype=torch.float32)
A_INV = torch.linalg.inv(A)


class Demixer(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        h = torch.tanh(self.fc1(x))
        y = self.fc2(h)
        return (y, h) if return_hidden else y


def make_data(n: int, seed: int, scale: float = 0.35):
    g = torch.Generator().manual_seed(seed)

    # Two independent, non-Gaussian sources with roughly matched variance.
    u = torch.rand(n, generator=g) - 0.5
    s1 = -torch.sign(u) * torch.log1p(-2 * torch.abs(u) + 1e-7) / math.sqrt(2.0)
    s2 = (torch.rand(n, generator=g) * 2 - 1) * math.sqrt(3.0)

    s = torch.stack([s1, s2], dim=1) * scale
    x = s @ A.T
    return x, s


def nmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a - b) ** 2) / (torch.var(b) + 1e-12))


def rel_fro(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a - b) / (torch.linalg.norm(b) + 1e-12))


def effective_affine(model: Demixer, x: torch.Tensor):
    with torch.no_grad():
        y = model(x)

    # Fit y = x @ B.T + b to the network, not to source labels.
    X = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    theta = torch.linalg.lstsq(X, y).solution
    B = theta[:2].T
    b = theta[2]
    return B, b


def average_jacobian(model: Demixer, x: torch.Tensor):
    """
    Exact Jacobian of the tiny tanh MLP, averaged over probe inputs.

    J_n[o,i] = sum_h W2[o,h] * tanh'(z_n[h]) * W1[h,i]
    """
    with torch.no_grad():
        z = model.fc1(x)
        d = 1 - torch.tanh(z) ** 2
        w1 = model.fc1.weight
        w2 = model.fc2.weight
        J = torch.einsum("oh,nh,hi->noi", w2, d, w1)
    return J.mean(0)


def ablation_importance(model: Demixer, x: torch.Tensor):
    with torch.no_grad():
        y, h = model(x, return_hidden=True)
        values = []
        for j in range(h.shape[1]):
            h_ablated = h.clone()
            h_ablated[:, j] = 0
            y_ablated = model.fc2(h_ablated)
            values.append(float(torch.mean((y_ablated - y) ** 2)))
    return values


def smallest_subcircuit(model: Demixer, x: torch.Tensor, fidelity_nmse: float = 0.01):
    with torch.no_grad():
        y, h = model(x, return_hidden=True)
        importance = ablation_importance(model, x)
        order = np.argsort(importance)[::-1].copy()

        curve = []
        for k in range(1, len(order) + 1):
            mask = torch.zeros(h.shape[1])
            mask[order[:k]] = 1
            y_small = model.fc2(h * mask)
            fidelity = nmse(y_small, y)
            curve.append([k, fidelity])
            if fidelity <= fidelity_nmse:
                return k, importance, curve

    return len(order), importance, curve


def evaluate(model: Demixer, x: torch.Tensor, s: torch.Tensor):
    with torch.no_grad():
        pred = model(x)

    B, b = effective_affine(model, x)
    affine_pred = x @ B.T + b
    J = average_jacobian(model, x)
    k, importance, curve = smallest_subcircuit(model, x)

    return {
        "network_mse": float(torch.mean((pred - s) ** 2)),
        "network_nmse": nmse(pred, s),
        "ls_matrix": B.tolist(),
        "ls_bias": b.tolist(),
        "ls_vs_exact_rel_fro": rel_fro(B, A_INV),
        "ls_surrogate_vs_network_nmse": nmse(affine_pred, pred),
        "avg_jacobian": J.tolist(),
        "jacobian_vs_exact_rel_fro": rel_fro(J, A_INV),
        "circuit_units_for_1pct_fidelity": int(k),
        "unit_ablation_importance": importance,
        "circuit_curve": curve,
    }


def run(seed: int, steps: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Demixer(hidden=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    x_train, s_train = make_data(8192, seed + 1)
    x_test, s_test = make_data(4096, seed + 2)

    checkpoints = []
    checkpoint_steps = {0, 10, 25, 50, 100, 200, 500, 1000, 2000, steps}

    for step in range(steps + 1):
        if step in checkpoint_steps:
            receipt = evaluate(model, x_test[:1024], s_test[:1024])
            receipt["step"] = step
            checkpoints.append(receipt)

        if step == steps:
            break

        idx = torch.randint(0, len(x_train), (256,))
        pred = model(x_train[idx])
        loss = torch.mean((pred - s_train[idx]) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    final = evaluate(model, x_test, s_test)

    # Stronger amplitude, identical physical law.
    x_ood, s_ood = make_data(4096, seed + 3, scale=0.70)

    with torch.no_grad():
        neural_ood = model(x_ood)

    B = torch.tensor(final["ls_matrix"])
    b = torch.tensor(final["ls_bias"])
    math_ood = x_ood @ B.T + b

    final["ood_2x_network_nmse"] = nmse(neural_ood, s_ood)
    final["ood_2x_extracted_math_nmse"] = nmse(math_ood, s_ood)
    final["exact_inverse"] = A_INV.tolist()
    final["mixing_matrix"] = A.tolist()

    return {
        "seed": seed,
        "architecture": "2 -> tanh(16) -> 2",
        "task": "supervised recovery of two independent sources from a fixed linear mixture",
        "checkpoints": checkpoints,
        "final": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/gate0_seed0.json")
    args = parser.parse_args()

    receipt = run(args.seed, args.steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))

    scalar = {
        k: v
        for k, v in receipt["final"].items()
        if isinstance(v, (int, float))
    }
    print(json.dumps(scalar, indent=2))
    print("exact inverse:")
    print(A_INV.numpy())
    print("decoded affine matrix:")
    print(np.array(receipt["final"]["ls_matrix"]))
    print("average Jacobian:")
    print(np.array(receipt["final"]["avg_jacobian"]))


if __name__ == "__main__":
    main()
