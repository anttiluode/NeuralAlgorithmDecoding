#!/usr/bin/env python3
"""
Gate 10A: P-KAS black-box equation recovery.

This is a calibration organism for the stronger NeuralAlgorithmDecoding goal.

We reimplement the published P-KAS core dynamics in a sealed "organism" class.  The
decoder is not handed the constants or equations.  It receives only permitted
experiments:

    state before
    intervention label / control argument
    state after
    timestamps

The decoder then asks whether it can recover a compact executable description of:

1. free Kuramoto-like phase relaxation,
2. pairwise growth / plasticity,
3. global pruning,
4. symmetry of the learned W layer.

This is intentionally easier than decoding an arbitrary neural network:
- operation boundaries are known;
- state variables are visible;
- interventions are clean;
- the candidate equation language is small.

The point is to turn the manual "P-KAS Doors" style autopsy into an automated
system-identification receipt before hiding more of the organism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TWOPI = 2.0 * np.pi


class HiddenPKASCore:
    """Known organism. Decoder code below must not read these attributes directly."""

    def __init__(self, n: int = 16, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.n = n

        # Ground truth copied from the historical P-KAS core.
        self._K = 0.87
        self._ETA_W = 0.045
        self._PRUNE_RATE = 0.005
        self._PRUNE_THRESH = 0.005
        self._DT = 0.01
        self._NOISE = 0.03

        self.phases = self.rng.random(n) * TWOPI
        w = self.rng.uniform(0.0, 0.001, (n, n))
        self.W = 0.5 * (w + w.T)
        np.fill_diagonal(self.W, 0.0)

    def set_probe_weights(self, low: float = 0.005, high: float = 0.15):
        w = self.rng.uniform(low, high, (self.n, self.n))
        self.W = 0.5 * (w + w.T)
        np.fill_diagonal(self.W, 0.0)

    def set_phase_probe(self):
        self.phases = self.rng.random(self.n) * TWOPI

    def free_step(self):
        """One unforced phase step. Return only observable before/after data."""
        before = self.phases.copy()
        W = self.W.copy()

        feature = np.array(
            [
                np.sum(
                    W[i]
                    * np.sin(before - before[i])
                )
                for i in range(self.n)
            ],
            dtype=float,
        )
        noise = self._NOISE * (
            self.rng.random(self.n) - 0.5
        )
        after = np.mod(
            before
            + self._DT
            * (self._K * feature + noise),
            TWOPI,
        )
        self.phases = after

        return {
            "phase_before": before,
            "phase_after": after,
            "W": W,
            "dt": self._DT,
        }

    def grow(self, i: int, j: int, target_relation: float):
        """Historical P-KAS pairwise growth operation."""
        before = self.W.copy()

        diff = abs(
            self.phases[i] - self.phases[j]
        )
        diff = min(diff, TWOPI - diff)

        reward = np.exp(
            -0.5
            * (
                (diff - target_relation)
                / (0.3 * np.pi)
            )
            ** 2
        )
        value = min(
            0.20,
            self.W[i, j]
            + self._ETA_W * reward,
        )
        self.W[i, j] = value
        self.W[j, i] = value

        return {
            "i": i,
            "j": j,
            "target_relation": target_relation,
            "phase_before": self.phases.copy(),
            "W_before": before,
            "W_after": self.W.copy(),
        }

    def prune(self):
        before = self.W.copy()
        self.W *= 1.0 - self._PRUNE_RATE
        self.W[
            self.W < self._PRUNE_THRESH
        ] = 0.0
        np.fill_diagonal(self.W, 0.0)

        return {
            "W_before": before,
            "W_after": self.W.copy(),
        }

    def set_single_edge(self, i: int, j: int, value: float):
        self.W[:] = 0.0
        self.W[i, j] = value
        self.W[j, i] = value


def wrapped_velocity(
    before: np.ndarray,
    after: np.ndarray,
    dt: float,
) -> np.ndarray:
    delta = (
        (after - before + np.pi)
        % TWOPI
        - np.pi
    )
    return delta / dt


def coupling_feature(
    phases: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            np.sum(
                W[i]
                * np.sin(phases - phases[i])
            )
            for i in range(len(phases))
        ]
    )


def decode_free_dynamics(
    records: list[dict[str, object]],
) -> dict[str, float]:
    features = []
    velocities = []

    for record in records:
        phase0 = np.asarray(
            record["phase_before"],
            dtype=float,
        )
        phase1 = np.asarray(
            record["phase_after"],
            dtype=float,
        )
        W = np.asarray(record["W"], dtype=float)
        dt = float(record["dt"])

        features.append(
            coupling_feature(phase0, W)
        )
        velocities.append(
            wrapped_velocity(
                phase0,
                phase1,
                dt,
            )
        )

    x = np.concatenate(features)
    y = np.concatenate(velocities)

    # Smallest first model: v ~= K * sum_j W_ij sin(phi_j-phi_i)
    k_hat = float(
        np.dot(x, y)
        / (np.dot(x, x) + 1e-15)
    )
    residual = y - k_hat * x

    # If residual is approximately uniform zero-mean drive, width is sqrt(12)*std.
    noise_width_hat = float(
        np.sqrt(12.0)
        * np.std(residual)
    )

    prediction = k_hat * x
    r2 = float(
        1.0
        - np.sum((y - prediction) ** 2)
        / (
            np.sum((y - y.mean()) ** 2)
            + 1e-15
        )
    )

    return {
        "coupling_gain_hat": k_hat,
        "residual_uniform_width_hat": noise_width_hat,
        "velocity_model_r2": r2,
        "residual_mean": float(
            residual.mean()
        ),
        "residual_std": float(
            residual.std()
        ),
    }


def decode_growth(
    records: list[dict[str, object]],
) -> dict[str, object]:
    squared_error = []
    log_delta = []
    symmetry_error = []

    for record in records:
        i = int(record["i"])
        j = int(record["j"])
        target = float(
            record["target_relation"]
        )
        phases = np.asarray(
            record["phase_before"],
            dtype=float,
        )
        before = np.asarray(
            record["W_before"],
            dtype=float,
        )
        after = np.asarray(
            record["W_after"],
            dtype=float,
        )

        diff = abs(phases[i] - phases[j])
        diff = min(diff, TWOPI - diff)

        delta_ij = after[i, j] - before[i, j]
        delta_ji = after[j, i] - before[j, i]
        symmetry_error.append(
            abs(delta_ij - delta_ji)
        )

        # Ignore capped points for functional-form recovery.
        if (
            delta_ij > 1e-10
            and after[i, j] < 0.199
        ):
            squared_error.append(
                (diff - target) ** 2
            )
            log_delta.append(
                np.log(delta_ij)
            )

    x = np.asarray(squared_error)
    y = np.asarray(log_delta)

    # Candidate discovered from the trace:
    # log(delta W) = intercept + slope * (phase_error)^2
    slope, intercept = np.polyfit(x, y, 1)

    eta_hat = float(np.exp(intercept))
    sigma_hat = float(
        np.sqrt(
            -1.0 / (2.0 * slope)
        )
    )

    pred = intercept + slope * x
    r2 = float(
        1.0
        - np.sum((y - pred) ** 2)
        / (
            np.sum((y - y.mean()) ** 2)
            + 1e-15
        )
    )

    return {
        "pair_update_is_symmetric": bool(
            max(symmetry_error) < 1e-12
        ),
        "max_pair_update_asymmetry": float(
            max(symmetry_error)
        ),
        "growth_amplitude_hat": eta_hat,
        "growth_phase_width_hat": sigma_hat,
        "growth_phase_width_over_pi": float(
            sigma_hat / np.pi
        ),
        "log_gaussian_fit_r2": r2,
        "decoded_law": (
            "delta_w ~= eta * exp(-0.5 * "
            "((wrapped_phase_difference-target)/sigma)^2)"
        ),
    }


def decode_pruning(
    organism: HiddenPKASCore,
) -> dict[str, float]:
    # First infer multiplicative shrinkage on a safely surviving weight.
    organism.set_single_edge(
        0,
        1,
        0.10,
    )
    record = organism.prune()
    before = float(
        record["W_before"][0, 1]
    )
    after = float(
        record["W_after"][0, 1]
    )
    keep_factor = after / before
    prune_rate_hat = 1.0 - keep_factor

    # Active experiment: binary-search the pre-prune survival boundary.
    lo = 0.0
    hi = 0.02

    for _ in range(35):
        mid = 0.5 * (lo + hi)
        organism.set_single_edge(
            0,
            1,
            mid,
        )
        outcome = organism.prune()
        survived = (
            outcome["W_after"][0, 1]
            > 0.0
        )

        if survived:
            hi = mid
        else:
            lo = mid

    pre_prune_boundary = 0.5 * (lo + hi)
    threshold_hat = (
        pre_prune_boundary
        * keep_factor
    )

    return {
        "multiplicative_keep_factor_hat": float(
            keep_factor
        ),
        "prune_rate_hat": float(
            prune_rate_hat
        ),
        "pre_prune_survival_boundary_hat": float(
            pre_prune_boundary
        ),
        "post_decay_threshold_hat": float(
            threshold_hat
        ),
        "active_binary_search_iterations": 35,
    }


def collect_and_decode(seed: int) -> dict[str, object]:
    organism = HiddenPKASCore(
        n=16,
        seed=seed,
    )
    organism.set_probe_weights()

    free_records = [
        organism.free_step()
        for _ in range(600)
    ]

    growth_records = []
    for _ in range(400):
        organism.set_phase_probe()
        i, j = organism.rng.choice(
            organism.n,
            2,
            replace=False,
        )
        target = organism.rng.uniform(
            0.0,
            np.pi,
        )

        # Keep the edge away from the 0.20 saturation cap.
        value = organism.rng.uniform(
            0.0,
            0.05,
        )
        organism.W[i, j] = value
        organism.W[j, i] = value

        growth_records.append(
            organism.grow(
                int(i),
                int(j),
                float(target),
            )
        )

    return {
        "seed": seed,
        "free_dynamics": decode_free_dynamics(
            free_records
        ),
        "growth": decode_growth(
            growth_records
        ),
        "pruning": decode_pruning(
            organism
        ),
    }


def summarize(
    receipts: list[dict[str, object]],
) -> dict[str, object]:
    paths = {
        "coupling_gain_hat": (
            "free_dynamics",
            "coupling_gain_hat",
        ),
        "noise_width_hat": (
            "free_dynamics",
            "residual_uniform_width_hat",
        ),
        "free_model_r2": (
            "free_dynamics",
            "velocity_model_r2",
        ),
        "growth_amplitude_hat": (
            "growth",
            "growth_amplitude_hat",
        ),
        "growth_width_over_pi": (
            "growth",
            "growth_phase_width_over_pi",
        ),
        "growth_fit_r2": (
            "growth",
            "log_gaussian_fit_r2",
        ),
        "prune_rate_hat": (
            "pruning",
            "prune_rate_hat",
        ),
        "prune_threshold_hat": (
            "pruning",
            "post_decay_threshold_hat",
        ),
    }

    output = {}
    for name, (block, key) in paths.items():
        values = [
            float(receipt[block][key])
            for receipt in receipts
        ]
        output[name] = {
            "values": values,
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
        }

    output["symmetric_growth_seeds"] = int(
        sum(
            bool(
                receipt["growth"][
                    "pair_update_is_symmetric"
                ]
            )
            for receipt in receipts
        )
    )

    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
    )
    p.add_argument(
        "--out",
        default="results/gate10a_pkas_blackbox_summary.json",
    )
    args = p.parse_args()

    receipts = [
        collect_and_decode(seed)
        for seed in range(args.seeds)
    ]

    result = {
        "experiment": "Gate 10A P-KAS black-box equation recovery",
        "seeds": list(range(args.seeds)),
        "decoder_access": (
            "states before/after, operation labels/control arguments, timestamps; "
            "no organism constants read by decoder"
        ),
        "receipts": receipts,
        "summary": summarize(receipts),
        "ground_truth_for_posthoc_scoring_only": {
            "coupling_gain": 0.87,
            "noise_width": 0.03,
            "growth_amplitude": 0.045,
            "growth_width_over_pi": 0.3,
            "prune_rate": 0.005,
            "prune_threshold": 0.005,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(result, indent=2)
    )
    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
