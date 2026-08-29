#!/usr/bin/env python3
"""
Gate 10B: discover P-KAS operator regimes from an unlabeled transition stream.

Gate 10A was deliberately generous:
    the decoder was told whether each transition was FREE, GROW, or PRUNE.

Gate 10B removes that label and also removes the growth target relation.

The decoder sees only:
    phase_before
    phase_after
    W_before
    W_after
    dt

The stream is shuffled.

The historical core still uses three known kinds of transitions:
    free Kuramoto-like phase evolution,
    one-pair symmetric growth,
    global multiplicative pruning + thresholding.

For growth, the hidden organism draws target relations from three historical values
that appeared in the old adapters:
    0.1              community-style same-phase growth
    pi/2             SAT adapter
    2*pi/3           3-color graph adapter

Those target values are NOT exposed to the decoder.

The decoder:
1. partitions transitions only by what changed;
2. fits the phase operator to the phase-only regime;
3. fits pruning from dense W transitions;
4. finds the changed pair in local W transitions;
5. infers how many hidden phase-target modes exist from the data;
6. fits a shared Gaussian phase-error growth law.

Ground-truth labels and constants are retained only for post-hoc scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


TWOPI = 2.0 * np.pi


def wrapped_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return (after - before + np.pi) % TWOPI - np.pi


class HiddenPKASStreamGenerator:
    """Known organism. Decoder below never reads these hidden constants."""

    def __init__(self, n: int = 16, seed: int = 0):
        self.n = n
        self.rng = np.random.default_rng(seed)

        self._K = 0.87
        self._ETA = 0.045
        self._SIGMA = 0.3 * np.pi
        self._PRUNE_RATE = 0.005
        self._PRUNE_THRESH = 0.005
        self._DT = 0.01
        self._NOISE = 0.03

        self._TARGETS = np.array(
            [0.1, np.pi / 2.0, 2.0 * np.pi / 3.0],
            dtype=float,
        )

    def _random_symmetric_W(self, low: float, high: float) -> np.ndarray:
        w = self.rng.uniform(low, high, (self.n, self.n))
        w = 0.5 * (w + w.T)
        np.fill_diagonal(w, 0.0)
        return w

    def free_transition(self):
        phase0 = self.rng.random(self.n) * TWOPI
        W = self._random_symmetric_W(0.005, 0.15)

        coupling = np.array(
            [
                np.sum(
                    W[i]
                    * np.sin(phase0 - phase0[i])
                )
                for i in range(self.n)
            ],
            dtype=float,
        )

        noise = self._NOISE * (
            self.rng.random(self.n) - 0.5
        )

        phase1 = np.mod(
            phase0
            + self._DT
            * (
                self._K * coupling
                + noise
            ),
            TWOPI,
        )

        record = {
            "phase_before": phase0,
            "phase_after": phase1,
            "W_before": W,
            "W_after": W.copy(),
            "dt": self._DT,
        }
        truth = {
            "operation": "free",
            "target_relation": None,
        }
        return record, truth

    def grow_transition(self):
        phase0 = self.rng.random(self.n) * TWOPI
        W0 = self._random_symmetric_W(0.0, 0.05)
        W1 = W0.copy()

        i, j = self.rng.choice(
            self.n,
            2,
            replace=False,
        )
        target = float(
            self.rng.choice(self._TARGETS)
        )

        diff = abs(
            phase0[i] - phase0[j]
        )
        diff = min(diff, TWOPI - diff)

        reward = np.exp(
            -0.5
            * (
                (diff - target)
                / self._SIGMA
            )
            ** 2
        )

        value = min(
            0.20,
            W0[i, j]
            + self._ETA * reward,
        )
        W1[i, j] = value
        W1[j, i] = value

        record = {
            "phase_before": phase0,
            "phase_after": phase0.copy(),
            "W_before": W0,
            "W_after": W1,
            "dt": self._DT,
        }
        truth = {
            "operation": "grow",
            "target_relation": target,
            "pair": [int(i), int(j)],
        }
        return record, truth

    def prune_transition(self):
        phase0 = self.rng.random(self.n) * TWOPI
        W0 = self._random_symmetric_W(0.0, 0.02)

        W1 = W0 * (
            1.0 - self._PRUNE_RATE
        )
        W1[
            W1 < self._PRUNE_THRESH
        ] = 0.0
        np.fill_diagonal(W1, 0.0)

        record = {
            "phase_before": phase0,
            "phase_after": phase0.copy(),
            "W_before": W0,
            "W_after": W1,
            "dt": self._DT,
        }
        truth = {
            "operation": "prune",
            "target_relation": None,
        }
        return record, truth


def make_unlabeled_stream(
    seed: int,
    n_free: int = 700,
    n_grow: int = 900,
    n_prune: int = 500,
):
    organism = HiddenPKASStreamGenerator(
        n=16,
        seed=seed,
    )

    records = []
    truth = []

    for generator, count in [
        (organism.free_transition, n_free),
        (organism.grow_transition, n_grow),
        (organism.prune_transition, n_prune),
    ]:
        for _ in range(count):
            record, hidden = generator()
            records.append(record)
            truth.append(hidden)

    order = organism.rng.permutation(
        len(records)
    )

    return (
        [records[i] for i in order],
        [truth[i] for i in order],
    )


def transition_change_signature(
    record: dict[str, object],
) -> dict[str, float | int]:
    p0 = np.asarray(
        record["phase_before"],
        dtype=float,
    )
    p1 = np.asarray(
        record["phase_after"],
        dtype=float,
    )
    W0 = np.asarray(
        record["W_before"],
        dtype=float,
    )
    W1 = np.asarray(
        record["W_after"],
        dtype=float,
    )

    phase_delta = wrapped_delta(
        p0,
        p1,
    )
    dW = W1 - W0
    upper = np.triu_indices_from(
        dW,
        1,
    )
    changed = np.abs(
        dW[upper]
    ) > 1e-12

    return {
        "phase_max_change": float(
            np.max(
                np.abs(phase_delta)
            )
        ),
        "changed_undirected_edges": int(
            changed.sum()
        ),
        "weight_delta_rms": float(
            np.sqrt(
                np.mean(
                    dW[upper] ** 2
                )
            )
        ),
    }


def discover_regime(
    record: dict[str, object],
) -> str:
    """
    No operation label is read.

    The first regime atlas uses only a simple invariant:
    what block of the state changed, and how spatially local was that change?
    """
    sig = transition_change_signature(
        record
    )

    phase_moves = (
        sig["phase_max_change"]
        > 1e-12
    )
    changed_edges = int(
        sig[
            "changed_undirected_edges"
        ]
    )

    if phase_moves and changed_edges == 0:
        return "phase_only"

    if (
        not phase_moves
        and changed_edges == 1
    ):
        return "pair_weight"

    if (
        not phase_moves
        and changed_edges > 1
    ):
        return "global_weight"

    return "unclassified"


def score_regime_discovery(
    discovered: list[str],
    truth: list[dict[str, object]],
) -> dict[str, object]:
    map_truth = {
        "free": "phase_only",
        "grow": "pair_weight",
        "prune": "global_weight",
    }

    expected = [
        map_truth[
            str(item["operation"])
        ]
        for item in truth
    ]

    names = [
        "phase_only",
        "pair_weight",
        "global_weight",
        "unclassified",
    ]

    confusion = {
        row: {
            col: 0
            for col in names
        }
        for row in names[:-1]
    }

    for e, p in zip(
        expected,
        discovered,
    ):
        confusion[e][p] += 1

    return {
        "accuracy": float(
            np.mean(
                np.asarray(discovered)
                == np.asarray(expected)
            )
        ),
        "confusion": confusion,
    }


def coupling_feature(
    phase: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            np.sum(
                W[i]
                * np.sin(
                    phase - phase[i]
                )
            )
            for i in range(
                len(phase)
            )
        ],
        dtype=float,
    )


def decode_phase_regime(
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
        W = np.asarray(
            record["W_before"],
            dtype=float,
        )
        dt = float(
            record["dt"]
        )

        features.append(
            coupling_feature(
                phase0,
                W,
            )
        )
        velocities.append(
            wrapped_delta(
                phase0,
                phase1,
            )
            / dt
        )

    x = np.concatenate(features)
    y = np.concatenate(velocities)

    gain = float(
        np.dot(x, y)
        / (
            np.dot(x, x)
            + 1e-15
        )
    )
    residual = y - gain * x

    noise_width = float(
        np.sqrt(12.0)
        * np.std(residual)
    )

    r2 = float(
        1.0
        - np.sum(
            residual ** 2
        )
        / (
            np.sum(
                (y - y.mean()) ** 2
            )
            + 1e-15
        )
    )

    return {
        "coupling_gain_hat": gain,
        "residual_uniform_width_hat": noise_width,
        "model_r2": r2,
    }


def decode_prune_regime(
    records: list[dict[str, object]],
) -> dict[str, float]:
    ratios = []

    for record in records:
        W0 = np.asarray(
            record["W_before"],
            dtype=float,
        )
        W1 = np.asarray(
            record["W_after"],
            dtype=float,
        )
        upper = np.triu_indices_from(
            W0,
            1,
        )

        before = W0[upper]
        after = W1[upper]
        survived = after > 0.0

        ratios.extend(
            (
                after[survived]
                / before[survived]
            ).tolist()
        )

    keep = float(
        np.median(ratios)
    )

    zero_candidates = []
    surviving_values = []

    for record in records:
        W0 = np.asarray(
            record["W_before"],
            dtype=float,
        )
        W1 = np.asarray(
            record["W_after"],
            dtype=float,
        )
        upper = np.triu_indices_from(
            W0,
            1,
        )

        before = W0[upper]
        after = W1[upper]
        candidate = keep * before

        zero_candidates.extend(
            candidate[
                after == 0.0
            ].tolist()
        )
        surviving_values.extend(
            after[
                after > 0.0
            ].tolist()
        )

    below = float(
        max(zero_candidates)
    )
    above = float(
        min(surviving_values)
    )
    threshold = 0.5 * (
        below + above
    )

    return {
        "multiplicative_keep_factor_hat": keep,
        "prune_rate_hat": float(
            1.0 - keep
        ),
        "threshold_lower_bound": below,
        "threshold_upper_bound": above,
        "threshold_hat": threshold,
    }


def kmeans_1d(
    x: np.ndarray,
    k: int,
    seed: int,
    restarts: int = 30,
):
    rng = np.random.default_rng(seed)
    x = np.asarray(
        x,
        dtype=float,
    )

    best = None

    for _ in range(restarts):
        centers = np.sort(
            rng.choice(
                x,
                k,
                replace=False,
            )
        )

        for _ in range(100):
            labels = np.argmin(
                np.abs(
                    x[:, None]
                    - centers[None, :]
                ),
                axis=1,
            )

            new_centers = np.array(
                [
                    (
                        x[labels == j].mean()
                        if np.any(
                            labels == j
                        )
                        else centers[j]
                    )
                    for j in range(k)
                ]
            )
            new_centers = np.sort(
                new_centers
            )

            if np.allclose(
                new_centers,
                centers,
                atol=1e-12,
                rtol=0.0,
            ):
                centers = new_centers
                break

            centers = new_centers

        labels = np.argmin(
            np.abs(
                x[:, None]
                - centers[None, :]
            ),
            axis=1,
        )
        sse = float(
            np.sum(
                (
                    x
                    - centers[labels]
                )
                ** 2
            )
        )

        if (
            best is None
            or sse < best[0]
        ):
            best = (
                sse,
                labels.copy(),
                centers.copy(),
            )

    return best


def silhouette_1d(
    x: np.ndarray,
    labels: np.ndarray,
) -> float:
    x = np.asarray(
        x,
        dtype=float,
    )
    labels = np.asarray(
        labels,
        dtype=int,
    )

    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0

    scores = []

    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False

        if np.any(same):
            a = float(
                np.mean(
                    np.abs(
                        x[i] - x[same]
                    )
                )
            )
        else:
            a = 0.0

        b = min(
            float(
                np.mean(
                    np.abs(
                        x[i]
                        - x[
                            labels == other
                        ]
                    )
                )
            )
            for other in unique
            if other != labels[i]
        )

        scores.append(
            (b - a)
            / max(a, b, 1e-12)
        )

    return float(
        np.mean(scores)
    )


def extract_growth_observations(
    records: list[dict[str, object]],
):
    phase_diff = []
    delta = []
    symmetry_error = []

    for record in records:
        phase = np.asarray(
            record["phase_before"],
            dtype=float,
        )
        W0 = np.asarray(
            record["W_before"],
            dtype=float,
        )
        W1 = np.asarray(
            record["W_after"],
            dtype=float,
        )

        dW = W1 - W0
        upper = np.triu_indices_from(
            dW,
            1,
        )
        magnitude = np.abs(
            dW[upper]
        )
        changed_index = int(
            np.argmax(magnitude)
        )

        i = int(
            upper[0][changed_index]
        )
        j = int(
            upper[1][changed_index]
        )

        diff = abs(
            phase[i] - phase[j]
        )
        diff = min(
            diff,
            TWOPI - diff,
        )

        phase_diff.append(
            diff
        )
        delta.append(
            float(dW[i, j])
        )
        symmetry_error.append(
            float(
                abs(
                    dW[i, j]
                    - dW[j, i]
                )
            )
        )

    return (
        np.asarray(
            phase_diff,
            dtype=float,
        ),
        np.asarray(
            delta,
            dtype=float,
        ),
        np.asarray(
            symmetry_error,
            dtype=float,
        ),
    )


def infer_growth_targets(
    phase_diff: np.ndarray,
    delta: np.ndarray,
    seed: int,
):
    # Strongest growth events lie nearest hidden preferred relations.
    high = (
        delta
        >= np.quantile(
            delta,
            0.95,
        )
    )
    x = phase_diff[high]

    silhouette = {}
    models = {}

    for k in range(2, 6):
        _sse, labels, centers = kmeans_1d(
            x,
            k,
            seed + k,
        )
        score = silhouette_1d(
            x,
            labels,
        )
        silhouette[k] = score
        models[k] = centers

    best_k = max(
        silhouette,
        key=silhouette.get,
    )
    initial_centers = models[
        best_k
    ]

    # Jointly refine centers, amplitude and shared width on the strongest 15%.
    fit_mask = (
        delta
        >= np.quantile(
            delta,
            0.85,
        )
    )
    x_fit = phase_diff[fit_mask]
    y_fit = np.log(
        delta[fit_mask]
    )

    def residual(params):
        centers = np.sort(
            params[:best_k]
        )
        log_eta = params[
            best_k
        ]
        sigma = np.exp(
            params[
                best_k + 1
            ]
        )

        assignment = np.argmin(
            np.abs(
                x_fit[:, None]
                - centers[None, :]
            ),
            axis=1,
        )

        prediction = (
            log_eta
            - 0.5
            * (
                (
                    x_fit
                    - centers[
                        assignment
                    ]
                )
                / sigma
            )
            ** 2
        )
        return (
            y_fit - prediction
        )

    start = np.concatenate(
        [
            initial_centers,
            [
                np.log(
                    delta.max()
                ),
                np.log(
                    0.3 * np.pi
                ),
            ],
        ]
    )

    lower = np.concatenate(
        [
            np.zeros(best_k),
            [
                np.log(1e-4),
                np.log(
                    0.05 * np.pi
                ),
            ],
        ]
    )
    upper = np.concatenate(
        [
            np.full(
                best_k,
                np.pi,
            ),
            [
                np.log(0.2),
                np.log(np.pi),
            ],
        ]
    )

    fit = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=3000,
    )

    centers = np.sort(
        fit.x[:best_k]
    )
    eta = float(
        np.exp(
            fit.x[best_k]
        )
    )
    sigma = float(
        np.exp(
            fit.x[
                best_k + 1
            ]
        )
    )

    err = residual(
        fit.x
    )
    r2 = float(
        1.0
        - np.sum(
            err ** 2
        )
        / (
            np.sum(
                (
                    y_fit
                    - y_fit.mean()
                )
                ** 2
            )
            + 1e-15
        )
    )

    return {
        "target_mode_count": int(
            best_k
        ),
        "target_modes_hat": centers.tolist(),
        "silhouette_by_k": {
            str(k): float(v)
            for k, v in silhouette.items()
        },
        "growth_amplitude_hat": eta,
        "growth_width_hat": sigma,
        "growth_width_over_pi_hat": float(
            sigma / np.pi
        ),
        "growth_log_gaussian_fit_r2": r2,
    }


def decode_growth_regime(
    records: list[dict[str, object]],
    seed: int,
):
    (
        phase_diff,
        delta,
        symmetry_error,
    ) = extract_growth_observations(
        records
    )

    result = infer_growth_targets(
        phase_diff,
        delta,
        seed,
    )
    result.update(
        {
            "pair_update_is_symmetric": bool(
                np.max(
                    symmetry_error
                )
                < 1e-12
            ),
            "max_pair_update_asymmetry": float(
                np.max(
                    symmetry_error
                )
            ),
        }
    )
    return result


def run_seed(seed: int):
    records, truth = make_unlabeled_stream(
        seed
    )

    discovered = [
        discover_regime(record)
        for record in records
    ]

    score = score_regime_discovery(
        discovered,
        truth,
    )

    groups = {
        name: [
            record
            for record, label
            in zip(
                records,
                discovered,
            )
            if label == name
        ]
        for name in [
            "phase_only",
            "pair_weight",
            "global_weight",
        ]
    }

    return {
        "seed": seed,
        "regime_discovery": score,
        "regime_sizes": {
            name: len(rows)
            for name, rows
            in groups.items()
        },
        "phase_only": decode_phase_regime(
            groups["phase_only"]
        ),
        "pair_weight": decode_growth_regime(
            groups["pair_weight"],
            seed,
        ),
        "global_weight": decode_prune_regime(
            groups["global_weight"]
        ),
    }


def summarize(
    receipts: list[dict[str, object]],
):
    def collect(
        block: str,
        key: str,
    ):
        values = [
            float(
                receipt[block][key]
            )
            for receipt in receipts
        ]
        return {
            "values": values,
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
        }

    return {
        "regime_accuracy": {
            "values": [
                float(
                    r[
                        "regime_discovery"
                    ][
                        "accuracy"
                    ]
                )
                for r in receipts
            ],
            "mean": float(
                np.mean(
                    [
                        r[
                            "regime_discovery"
                        ][
                            "accuracy"
                        ]
                        for r in receipts
                    ]
                )
            ),
        },
        "phase_coupling_gain_hat": collect(
            "phase_only",
            "coupling_gain_hat",
        ),
        "phase_noise_width_hat": collect(
            "phase_only",
            "residual_uniform_width_hat",
        ),
        "phase_model_r2": collect(
            "phase_only",
            "model_r2",
        ),
        "growth_target_mode_count": {
            "values": [
                int(
                    r[
                        "pair_weight"
                    ][
                        "target_mode_count"
                    ]
                )
                for r in receipts
            ]
        },
        "growth_target_modes_hat": [
            r[
                "pair_weight"
            ][
                "target_modes_hat"
            ]
            for r in receipts
        ],
        "growth_amplitude_hat": collect(
            "pair_weight",
            "growth_amplitude_hat",
        ),
        "growth_width_over_pi_hat": collect(
            "pair_weight",
            "growth_width_over_pi_hat",
        ),
        "growth_fit_r2": collect(
            "pair_weight",
            "growth_log_gaussian_fit_r2",
        ),
        "symmetric_growth_seeds": int(
            sum(
                bool(
                    r[
                        "pair_weight"
                    ][
                        "pair_update_is_symmetric"
                    ]
                )
                for r in receipts
            )
        ),
        "prune_rate_hat": collect(
            "global_weight",
            "prune_rate_hat",
        ),
        "prune_threshold_hat": collect(
            "global_weight",
            "threshold_hat",
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
    )
    p.add_argument(
        "--out",
        default="results/gate10b_pkas_unlabeled_summary.json",
    )
    args = p.parse_args()

    receipts = [
        run_seed(seed)
        for seed in range(
            args.seeds
        )
    ]

    result = {
        "experiment": (
            "Gate 10B P-KAS unlabeled transition-stream decoding"
        ),
        "decoder_access": (
            "phase_before/after, W_before/after, dt only; "
            "no operation label, no growth pair, no growth target"
        ),
        "seeds": list(
            range(args.seeds)
        ),
        "receipts": receipts,
        "summary": summarize(
            receipts
        ),
        "ground_truth_for_posthoc_scoring_only": {
            "regimes": [
                "free",
                "grow",
                "prune",
            ],
            "coupling_gain": 0.87,
            "noise_width": 0.03,
            "growth_amplitude": 0.045,
            "growth_width_over_pi": 0.3,
            "growth_target_modes": [
                0.1,
                float(
                    np.pi / 2.0
                ),
                float(
                    2.0 * np.pi / 3.0
                ),
            ],
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
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        json.dumps(
            result["summary"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
