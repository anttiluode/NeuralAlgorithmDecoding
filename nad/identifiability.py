"""Identifiability tools inherited from TransientWaveCompiler.

Sensitivity is not identifiability.

Given a response-space Jacobian/basis J for effects that an existing explanation
(or nuisance model) can already produce, and a candidate response direction g,
measure only the part of g that is orthogonal to span(J):

    eta = ||(I - P_J) g|| / ||g||

eta ~= 0:
    the candidate may have a large effect, but the current experiment cannot
    distinguish it from existing/nuisance degrees of freedom.

eta ~= 1:
    the candidate produces a response direction that is largely novel.

This is a local diagnostic, not a proof of global causal identifiability.
"""
from __future__ import annotations

import numpy as np


def orthogonal_novelty_fraction(
    nuisance_directions: np.ndarray,
    candidate_direction: np.ndarray,
    *,
    rcond: float = 1e-10,
) -> dict[str, float | int | None]:
    J = np.asarray(nuisance_directions, dtype=float)
    g = np.asarray(candidate_direction, dtype=float).reshape(-1)

    if J.ndim != 2:
        raise ValueError("nuisance_directions must be a matrix")
    if J.shape[0] != len(g):
        raise ValueError("response rows must match candidate length")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(g)):
        raise ValueError("inputs must be finite")
    if rcond <= 0 or not np.isfinite(rcond):
        raise ValueError("rcond must be positive and finite")

    gnorm = float(np.linalg.norm(g))
    if gnorm <= 1e-300:
        return {
            "novelty_fraction": 0.0,
            "projected_fraction": 0.0,
            "candidate_norm": gnorm,
            "residual_norm": 0.0,
            "rank": 0,
            "condition": None,
        }

    if J.shape[1] == 0:
        return {
            "novelty_fraction": 1.0,
            "projected_fraction": 0.0,
            "candidate_norm": gnorm,
            "residual_norm": gnorm,
            "rank": 0,
            "condition": None,
        }

    u, singular, _vh = np.linalg.svd(J, full_matrices=False)
    if len(singular) == 0 or singular[0] <= 0:
        rank = 0
    else:
        rank = int(np.sum(singular > rcond * singular[0]))

    if rank:
        basis = u[:, :rank]
        projected = basis @ (basis.T @ g)
        condition = float(singular[0] / singular[rank - 1])
    else:
        projected = np.zeros_like(g)
        condition = None

    residual = g - projected
    pnorm = float(np.linalg.norm(projected))
    rnorm = float(np.linalg.norm(residual))

    return {
        "novelty_fraction": float(np.clip(rnorm / gnorm, 0.0, 1.0)),
        "projected_fraction": float(np.clip(pnorm / gnorm, 0.0, 1.0)),
        "candidate_norm": gnorm,
        "residual_norm": rnorm,
        "rank": rank,
        "condition": condition,
    }


def whitened_orthogonal_novelty_fraction(
    nuisance_directions: np.ndarray,
    candidate_direction: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge: float = 1e-8,
    rcond: float = 1e-10,
) -> dict[str, float | int | None]:
    """Noise-whiten the response space before asking what is identifiable.

    Directions with large natural/measurement variance are discounted.  This is
    the neural-decoding analogue of the detectability lesson from TWC v0.7:
    breaking an exact ambiguity is not enough if the distinguishing residual is
    tiny relative to response noise.
    """
    J = np.asarray(nuisance_directions, dtype=float)
    g = np.asarray(candidate_direction, dtype=float).reshape(-1)
    C = np.asarray(covariance, dtype=float)

    if C.shape != (len(g), len(g)):
        raise ValueError("covariance shape mismatch")
    if not np.allclose(C, C.T, atol=1e-10, rtol=0):
        raise ValueError("covariance must be symmetric")

    eig, vec = np.linalg.eigh(C + ridge * np.eye(len(g)))
    eig = np.maximum(eig, ridge)
    W = vec @ np.diag(1.0 / np.sqrt(eig)) @ vec.T

    return orthogonal_novelty_fraction(
        W @ J,
        W @ g,
        rcond=rcond,
    )


def residualized_disagreement(
    nuisance_directions: np.ndarray,
    candidate_a: np.ndarray,
    candidate_b: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Score disagreement only after removing already-explainable directions."""
    delta = np.asarray(candidate_a, dtype=float).reshape(-1) - np.asarray(
        candidate_b, dtype=float
    ).reshape(-1)

    if covariance is None:
        result = orthogonal_novelty_fraction(nuisance_directions, delta)
    else:
        result = whitened_orthogonal_novelty_fraction(
            nuisance_directions,
            delta,
            covariance,
        )

    result = dict(result)
    result["raw_disagreement_norm"] = float(np.linalg.norm(delta))
    result["residualized_disagreement_norm"] = float(result["residual_norm"])
    return result
