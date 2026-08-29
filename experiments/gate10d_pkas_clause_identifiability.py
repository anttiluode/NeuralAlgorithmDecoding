#!/usr/bin/env python3
"""
Gate 10D: infer which SAT clause caused an intervention, and say NOT IDENTIFIABLE
when the transition cannot distinguish aliases.

This composes earlier recovered pieces:
- Gate 10B: local pairwise W growth leaves a sparse structural trace.
- Gate 10C: target sign follows clause[0], not the selected literal.

The decoder now does NOT receive the acted-on clause index.

It receives:
    all currently unsatisfied clauses
    selected/stamped variable
    target phase
    W_before
    W_after

The hidden adapter grows all three variable-pairs within the selected 3-literal clause.

Decoder:
1. infer the variable set from changed W-pair support;
2. retain clauses with that absolute variable set;
3. apply the previously decoded clause[0]-sign target rule;
4. return the unique clause if exactly one remains;
5. otherwise return NOT_IDENTIFIABLE plus the surviving candidate set.

The test contexts intentionally include alias clause pairs with the same absolute variable
set and the same clause[0] sign. Those aliases are behaviorally indistinguishable under
the declared observation family. A correct decoder must refuse to choose between them.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


TWOPI = 2.0 * np.pi


def random_clause(
    rng: np.random.Generator,
    num_vars: int = 20,
) -> list[int]:
    variables = rng.choice(
        np.arange(1, num_vars + 1),
        3,
        replace=False,
    )
    signs = rng.choice(
        [-1, 1],
        3,
    )
    return (
        variables * signs
    ).astype(int).tolist()


def clause_observable_signature(
    clause: list[int],
):
    """
    Signature visible from:
      all-pair W support + Gate-10C target rule.
    """
    variable_set = tuple(
        sorted(
            abs(literal) - 1
            for literal in clause
        )
    )
    first_positive = clause[0] > 0
    return (
        variable_set,
        first_positive,
    )


def make_context(
    rng: np.random.Generator,
    *,
    clause_count: int = 18,
    num_vars: int = 20,
    alias_pairs: int = 3,
):
    clauses = []

    # Deliberately insert aliases that the declared observation family cannot
    # distinguish: same absolute variable set, same first-literal sign, but
    # different non-first literal signs.
    for _ in range(alias_pairs):
        base = random_clause(
            rng,
            num_vars,
        )
        twin = base.copy()
        twin[1] *= -1
        if rng.random() < 0.5:
            twin[2] *= -1

        clauses.extend(
            [base, twin]
        )

    used = {
        clause_observable_signature(
            clause
        )
        for clause in clauses
    }

    # Remaining clauses have unique observable signatures.
    while len(clauses) < clause_count:
        clause = random_clause(
            rng,
            num_vars,
        )
        signature = clause_observable_signature(
            clause
        )

        if signature in used:
            continue

        used.add(signature)
        clauses.append(clause)

    rng.shuffle(clauses)
    return clauses


class HiddenSATTransitionGenerator:
    """Sealed historical-style adapter action plus its pair-growth structural trace."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._ETA = 0.045
        self._SIGMA = 0.3 * np.pi
        self._GROW_TARGET = np.pi / 2.0

    def action_batch(
        self,
        clauses: list[list[int]],
        max_actions: int = 10,
    ):
        selected_clause_ids = self.rng.choice(
            len(clauses),
            size=min(
                max_actions,
                len(clauses),
            ),
            replace=False,
        )

        num_vars = max(
            abs(literal)
            for clause in clauses
            for literal in clause
        )

        actions = []

        for clause_index in selected_clause_ids:
            clause = clauses[
                int(clause_index)
            ]

            selected_literal = int(
                self.rng.choice(
                    clause
                )
            )
            selected_variable = (
                abs(selected_literal) - 1
            )

            target_phase = (
                0.1
                if clause[0] > 0
                else np.pi + 0.1
            )

            phases = (
                self.rng.random(num_vars)
                * TWOPI
            )
            W0 = np.zeros(
                (num_vars, num_vars),
                dtype=float,
            )
            W1 = W0.copy()

            variables = [
                abs(literal) - 1
                for literal in clause
            ]

            for a in range(3):
                for b in range(
                    a + 1,
                    3,
                ):
                    i = variables[a]
                    j = variables[b]

                    diff = abs(
                        phases[i]
                        - phases[j]
                    )
                    diff = min(
                        diff,
                        TWOPI - diff,
                    )

                    reward = np.exp(
                        -0.5
                        * (
                            (
                                diff
                                - self._GROW_TARGET
                            )
                            / self._SIGMA
                        )
                        ** 2
                    )

                    delta = (
                        self._ETA
                        * reward
                    )

                    W1[i, j] += delta
                    W1[j, i] += delta

            observable = {
                "selected_variable": int(
                    selected_variable
                ),
                "target_phase": float(
                    target_phase
                ),
                "W_before": W0,
                "W_after": W1,
            }

            hidden = {
                "selected_clause_index": int(
                    clause_index
                ),
            }

            actions.append(
                (
                    observable,
                    hidden,
                )
            )

        return actions


def target_is_positive(
    target_phase: float,
) -> bool:
    return target_phase < np.pi


def changed_variable_set(
    W_before: np.ndarray,
    W_after: np.ndarray,
):
    dW = (
        np.asarray(
            W_after,
            dtype=float,
        )
        - np.asarray(
            W_before,
            dtype=float,
        )
    )

    upper = np.triu_indices_from(
        dW,
        1,
    )
    changed = np.where(
        np.abs(
            dW[upper]
        )
        > 1e-12
    )[0]

    variables = set()

    for index in changed:
        variables.add(
            int(
                upper[0][index]
            )
        )
        variables.add(
            int(
                upper[1][index]
            )
        )

    return tuple(
        sorted(variables)
    )


def decode_clause_candidates(
    clauses: list[list[int]],
    observable: dict[str, object],
):
    inferred_variables = (
        changed_variable_set(
            np.asarray(
                observable[
                    "W_before"
                ]
            ),
            np.asarray(
                observable[
                    "W_after"
                ]
            ),
        )
    )

    selected_variable = int(
        observable[
            "selected_variable"
        ]
    )
    target_positive = target_is_positive(
        float(
            observable[
                "target_phase"
            ]
        )
    )

    candidates = []

    for clause_index, clause in enumerate(
        clauses
    ):
        variable_set = tuple(
            sorted(
                abs(literal) - 1
                for literal in clause
            )
        )

        if (
            variable_set
            != inferred_variables
        ):
            continue

        if (
            selected_variable
            not in variable_set
        ):
            continue

        # Reuse the target policy earned by Gate 10C.
        if (
            (clause[0] > 0)
            != target_positive
        ):
            continue

        candidates.append(
            int(clause_index)
        )

    if len(candidates) == 1:
        decision = {
            "status": "IDENTIFIED",
            "clause_index": candidates[0],
            "candidates": candidates,
        }
    else:
        decision = {
            "status": "NOT_IDENTIFIABLE",
            "clause_index": None,
            "candidates": candidates,
        }

    return decision


def baseline_candidates_without_W(
    clauses: list[list[int]],
    observable: dict[str, object],
):
    """
    Attacker: infer clause from only stamped variable + target sign,
    ignoring the structural pair-growth trace.
    """
    selected_variable = int(
        observable[
            "selected_variable"
        ]
    )
    target_positive = target_is_positive(
        float(
            observable[
                "target_phase"
            ]
        )
    )

    return [
        int(index)
        for index, clause in enumerate(
            clauses
        )
        if (
            selected_variable
            in {
                abs(literal) - 1
                for literal in clause
            }
            and (
                (clause[0] > 0)
                == target_positive
            )
        )
    ]


def run_seed(
    seed: int,
    *,
    contexts: int = 500,
):
    rng = np.random.default_rng(
        seed + 5000
    )
    organism = HiddenSATTransitionGenerator(
        seed
    )

    total = 0
    true_in_candidate_set = 0
    unique = 0
    unique_correct = 0
    not_identifiable = 0
    identifiability_class_match = 0

    full_candidate_sizes = []
    baseline_candidate_sizes = []

    for _ in range(contexts):
        clauses = make_context(
            rng
        )

        signature_count = Counter(
            clause_observable_signature(
                clause
            )
            for clause in clauses
        )

        for observable, hidden in organism.action_batch(
            clauses
        ):
            total += 1

            true_index = int(
                hidden[
                    "selected_clause_index"
                ]
            )
            true_clause = clauses[
                true_index
            ]

            decision = decode_clause_candidates(
                clauses,
                observable,
            )
            candidates = decision[
                "candidates"
            ]

            baseline = baseline_candidates_without_W(
                clauses,
                observable,
            )

            full_candidate_sizes.append(
                len(candidates)
            )
            baseline_candidate_sizes.append(
                len(baseline)
            )

            true_in_candidate_set += int(
                true_index in candidates
            )

            truth_is_identifiable = (
                signature_count[
                    clause_observable_signature(
                        true_clause
                    )
                ]
                == 1
            )

            decoder_identified = (
                decision[
                    "status"
                ]
                == "IDENTIFIED"
            )

            identifiability_class_match += int(
                truth_is_identifiable
                == decoder_identified
            )

            if decoder_identified:
                unique += 1
                unique_correct += int(
                    decision[
                        "clause_index"
                    ]
                    == true_index
                )
            else:
                not_identifiable += 1

    full_candidate_sizes = np.asarray(
        full_candidate_sizes,
        dtype=float,
    )
    baseline_candidate_sizes = np.asarray(
        baseline_candidate_sizes,
        dtype=float,
    )

    return {
        "seed": seed,
        "events": total,
        "true_clause_coverage": float(
            true_in_candidate_set
            / total
        ),
        "identified_fraction": float(
            unique / total
        ),
        "identified_clause_accuracy": float(
            unique_correct
            / max(1, unique)
        ),
        "not_identifiable_fraction": float(
            not_identifiable
            / total
        ),
        "identifiability_class_accuracy": float(
            identifiability_class_match
            / total
        ),
        "mean_candidate_set_size_with_W_trace": float(
            full_candidate_sizes.mean()
        ),
        "mean_candidate_set_size_without_W_trace": float(
            baseline_candidate_sizes.mean()
        ),
        "unique_fraction_with_W_trace": float(
            np.mean(
                full_candidate_sizes
                == 1
            )
        ),
        "unique_fraction_without_W_trace": float(
            np.mean(
                baseline_candidate_sizes
                == 1
            )
        ),
    }


def summarize(receipts):
    scalar_keys = [
        "true_clause_coverage",
        "identified_fraction",
        "identified_clause_accuracy",
        "not_identifiable_fraction",
        "identifiability_class_accuracy",
        "mean_candidate_set_size_with_W_trace",
        "mean_candidate_set_size_without_W_trace",
        "unique_fraction_with_W_trace",
        "unique_fraction_without_W_trace",
    ]

    return {
        key: {
            "values": [
                float(r[key])
                for r in receipts
            ],
            "mean": float(
                np.mean(
                    [
                        r[key]
                        for r in receipts
                    ]
                )
            ),
        }
        for key in scalar_keys
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
    )
    p.add_argument(
        "--contexts",
        type=int,
        default=500,
    )
    p.add_argument(
        "--out",
        default="results/gate10d_pkas_clause_identifiability_summary.json",
    )
    args = p.parse_args()

    receipts = [
        run_seed(
            seed,
            contexts=args.contexts,
        )
        for seed in range(
            args.seeds
        )
    ]

    result = {
        "experiment": "Gate 10D P-KAS clause identifiability",
        "decoder_access": (
            "all clauses + selected variable + target phase + W before/after; "
            "selected clause index hidden"
        ),
        "reused_decoded_rule": (
            "Gate 10C: target sign follows clause[0]"
        ),
        "seeds": list(
            range(args.seeds)
        ),
        "receipts": receipts,
        "summary": summarize(receipts),
        "test_design": {
            "clauses_per_context": 18,
            "actions_per_context": 10,
            "deliberate_alias_pairs_per_context": 3,
            "alias_definition": (
                "same absolute variable set and same clause[0] sign; "
                "different non-first literal signs"
            ),
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
