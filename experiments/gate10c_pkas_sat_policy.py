#!/usr/bin/env python3
"""
Gate 10C: recover a P-KAS SAT intervention policy — including its sign bug —
from black-box behavior rather than source inspection.

Historical SAT adapter behavior:
    choose up to 10 unsatisfied clauses without replacement
    choose one literal from each selected clause
    intervene on that literal's VARIABLE
    but set the target sign from clause[0], not from the selected literal

The decoder is not allowed to read that rule.  It only sees:
    unsatisfied-clause context
    which clause was acted on
    which variable was intervened on
    which target phase was applied

It then:
1. tests whether clause selection is consistent with uniform sampling without replacement;
2. identifies how the selected variable relates to literal position;
3. enumerates compact target-sign policies;
4. actively flips selected-literal sign vs first-literal sign to distinguish rival rules.

This is a policy-recovery calibration, not a claim that the whole SAT solver was learned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class HiddenSATAdapter:
    """Reproduces the historical adapter policy behind a query interface."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def actions(
        self,
        unsatisfied_clauses: list[list[int]],
        max_actions: int = 10,
    ) -> list[dict[str, object]]:
        count = min(max_actions, len(unsatisfied_clauses))
        chosen = self.rng.choice(
            len(unsatisfied_clauses),
            size=count,
            replace=False,
        )

        out = []
        for clause_index in chosen:
            clause = unsatisfied_clauses[int(clause_index)]

            selected_literal = int(
                self.rng.choice(clause)
            )
            selected_variable = abs(selected_literal) - 1

            # Historical bug / policy mismatch:
            target_phase = (
                0.1
                if clause[0] > 0
                else np.pi + 0.1
            )

            out.append(
                {
                    "clause_index": int(clause_index),
                    "selected_variable": int(selected_variable),
                    "target_phase": float(target_phase),
                }
            )

        return out


def random_clause(
    rng: np.random.Generator,
    num_vars: int = 60,
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


def literal_position(
    clause: list[int],
    selected_variable: int,
) -> int:
    for position, literal in enumerate(clause):
        if abs(literal) - 1 == selected_variable:
            return position
    raise ValueError("selected variable is not in clause")


def target_is_positive(target_phase: float) -> bool:
    return target_phase < np.pi


def clause_selection_probe(
    policy: HiddenSATAdapter,
    rng: np.random.Generator,
    *,
    trials: int = 4000,
    clause_count: int = 18,
):
    clauses = [
        random_clause(rng)
        for _ in range(clause_count)
    ]

    inclusion = np.zeros(
        clause_count,
        dtype=int,
    )
    duplicate_count = 0
    action_sizes = []

    for _ in range(trials):
        actions = policy.actions(clauses)
        ids = [
            int(a["clause_index"])
            for a in actions
        ]

        action_sizes.append(len(ids))
        duplicate_count += (
            len(ids) - len(set(ids))
        )

        for clause_index in ids:
            inclusion[clause_index] += 1

    inclusion_rate = (
        inclusion / trials
    )
    expected = 10.0 / clause_count

    return {
        "clause_count": clause_count,
        "actions_per_trial": 10,
        "expected_uniform_inclusion_rate": expected,
        "mean_inclusion_rate": float(
            inclusion_rate.mean()
        ),
        "max_absolute_inclusion_deviation": float(
            np.max(
                np.abs(
                    inclusion_rate - expected
                )
            )
        ),
        "duplicate_action_rate": float(
            duplicate_count
            / max(
                1,
                trials * 10,
            )
        ),
        "all_trials_have_10_unique_clauses": bool(
            np.all(
                np.asarray(action_sizes)
                == 10
            )
            and duplicate_count == 0
        ),
    }


def collect_policy_records(
    policy: HiddenSATAdapter,
    rng: np.random.Generator,
    *,
    trials: int = 3000,
    clause_count: int = 18,
):
    records = []

    for _ in range(trials):
        clauses = [
            random_clause(rng)
            for _ in range(clause_count)
        ]

        for action in policy.actions(clauses):
            clause = clauses[
                int(action["clause_index"])
            ]
            selected_variable = int(
                action["selected_variable"]
            )
            position = literal_position(
                clause,
                selected_variable,
            )

            records.append(
                {
                    "clause": clause,
                    "selected_variable": selected_variable,
                    "selected_position": position,
                    "target_phase": float(
                        action["target_phase"]
                    ),
                }
            )

    return records


def variable_selection_probe(
    records: list[dict[str, object]],
):
    positions = np.asarray(
        [
            int(r["selected_position"])
            for r in records
        ],
        dtype=int,
    )

    counts = np.bincount(
        positions,
        minlength=3,
    )
    frequency = (
        counts / counts.sum()
    )

    return {
        "literal_position_frequency": frequency.tolist(),
        "max_deviation_from_uniform_one_third": float(
            np.max(
                np.abs(
                    frequency
                    - 1.0 / 3.0
                )
            )
        ),
    }


def candidate_target_predictions(
    clause: list[int],
    selected_position: int,
):
    return {
        "clause[0] sign": clause[0] > 0,
        "clause[1] sign": clause[1] > 0,
        "clause[2] sign": clause[2] > 0,
        "selected literal sign": clause[selected_position] > 0,
        "majority literal sign": sum(
            literal > 0
            for literal in clause
        ) >= 2,
        "constant positive": True,
        "constant negative": False,
    }


def target_policy_search(
    records: list[dict[str, object]],
):
    names = list(
        candidate_target_predictions(
            [1, 2, 3],
            0,
        ).keys()
    )

    correct = {
        name: 0
        for name in names
    }

    for record in records:
        clause = [
            int(x)
            for x in record["clause"]
        ]
        position = int(
            record["selected_position"]
        )
        truth = target_is_positive(
            float(
                record["target_phase"]
            )
        )

        predictions = candidate_target_predictions(
            clause,
            position,
        )

        for name, prediction in predictions.items():
            correct[name] += int(
                prediction == truth
            )

    accuracy = {
        name: float(
            correct[name]
            / len(records)
        )
        for name in names
    }

    # Minimum-description tie break is only used after accuracy.
    # The exact character count is not a scientific quantity; it simply makes
    # the selection deterministic if several policies fit equally well.
    description_length = {
        "clause[0] sign": len("clause[0] > 0"),
        "clause[1] sign": len("clause[1] > 0"),
        "clause[2] sign": len("clause[2] > 0"),
        "selected literal sign": len("clause[selected_position] > 0"),
        "majority literal sign": len("sum(signs)>1"),
        "constant positive": len("True"),
        "constant negative": len("False"),
    }

    winner = min(
        names,
        key=lambda name: (
            -accuracy[name],
            description_length[name],
            name,
        ),
    )

    return {
        "candidate_accuracy": accuracy,
        "selected_policy": winner,
        "selected_policy_accuracy": accuracy[winner],
    }


def query_until_selected_position(
    policy: HiddenSATAdapter,
    clause: list[int],
    position: int,
):
    attempts = 0

    while True:
        attempts += 1
        action = policy.actions(
            [clause],
            max_actions=1,
        )[0]

        selected_position = literal_position(
            clause,
            int(
                action["selected_variable"]
            ),
        )

        if selected_position == position:
            return (
                float(
                    action["target_phase"]
                ),
                attempts,
            )


def active_counterfactual_probe(
    policy: HiddenSATAdapter,
    rng: np.random.Generator,
    *,
    trials: int = 300,
):
    selected_flip_unchanged = []
    first_flip_changes = []
    attempts = []

    for _ in range(trials):
        clause = random_clause(rng)
        selected_position = int(
            rng.choice([1, 2])
        )

        baseline, a0 = query_until_selected_position(
            policy,
            clause,
            selected_position,
        )

        selected_flip = clause.copy()
        selected_flip[
            selected_position
        ] *= -1

        changed_selected, a1 = query_until_selected_position(
            policy,
            selected_flip,
            selected_position,
        )

        first_flip = clause.copy()
        first_flip[0] *= -1

        changed_first, a2 = query_until_selected_position(
            policy,
            first_flip,
            selected_position,
        )

        selected_flip_unchanged.append(
            abs(
                baseline
                - changed_selected
            )
            < 1e-12
        )
        first_flip_changes.append(
            abs(
                baseline
                - changed_first
            )
            > 1.0
        )
        attempts.append(
            a0 + a1 + a2
        )

    return {
        "flip_selected_nonfirst_literal_target_unchanged": float(
            np.mean(
                selected_flip_unchanged
            )
        ),
        "flip_first_literal_target_changes": float(
            np.mean(
                first_flip_changes
            )
        ),
        "mean_blackbox_queries_for_three_conditioned_actions": float(
            np.mean(attempts)
        ),
    }


def run_seed(seed: int):
    policy = HiddenSATAdapter(seed)
    rng = np.random.default_rng(
        seed + 10_000
    )

    selection = clause_selection_probe(
        policy,
        rng,
    )

    records = collect_policy_records(
        policy,
        rng,
    )

    variable = variable_selection_probe(
        records
    )

    target = target_policy_search(
        records
    )

    counterfactual = active_counterfactual_probe(
        policy,
        rng,
    )

    return {
        "seed": seed,
        "clause_selection": selection,
        "variable_selection": variable,
        "target_policy": target,
        "active_counterfactual": counterfactual,
    }


def summarize(receipts):
    candidate_names = list(
        receipts[0][
            "target_policy"
        ][
            "candidate_accuracy"
        ].keys()
    )

    candidate_mean = {
        name: float(
            np.mean(
                [
                    r[
                        "target_policy"
                    ][
                        "candidate_accuracy"
                    ][name]
                    for r in receipts
                ]
            )
        )
        for name in candidate_names
    }

    return {
        "selected_policy": [
            r[
                "target_policy"
            ][
                "selected_policy"
            ]
            for r in receipts
        ],
        "candidate_accuracy_mean": candidate_mean,
        "mean_selected_literal_position_frequency": np.mean(
            np.asarray(
                [
                    r[
                        "variable_selection"
                    ][
                        "literal_position_frequency"
                    ]
                    for r in receipts
                ]
            ),
            axis=0,
        ).tolist(),
        "mean_max_clause_inclusion_deviation": float(
            np.mean(
                [
                    r[
                        "clause_selection"
                    ][
                        "max_absolute_inclusion_deviation"
                    ]
                    for r in receipts
                ]
            )
        ),
        "duplicate_clause_selection_rate": float(
            np.mean(
                [
                    r[
                        "clause_selection"
                    ][
                        "duplicate_action_rate"
                    ]
                    for r in receipts
                ]
            )
        ),
        "counterfactual_selected_literal_flip_unchanged": float(
            np.mean(
                [
                    r[
                        "active_counterfactual"
                    ][
                        "flip_selected_nonfirst_literal_target_unchanged"
                    ]
                    for r in receipts
                ]
            )
        ),
        "counterfactual_first_literal_flip_changes": float(
            np.mean(
                [
                    r[
                        "active_counterfactual"
                    ][
                        "flip_first_literal_target_changes"
                    ]
                    for r in receipts
                ]
            )
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
        default="results/gate10c_pkas_sat_policy_summary.json",
    )
    args = p.parse_args()

    receipts = [
        run_seed(seed)
        for seed in range(
            args.seeds
        )
    ]

    result = {
        "experiment": "Gate 10C P-KAS SAT policy recovery",
        "decoder_access": (
            "unsatisfied clauses + observed acted-on clause index, selected variable, "
            "and target phase; hidden adapter source is not used by decoder"
        ),
        "seeds": list(
            range(args.seeds)
        ),
        "receipts": receipts,
        "summary": summarize(receipts),
        "ground_truth_for_posthoc_scoring_only": {
            "clause_selection": "uniform subset without replacement, up to 10",
            "variable_selection": "uniform literal within selected clause",
            "target_rule": "positive target iff clause[0] > 0",
            "historical_mismatch": (
                "selected variable can come from a different literal than the literal "
                "whose sign sets the target"
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
