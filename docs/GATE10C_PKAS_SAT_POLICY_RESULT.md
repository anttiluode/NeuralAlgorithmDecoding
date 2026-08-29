# Gate 10C — recover the SAT policy and its historical sign bug

Date: 2026-08-29

Status: **executed, five seeds**

This is the first P-KAS gate that is no longer mainly about equations of motion.

Gate 10A recovered named low-level operators.

Gate 10B removed the operator labels and recovered the same P-KAS core laws from an
unlabeled transition stream.

Gate 10C moves one level upward:

> **Can black-box behavior reveal the policy that chooses what the core should do?**

The historical SAT adapter gives us an unusually sharp test because manual source
inspection had already exposed a real mismatch between the variable being acted on and
the literal whose sign chooses the target phase.

The decoder is not allowed to use that source.

---

# Historical behavior being hidden

The SAT wrapper repeatedly acts on currently unsatisfied clauses.

The black-box policy produces observable actions:

```text
selected clause
selected variable
target phase
```

The decoder sees those actions and the clause context.

It does **not** receive:

- the random choice implementation;
- the literal position used to select the variable;
- the rule used to choose the target sign.

The hidden adapter source contains the equivalent of:

```python
selected_literal = random.choice(clause)
selected_variable = abs(selected_literal) - 1

target = positive_phase if clause[0] > 0 else negative_phase
```

The suspicious part is obvious only after reading the code:

```text
selected variable
    comes from one random literal

target sign
    comes from clause[0]
```

Those can be different literals.

Gate 10C asks whether behavior alone exposes that.

---

# Part 1 — which clause gets acted on?

Use a fixed set of 18 candidate unsatisfied clauses and repeatedly query the policy.

Every call returns 10 acted-on clauses.

Observed:

```text
duplicate selected clauses inside one action batch    0
all trials contain 10 unique clauses                  yes
expected uniform inclusion probability                10/18 = .5556
mean inclusion probability                            .5556
mean max per-clause deviation across seeds            .0182
```

This is consistent with:

> choose a size-10 subset approximately uniformly without replacement.

This is not a deep discovery; it is a behavioral characterization of the first policy
stage.

---

# Part 2 — which variable inside the clause?

Across large randomly generated 3-literal clause contexts, map the observed selected
variable back to its literal position.

Mean five-seed frequencies:

```text
literal position 0     .33452
literal position 1     .33201
literal position 2     .33347
```

So the selected variable is consistent with:

> choose one of the clause's three literals approximately uniformly, then use that
> literal's variable index.

Again, this is the sensible part of the historical policy.

---

# Part 3 — what determines the target sign?

Now enumerate compact rival explanations for the observed positive/negative target phase.

Candidates:

```text
sign of clause[0]
sign of clause[1]
sign of clause[2]
sign of the selected literal
majority sign of the clause
constant positive
constant negative
```

Mean held-out behavior across five seeds:

| candidate target rule | accuracy |
| --- | ---: |
| **sign of clause[0]** | **1.0000** |
| majority sign | .7497 |
| sign of selected literal | .6687 |
| sign of clause[2] | .5011 |
| sign of clause[1] | .4999 |
| constant positive | .5005 |
| constant negative | .4995 |

Every seed independently selects:

```text
target sign = sign of clause[0]
```

The more natural explanation

```text
target sign = sign of selected literal
```

is substantially worse.

That is already enough to rediscover the mismatch observationally.

But correlation is not the standard we want.

---

# Part 4 — causal bug test

Construct a clause and wait until the black box happens to choose a **non-first**
literal, position 1 or 2.

Then make two counterfactual clauses.

## Counterfactual A — flip the selected literal

Keep `clause[0]` unchanged.

Flip only the sign of the non-first literal whose variable is being selected.

Query until the same literal position is selected again.

Prediction:

```text
selected-literal rule:
    target should flip

first-literal rule:
    target should stay the same
```

Observed across all five seeds:

```text
target unchanged     1.000
```

## Counterfactual B — flip clause[0]

Keep the selected non-first literal unchanged.

Flip only the first literal's sign.

Prediction:

```text
selected-literal rule:
    target should stay the same

first-literal rule:
    target should flip
```

Observed:

```text
target flips         1.000
```

So the causal receipt is exact:

> **the target phase depends on the first literal, not on the literal whose variable is
> being acted on.**

This is the historical bug/mismatch we previously found by reading source.

Gate 10C recovers it from behavior.

---

# Compact decoded policy

The decoder can now describe the observable SAT intervention policy as:

```text
given unsatisfied clauses:

1. choose up to 10 distinct clauses
   approximately uniformly without replacement

2. for each selected clause:
      choose one literal approximately uniformly
      selected_variable = variable(selected_literal)

3. choose target sign from CLAUSE[0], not selected_literal

      if clause[0] is positive:
          target = positive phase
      else:
          target = negative phase
```

That third line is the bug receipt.

---

# Why this matters more than another fitted constant

The progression is now:

```text
Gate 10A
known operation -> recover equation

Gate 10B
unknown operation family -> recover regimes + equations

Gate 10C
observed decisions -> recover policy + falsify rival policy
```

This is much closer to the long-term target.

The decoder did not merely say:

> "variable 7 correlates with negative phase."

It returned an executable rule that predicts actions under counterfactual clause edits.

And the rule exposes behavior that a human would reasonably call a bug.

---

# Important limits

## 1. The acted-on clause index is observed

This is still a generous interface.

A harder decoder should receive only the full unsatisfied-clause set plus downstream
phase/weight changes and infer **which clause must have been acted on**.

## 2. We are decoding the adapter, not proving P-KAS learned SAT

Quite the opposite.

The recovered policy reinforces the earlier audit:

> substantial solver logic is supplied explicitly by the adapter.

The core dynamics are not autonomously inventing the clause-selection procedure.

## 3. This is a tiny candidate-language search

The decoder enumerates a handful of plausible sign rules.

It is not open-ended program synthesis.

But the candidate rules are attacked with interventions rather than selected only by
correlation.

---

# Gate 10D

Remove the observed clause identity.

Give the decoder:

```text
all currently unsatisfied clauses
phase / W state before
phase / W state after
observed stamped node + target
```

Ask it to infer:

1. which constraint was acted on;
2. which variable-selection policy generated the node;
3. which sign-selection policy generated the target;
4. which downstream W edits belong to that same selected clause.

Then compress all of that into one executable policy receipt.

The stronger kill condition is:

> if several clauses are behaviorally indistinguishable from the transition, return
> **NOT IDENTIFIABLE** rather than inventing one.

That would bring the TransientWaveCompiler identifiability lesson directly into program
recovery.

---

# Files

- `experiments/gate10c_pkas_sat_policy.py`
- `results/gate10c_pkas_sat_policy_summary.json`

Related:

- `docs/GATE10A_PKAS_BLACKBOX_RESULT.md`
- `docs/GATE10B_PKAS_UNLABELED_RESULT.md`
