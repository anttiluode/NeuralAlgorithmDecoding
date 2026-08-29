# Gate 10D — the decoder must sometimes refuse to explain

Date: 2026-08-29

Status: **executed, five seeds**

Gate 10C recovered the historical SAT adapter's target-sign mismatch from behavior.

But it was still told which clause had been acted on.

Gate 10D removes that information.

The new question is:

> **Can the downstream transition identify the hidden constraint that caused it?**

And, more importantly:

> **Will the decoder admit when the answer is not identifiable?**

This is where the old TransientWaveCompiler lesson becomes part of program decoding rather
than a side note.

---

# What the decoder sees

For every SAT action:

```text
all currently unsatisfied clauses
selected / stamped variable
target phase
W before
W after
```

It does **not** see:

```text
selected clause index
```

The hidden adapter then performs the old within-clause growth operation on all three
variable pairs belonging to the selected 3-literal clause.

So the transition leaves a structural trace.

---

# Compose previously decoded knowledge

Gate 10D does not rediscover everything from zero.

It reuses one policy already earned in Gate 10C:

```text
target sign = sign of clause[0]
```

This is intentional.

A useful decompiler eventually has to compose discoveries:

```text
low-level operator
        +
decoded policy fragment
        +
new observation
        ->
higher-level causal explanation
```

---

# Infer the hidden clause from W support

From

```text
delta W = W_after - W_before
```

the decoder extracts every changed undirected pair.

For a three-variable clause this yields the triangle:

```text
(a,b)
(a,c)
(b,c)
```

and therefore the hidden absolute variable set:

```text
{a,b,c}
```

Candidate clauses must then satisfy:

1. same absolute variable set;
2. contain the observed selected variable;
3. have a `clause[0]` sign compatible with the observed target phase.

If exactly one clause survives:

```text
IDENTIFIED
```

If several survive:

```text
NOT_IDENTIFIABLE
candidate set = [...]
```

No arbitrary tie-breaking is allowed.

---

# Deliberately construct an identifiability trap

Every 18-clause context contains three alias pairs.

Each alias pair has:

```text
same three absolute variables
same sign of clause[0]
different non-first literal signs
```

Example:

```text
[ +3, +7, -11 ]
[ +3, -7, +11 ]
```

Under the declared observations, these two clauses produce the same:

- variable-set triangle in `delta W`;
- possible selected variables;
- target sign, because `clause[0]` is positive in both.

There is no information in this experiment that can tell them apart.

That is not decoder failure.

Choosing one would be decoder fabrication.

---

# Five-seed result

Each seed:

```text
500 contexts
18 clauses / context
10 actions / context
5000 hidden-clause events
```

## Coverage

The true hidden clause is always inside the returned compatible set:

```text
true-clause coverage       1.000
```

## Unique cases

About two thirds of actions are genuinely identifiable:

```text
identified fraction       0.66644
```

When the decoder says a unique clause is identifiable:

```text
identified-clause accuracy    1.000
```

## Alias cases

The deliberately ambiguous aliases occupy the remaining third:

```text
NOT_IDENTIFIABLE fraction     0.33356
```

The decoder's identified/not-identifiable classification matches the known observational
equivalence class:

```text
identifiability-class accuracy    1.000
```

So it does not "solve" the alias cases.

It correctly refuses them.

---

# Does the W trace actually buy anything?

Attacker:

> ignore the structural growth trace and use only the observed selected variable plus
> target phase.

That leaves, on average:

```text
2.576 candidate clauses
```

and only:

```text
22.2%
```

of events are uniquely identified.

Using the downstream W triangle:

```text
mean candidate set       1.334
unique events            66.6%
```

So the structural transition approximately triples the fraction of uniquely decodable
hidden actions.

That is a clean demonstration of why **how the system changes afterward** can be more
informative than the intervention output alone.

---

# The important result is the refusal

Most interpretability demos are rewarded for always returning an explanation.

Gate 10D is designed so that this is wrong.

For an alias pair:

```text
clause A
clause B
      |
      | same declared observable signature
      v

no experiment in the current family distinguishes them
```

The correct output is:

```text
NOT_IDENTIFIABLE
```

That is exactly the epistemic rule TransientWaveCompiler learned from gauge-equivalent
internal realizations.

The new project now uses it operationally:

> **decode only to the quotient supported by the observations.**

---

# Progression through P-KAS

```text
Gate 10A
named low-level operation
        ->
recover equation


Gate 10B
unlabeled low-level transitions
        ->
discover operator regimes
        ->
recover equations


Gate 10C
observed SAT decisions
        ->
recover executable policy
        ->
discover historical sign bug


Gate 10D
hidden clause identity
        ->
use downstream structural trace
        ->
identify when possible
        ->
REFUSE when observationally aliased
```

This is a considerably healthier endpoint than trying to force every hidden state into one
"true explanation."

---

# What remains easy

The aliases are engineered cleanly.

The W edits are noise-free and completely visible.

A real neural system will have:

- overlapping mechanisms;
- partially observed state;
- measurement noise;
- approximate rather than exact equivalence;
- many candidate abstractions at different scales.

So the future version needs the quantitative identifiability machinery already ported from
TWC:

```text
candidate response direction
        ->
project away existing / nuisance span
        ->
noise-whiten
        ->
measure residual novelty
```

Then `NOT_IDENTIFIABLE` becomes graded rather than exact.

---

# Files

- `experiments/gate10d_pkas_clause_identifiability.py`
- `results/gate10d_pkas_clause_identifiability_summary.json`

Related:

- `nad/identifiability.py`
- `docs/GATE10B_PKAS_UNLABELED_RESULT.md`
- `docs/GATE10C_PKAS_SAT_POLICY_RESULT.md`
