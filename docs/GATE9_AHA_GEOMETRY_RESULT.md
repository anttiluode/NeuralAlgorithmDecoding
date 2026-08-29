# Gate 9 — Artificial Aha microscope: does the geometry actually click?

Date: 2026-08-29

Status: **executed 5-seed model-organism experiment**

## Question

The motivating brain-sized idea was:

```text
experience / world
      ->
internal geometry forms
      ->
geometry constrains trajectories
      ->
a formerly inaccessible solution becomes reachable
      ->
AHA
```

That is attractive, but it hides two very different possibilities.

### Strong "geometry click" hypothesis

The internal computational geometry itself changes abruptly at approximately the same
moment that behavior suddenly becomes successful.

### Progress-before-click hypothesis

The internal causal representation organizes gradually, while a downstream behavioral
criterion crosses a threshold and therefore *looks* abrupt.

Gate 9 was built to tell these apart in one artificial organism whose final algorithm we
already know.

The result favors the second picture.

---

# Why reuse the Gate-5 addition GRU?

Gate 5 gave us a useful model organism:

```text
two input digits
      ->
16-D GRU hidden state
      ->
one output digit
```

The model is trained on ordinary column addition, least-significant column first.

At convergence, Gate 5 independently decoded the network into:

```python
total = a + b + carry
digit = total % 10
carry = int(total >= 10)
```

and causal hidden-state swaps followed the extracted two-state machine.

That means Gate 9 does **not** need to guess what computation to measure. We already have
an earned causal abstraction.

This is methodologically similar to the "progress measure" idea in Nanda et al. (2023):
first reverse-engineer the final mechanism, then use that mechanism as a microscope for
earlier checkpoints.

Important boundary:

> Gate 9 is a **formation experiment**, not a fresh blind-decoding result.

The true carry response table is used only to score progress toward the Gate-5 mechanism.

---

# The visible "Aha-like" readout

The GRU is trained on sequences of only 8 columns.

At every 50 training steps we also test it on **128-column additions**.

We record two different quantities:

1. **per-digit accuracy** over all 128 columns;
2. **exact-sequence success**: all 128 output digits must be correct.

The second is deliberately harsh.

A network can be "almost right" at each local step yet fail almost every complete
128-column problem.

If independent per-digit correctness were (p), then exact 128-step success would be
approximately:

```text
p = 0.9900  -> p^128 ~= 0.276
p = 0.9950  -> p^128 ~= 0.526
p = 0.9990  -> p^128 ~= 0.880
p = 0.9995  -> p^128 ~= 0.938
```

So a smooth change in local reliability can appear as a much sharper change in complete
problem-solving ability.

That makes exact-sequence success a useful artificial analogue of a visible "click" while
letting us inspect everything underneath it.

It is **not** a claim that human insight is 128 independent Bernoulli decisions.

---

# Internal progress measures

At every checkpoint we take actual hidden states from the running GRU.

For each hidden state (h), we perform 100 interventions:

```text
inject h
   |
   +-- next input (0,0) -> neural output ?
   +-- next input (0,1) -> neural output ?
   +-- ...
   +-- next input (9,9) -> neural output ?
```

This gives the state a complete 100-input **causal response signature**.

We then measure three things.

## 1. Causal signature error

Compare the neural response signature with the nearest exact Gate-5 carry-state response
table.

```text
0.00 = complete causal response matches carry=0 or carry=1 exactly
1.00 = complete mismatch
```

This asks whether the network has actually acquired the local computational law.

## 2. Exact-signature fraction

What fraction of sampled hidden states already behave **exactly** like one of the two
carry states under all 100 next-input interventions?

This is stricter than task accuracy.

## 3. Causal geometry separability

Label each sampled neural state by the nearest carry-response class and ask whether those
two intervention-defined classes are geometrically separable in the raw 16-D hidden
space.

This is intentionally only a microscope. It does not claim that a linear separator is
the algorithm.

---

# Frozen protocol

```text
architecture             GRU hidden size 16
task                     decimal column addition
training horizon         8 columns
long test horizon        128 columns
training steps           1200
checkpoint interval      50
seeds                    0..4
optimizer                 AdamW
learning rate            3e-3
weight decay             1e-5
training batch           384
```

Code:

`experiments/gate9_aha_geometry.py`

Receipt:

`results/gate9_summary.json`

---

# Result

Median first checkpoint across the five seeds:

| event | median step | seeds reaching it by 1200 |
| --- | ---: | ---: |
| causal geometry separability >= 90% | **300** | 5/5 |
| short-horizon digit accuracy >= 90% | **400** | 5/5 |
| causal signature error <= 5% | **450** | 5/5 |
| >=10% hidden states have exact 100-query carry signature | **450** | 5/5 |
| exact 128-column success >=10% | **700** | 5/5 |
| causal signature error <=1% | **800** | 5/5 |
| >=50% hidden states have exact carry signature | **700** | 4/5 |
| exact 128-column success >=50% | **875** | 4/5 |
| exact 128-column success >=90% | **1000** | 2/5 |

The ordering is the important part:

```text
causal geometry forms
        ↓
local behavior becomes good
        ↓
causal response law sharpens
        ↓
only later does exact long-horizon behavior "click"
```

The visible solution transition is late relative to the emergence of a useful internal
organization.

---

# One seed in detail

Seed 0:

| step | short digit acc | 128-digit acc | exact 128-col success | causal signature error | exact carry signatures | geometry separability |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .106 | .102 | .000 | .889 | .000 | .664 |
| 200 | .503 | .494 | .000 | .474 | .000 | .805 |
| 300 | .806 | .823 | .000 | .177 | .000 | **.953** |
| 400 | .933 | .928 | .000 | .061 | .102 | .969 |
| 500 | .979 | .968 | .008 | .026 | .289 | .914 |
| 600 | .982 | .982 | .094 | .016 | .453 | .945 |
| 700 | .994 | .993 | **.422** | .006 | .625 | .969 |
| 800 | .997 | .997 | **.617** | .003 | .742 | .984 |
| 1000 | .998 | .999 | **.836** | .001 | .898 | .984 |
| 1200 | 1.000 | .99945 | **.938** | .00016 | .984 | .969 |

The hidden causal states are already geometrically organized at step 300, while the
network still solves **zero** of the sampled 128-column problems perfectly.

That is difficult to reconcile with the strong version of:

> "the geometry suddenly clicks at the behavioral Aha."

At least in this organism, useful geometry precedes the visible click.

---

# Progress association

Pooling checkpoints after step 100:

```text
Spearman rho(
    causal-signature error,
    exact long-sequence success
) = -0.965

Spearman rho(
    causal-geometry separability,
    exact long-sequence success
) = +0.566
```

Do **not** read these as causal effect estimates.

Training time drives all three measures, so the pooled correlations mainly say they
co-develop.

The ordering and interventions are more informative than the raw correlations.

---

# What was killed?

## Killed: a literal one-moment geometry click in this organism

We did not see an abrupt internal event at which an unstructured hidden space suddenly
became the carry geometry.

The representation becomes increasingly organized well before exact long-horizon
problem solving appears.

## Survives: "pre-Aha geometry" / readiness

A stronger causal state geometry exists while the system is still behaviorally unreliable
on complete problems.

The state can be **near the right computation before the external criterion declares
success**.

## Survives: thresholded visibility

Exact complete-problem performance is a nonlinear readout. Near-perfect local computation
can turn into a steep change in complete success.

So an apparently discrete behavioral event need not imply a discrete underlying learning
event.

---

# Relation to human Aha work

This experiment is deliberately not neuroscience.

Still, it gives us a useful separation of hypotheses when reading the human literature.

Human studies now report, among other things:

- representational change associated with insight;
- measurable neural differences immediately before successful Aha reports;
- pre-stimulus states that predict later Aha strength;
- different large-scale brain-state dynamics during insight solving;
- subsequent-memory effects;
- trait-level structural-connectivity associations.

Those observations are compatible with both:

```text
A) sudden internal restructuring
```

and:

```text
B) gradual / latent preparation
   + fast state transition
   + thresholded conscious report
   + later reinforcement
```

Gate 9 is evidence only about an artificial GRU, but it demonstrates concretely why
behavioral suddenness alone cannot distinguish A from B.

That is also strongly consonant with mechanistic work on grokking, where a dramatic
generalization transition can sit on top of much smoother circuit formation.

See `docs/AHA_SIGNAL_PAPER_REVIEW_2026-08-29.md`.

---

# Connection to the splat geometry lineage

The old SplatField picture was:

```text
data world
   ->
geometry / basis is learned
   ->
geometry is frozen
   ->
operator world evolves inside those constraints
```

Gates 6-8 showed an artificial-neural analogue:

```text
training statistics
   ->
different neural realization geometry
   ->
same abstract carry algorithm
   ->
different causal robustness
```

Gate 9 adds time:

```text
geometry begins to organize
   ->
causal response law sharpens
   ->
robust externally visible computation arrives later
```

So the current working picture is not "Aha equals geometry."

It is:

> **geometry may create a landscape of possible computation; the system can become close
> to a useful computational basin before behavior makes the transition obvious.**

That is a much more testable statement.

---

# Limits

1. **No human claim.** This is a GRU trained on decimal addition.
2. **No subjective Aha.** "Aha-like" refers only to a sudden-looking external success
   criterion.
3. **Post-hoc mechanism target.** Gate 5 had already decoded carry, so Gate 9 uses the
   true carry response table as a progress microscope.
4. **Geometry microscope is simple.** The 90% separability metric is based on a simple
   two-class mean direction, not a complete manifold analysis.
5. **Exact-sequence success is deliberately nonlinear.** Some of the apparent click is
   mathematically guaranteed by compounding.
6. **Five seeds is a model-organism receipt, not a population-level theorem.**

---

# Next decisive experiment

The stronger analogue is a system with a genuine delayed-generalization phase where
training behavior saturates long before test generalization — modular-addition grokking is
the obvious model organism.

The target would be:

```text
checkpoint
   ->
behavior
   ->
causal / representational geometry
   ->
smallest decoded algorithm
   ->
intervention stability
```

and then ask:

> Is there a measurable "distance to insight" that predicts the later generalization
> transition before ordinary behavior does?

Gate 9 says the instrumentation is worth building.

It also says we should expect the answer to be subtler than a single neural flash.
