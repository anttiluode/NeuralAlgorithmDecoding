# Gate 10B — remove the FREE / GROW / PRUNE labels

Date: 2026-08-29

Status: **executed, five seeds**

Gate 10A was intentionally kind to the decoder.

It was told:

```text
this transition is FREE
this transition is GROW
this transition is PRUNE
```

and a growth intervention exposed its target relation.

Gate 10B removes both pieces of information.

The decoder receives one shuffled transition stream containing only:

```text
phase_before
phase_after
W_before
W_after
dt
```

No operation label.

No selected growth pair.

No target phase relation.

The hidden truth is kept separately and used only after decoding to score the result.

---

## The organism

The black-box stream contains three historical P-KAS core operations:

```text
free phase evolution       700 samples / seed
pair growth                900 samples / seed
global prune               500 samples / seed
```

The growth organism secretly draws one of three target relations:

```text
0.1
pi/2
2*pi/3
```

These are not arbitrary values invented for the decoder. They correspond to relation
values that appeared in the old P-KAS adapters:

- near-same-phase growth in the community branch;
- `pi/2` in the SAT adapter;
- `2*pi/3` for three-color graph separation.

The decoder does not receive those target values.

---

# Step 1 — discover what kinds of transitions exist

The first regime atlas is deliberately primitive.

For each black-box transition ask only:

```text
did phase move?
how many undirected W edges moved?
```

That produces three distinct empirical signatures:

```text
A
phase changes
W does not
        ->
PHASE-ONLY REGIME

B
phase does not change
exactly one undirected W pair changes
        ->
PAIR-WEIGHT REGIME

C
phase does not change
many W edges change
        ->
GLOBAL-WEIGHT REGIME
```

Only after this partition is frozen do we compare with the hidden operation labels.

Result:

```text
5 seeds
2100 transitions / seed

regime classification accuracy
1.000
1.000
1.000
1.000
1.000
```

So the FREE / GROW / PRUNE labels from Gate 10A were unnecessary for this particular
organism.

### Important limitation

This is an easy regime-discovery problem.

The operators alter visibly different state blocks and supports.

A realistic neural system may have several mechanisms changing the same variables at the
same time.

So the correct claim is:

> **P-KAS's core operator families are identifiable from their transition support without
> being named in advance.**

Not:

> "we solved arbitrary unsupervised mechanism discovery."

---

# Step 2 — decode the phase-only operator

On the phase-only cluster the decoder again constructs

```text
c_i = sum_j W_ij sin(phi_j - phi_i)
```

from the observed state and asks whether velocity is approximately one scalar multiple
of that quantity.

Across five seeds:

```text
true coupling gain K      0.870000
decoded mean K            0.869688
std                       0.000107

true residual width       0.030000
decoded mean width        0.029957

mean model R^2            0.99810
```

So the unlabeled stream still yields:

```text
dphi_i/dt
   ~= 0.86969 *
      sum_j W_ij sin(phi_j - phi_i)
      + bounded zero-mean residual
```

---

# Step 3 — decode the pair-weight regime without knowing the target relation

This is the new part.

Every transition in the pair-weight regime changes one reciprocal pair:

```text
W_ij
W_ji
```

The decoder finds that pair by looking at the support of `delta W`.

It then records:

```text
wrapped phase difference d
amount of growth delta_w
```

but still does **not** know which hidden target relation generated the event.

## Find preferred relations from the strongest growth events

If growth is largest when phase difference is near some preferred relation, then the
largest updates should accumulate around those relations.

The decoder therefore takes the strongest 5% of pair-growth events and asks how many
one-dimensional modes best describe their phase differences.

Candidate mode counts:

```text
2
3
4
5
```

are compared by silhouette score.

All five seeds select:

```text
3 modes
```

The inferred centers are, to numerical precision:

```text
0.100000...
1.570796...  ~= pi/2
2.094395...  ~= 2*pi/3
```

The decoder has therefore recovered three hidden relation preferences without receiving
the target relation input.

## Recover the shared functional law

Using the inferred centers as initialization, jointly fit:

```text
eta
sigma
target centers
```

to the strongest growth observations.

The recovered law is:

```text
delta_w
 ~= eta * exp(
      -0.5 *
      ((d - target) / sigma)^2
    )
```

with:

```text
true eta                 0.045
decoded eta              0.045

true sigma/pi            0.300
decoded sigma/pi         0.300

fit R^2                  1.000
```

The pair update is also exactly reciprocal:

```text
delta W_ij = delta W_ji
```

on all five seeds.

This is a useful step beyond Gate 10A:

```text
unlabeled local plasticity events
        ->
find changed relation
        ->
find preferred phase-difference modes
        ->
recover one shared mathematical update law
```

---

# Step 4 — decode global pruning passively

The global-weight cluster changes many weights together.

Surviving edges reveal a common multiplicative ratio:

```text
W_after / W_before ~= 0.995
```

therefore:

```text
prune_rate ~= 0.005
```

For the hard threshold, the decoder reconstructs the would-be post-decay value:

```text
candidate = 0.995 * W_before
```

and compares:

```text
largest candidate that disappeared
smallest candidate that survived
```

The threshold lies between them.

Five-seed result:

```text
true threshold        0.005000000
decoded mean          0.004999856
std                   9.95e-8
```

So Gate 10B no longer needs Gate 10A's active threshold binary search either. The mixed
stream itself contains enough near-threshold examples.

---

# What the decoder returns

Starting from an unlabeled stream, it reconstructs approximately:

```text
REGIME 1 — PHASE FLOW

dphi_i/dt =
    0.86969 *
    sum_j W_ij sin(phi_j - phi_i)
    + bounded zero-mean residual


REGIME 2 — LOCAL PLASTICITY

delta W_ij = delta W_ji

target in {
    0.1,
    pi/2,
    2*pi/3
}

delta_w =
    0.045 *
    exp(
      -0.5 *
      ((phase_difference-target)/(0.3*pi))^2
    )


REGIME 3 — GLOBAL PRUNING

W <- 0.995 * W

if W < ~0.005:
    W = 0
```

The labels "phase flow", "local plasticity" and "global pruning" are descriptions added
after the transition signatures are identified.

The equations themselves come from the observed transitions.

---

# What Gate 10B actually earned

Gate 10A:

```text
named operation
      ->
fit equation
```

Gate 10B:

```text
unlabeled transition
      ->
discover transition family from support
      ->
infer hidden relation modes
      ->
fit equation
```

That is progress.

But the remaining gap is still large.

The decoder is handed the state decomposition:

```text
phi
W
```

and the P-KAS operations are almost embarrassingly separable.

The interesting next boundary is not another decimal place of parameter recovery.

It is **policy**.

---

# Gate 10C question

The historical SAT wrapper contains logic above the P-KAS core:

```text
evaluate current Boolean assignment
      ->
find unsatisfied clauses
      ->
choose clauses to fix
      ->
choose a variable
      ->
choose a target Boolean phase
      ->
invoke core stamp / grow
```

Manual source inspection found a particularly useful bug:

```python
var = abs(np.random.choice(clause)) - 1
target = 0.1 if clause[0] > 0 else np.pi + 0.1
```

The chosen variable can come from one literal while the target sign is taken from
`clause[0]`.

That gives Gate 10C a beautiful falsifiable target.

Hide the source.

Expose only:

```text
SAT instance
current phases / assignment
unsatisfied clauses
which node was intervened on
which target phase was applied
subsequent trajectory
```

Then ask the decoder:

> **What policy predicts intervention choice?**

If it correctly discovers that:

```text
selected variable  <- random literal in selected clause
target sign        <- FIRST literal of selected clause
```

rather than the more sensible:

```text
target sign <- sign of selected literal
```

then the project has crossed an important line:

> from recovering equations of motion to recovering **algorithmic policy and a bug**
> from behavior.

That is Gate 10C.

---

# Files

- `experiments/gate10b_pkas_unlabeled_regimes.py`
- `results/gate10b_pkas_unlabeled_summary.json`

The prior calibration remains:

- `docs/GATE10A_PKAS_BLACKBOX_RESULT.md`

The next result should not be called a success unless the decoder finds the policy mismatch
without reading the SAT source.
