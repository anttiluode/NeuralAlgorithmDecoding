# Gate 10A — P-KAS black-box equation recovery

Date: 2026-08-29

Status: **executed calibration**

## Why return to P-KAS?

P-KAS was originally interesting because it looked like a system whose distributed phase
dynamics and changing connectivity might "learn a solver."

The later audit weakened that claim. The problem adapters supplied substantial solver
grammar.

But that disappointment creates a useful test organism for NeuralAlgorithmDecoding.

We know the P-KAS core contains a small set of dynamical rules. Instead of reading the
source and manually naming them, can a decoder recover those rules from controlled
observations and interventions?

That is exactly the larger project in miniature.

## Experimental boundary

The organism is a sealed reimplementation of the historical P-KAS core.

The decoder can observe:

```text
state before
state after
W before / after
phase before / after
elapsed dt
which intervention was invoked
control argument such as target_relation
```

The decoder **cannot read the hidden constants**.

This is still generous. The operation boundaries are labeled.

So Gate 10A is not "arbitrary black box -> equations."

It is:

> **Can the equation-recovery machinery work at all when the hidden answer is known?**

## Part 1 — free phase dynamics

We record unforced phase evolution while the coupling matrix is visible.

The decoder constructs the candidate feature

```text
c_i = sum_j W_ij sin(phi_j - phi_i)
```

and fits the smallest scalar model:

```text
dphi_i/dt ~= K c_i
```

Five seeds:

```text
true K             0.870000
decoded mean K     0.869945
std                 0.000191

mean R^2            ~0.99906
```

The remaining residual is approximately bounded zero-mean drive.

From its variance the decoder estimates the width:

```text
true noise width       0.030000
decoded mean width     0.030157
```

So it recovers:

```text
dphi_i/dt
   ~=
K * sum_j W_ij sin(phi_j - phi_i)
+ bounded residual
```

without reading the configured (K).

## Part 2 — growth law

For pairwise growth interventions the decoder sees:

- current phases;
- selected pair (i,j);
- target relation;
- W before and after.

First it notices:

```text
delta W_ij == delta W_ji
```

on every measured update.

That is already a structural discovery:

> **the plastic pair update is reciprocal / symmetric.**

Then it plots:

```text
log(delta W)
```

against:

```text
(wrapped_phase_difference - target_relation)^2
```

The relationship is perfectly linear on non-saturated observations.

That implies:

```text
delta_w
 ~= eta * exp(
      -0.5 * (phase_error / sigma)^2
    )
```

Recovered across all five seeds:

```text
true eta                 0.045
decoded eta              0.045

true sigma/pi            0.300
decoded sigma/pi         0.300

log-Gaussian fit R^2     1.000
```

This is a small but literal example of:

```text
observed neural-ish plasticity
        ->
transform the observations
        ->
recognize mathematical family
        ->
recover executable equation
```

## Part 3 — pruning by active interrogation

A passive trace can show weights shrinking, but locating a hard survival threshold is an
experiment-design problem.

So the decoder actively chooses edge values and repeatedly asks:

> if I set this one edge to (w), does it survive one prune operation?

First a safely large edge reveals multiplicative decay:

```text
true prune rate       0.005
decoded               0.005
```

Then a 35-query binary search finds the survival boundary.

After accounting for the multiplicative decay, the inferred hard threshold is:

```text
true threshold        0.005000000000
decoded               0.00500000000002
```

This is perhaps the nicest part of the calibration because it already uses the future
decoder loop:

```text
candidate mechanism
      ->
choose discriminating intervention
      ->
observe response
      ->
shrink uncertainty
```

## What we have automatically reconstructed

The decoder now emits approximately:

```text
FREE DYNAMICS

dphi_i/dt =
    0.869945 * sum_j W_ij sin(phi_j - phi_i)
    + bounded zero-mean residual


PLASTICITY

delta W_ij = delta W_ji

delta_w =
    0.045 *
    exp(
       -0.5 *
       ((wrapped_phase_difference-target) / (0.3*pi))^2
    )


PRUNING

W <- 0.995 * W

if W < 0.005:
    W = 0
```

Those are recognizably the hidden P-KAS core laws.

## Why this is not yet the dream

Because the decoder was handed too much structure.

It knows which transitions are:

```text
FREE
GROW
PRUNE
```

and it is handed the target relation for a growth intervention.

The difficult version receives:

```text
one mixed trajectory
```

and must itself discover:

- that multiple regimes/operators exist;
- where transitions change regime;
- which controls matter;
- what latent variables are sufficient;
- what mathematical family describes each regime.

And even that would only decode the P-KAS **core**.

It would not yet explain the higher-level SAT behavior because the historical adapter
contains explicit operations such as selecting unsatisfied clauses and choosing variables
to stamp.

So the hierarchy is:

```text
Gate 10A
labeled low-level transitions -> equations                PASS

Gate 10B
unlabeled mixed transitions -> discover regimes -> equations

Gate 10C
core + SAT adapter traces -> separate "physics" from policy

Gate 10D
ask what compact executable solver description remains
after the supplied adapter logic is accounted for
```

Gate 10A earns permission to attempt 10B.

It does not earn a claim that P-KAS learned SAT.

## Connection to the larger project

This calibration contains almost every ingredient in toy form:

- **operator estimation** — recover Kuramoto coupling;
- **invariant discovery** — symmetric plasticity;
- **symbolic fitting** — Gaussian phase-error growth;
- **active experiment selection** — binary-search the prune threshold;
- **description compression** — hundreds of transitions become a few lines of equations.

The next job is to remove the labels one by one.
