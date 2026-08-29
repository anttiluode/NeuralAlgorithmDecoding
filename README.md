# NeuralAlgorithmDecoding

> **Can a fuzzy learned neural computation be decoded back into a compact causal algorithm?**

This repository is a small falsification-driven lab for **black-box neural computation -> human-readable mathematics / executable programs**.

The motivating picture is simple:

```text
data / signals
      |
      v
neural network learns however it wants
      |
      v
distributed nonlinear circuit
      |
      v
measure + perturb + identify coordinates
      |
      v
candidate equations / state machines / small programs
      |
      v
actively search for counterexamples
      |
      v
small executable causal surrogate
```

The target is **not** an attention map, a neuron label, or a story about what a feature "means."

The target is:

> **A smaller executable model that agrees with the neural system on behavior and on declared interventions, generalizes where the neural approximation generalizes, and is simple enough for a human to inspect.**

There may be no unique "true algorithm" inside a neural network. The practical target is therefore the **simplest faithful causal abstraction we can earn under an explicit intervention/test family**.

---

## Why this repo exists

A neural network can learn a computation without ever representing the mathematics in the form a mathematician would write it.

For example, a network trained to undo a mixture may approximate

```text
x = A s
s = A^-1 x
```

through many nonlinear hidden units. The inverse matrix can be present **functionally** without appearing as one explicit matrix anywhere in the learned circuit.

The central question is whether we can go the other direction:

```text
learned nonlinear approximation
        ->
experiments on the learned circuit
        ->
compact mathematics
```

If that works on controlled model organisms, the same strategy can later be attacked on RNNs, transformers, learned optimizers, neural algorithmic reasoners, and eventually richer scientific surrogate models.

---

## The decoding pipeline

The repo will keep these stages separate:

1. **Behavioral contract**  
   Define exactly which behavior is being decoded and what counts as fidelity.

2. **Coordinate discovery**  
   Raw neurons may be a terrible basis. Search for stable / cause-like coordinates before interpreting individual units.

3. **Causal circuit localization**  
   Use ablation, patching, state swapping, and counterfactual interventions to find components that are necessary/sufficient for the behavior.

4. **Operator estimation**  
   Estimate Jacobians, local linear operators, state-transition maps, low-rank structure, symmetries, and invariants.

5. **Symbolic / program fitting**  
   Fit sparse equations, finite-state machines, small programs, or other concise executable descriptions.

6. **Disagreement search**  
   Generate inputs/interventions where competing candidate explanations predict different outcomes.

7. **Kill weak explanations**  
   Preserve failed candidates. A pretty equation that does not survive interventions is not the algorithm.

8. **Executable receipt**  
   Emit a small model plus quantitative behavioral, OOD, intervention, and description-length scores.

---

# Gate 0 — decode a neural demixer back into its matrix

This is deliberately easy. It is a calibration organism, not a novelty claim.

Two independent non-Gaussian sources are mixed by a fixed matrix:

```text
s -> A -> x
```

A small nonlinear network is trained **only as a neural function approximator**:

```text
2 inputs -> tanh(16) -> 2 outputs
```

The true computation is known:

```text
s = A^-1 x
```

After training, the decoder is asked whether the fuzzy 16-unit nonlinear circuit can be compressed back into the effective 2x2 operation.

The current Gate-0 decoder uses two deliberately boring probes:

- least-squares effective affine operator from network I/O;
- mean input-output Jacobian computed from the learned circuit.

It also performs hidden-unit ablations and records the circuit size needed to preserve 1% neural-output NMSE.

Run:

```bash
python experiments/gate0_linear_demixer.py --steps 4000 --seed 0
```

## First 5-seed receipt

Local CPU run, seeds 0-4, 4000 AdamW steps:

| quantity | mean |
| --- | ---: |
| neural network NMSE | `1.32e-4` |
| extracted matrix relative Frobenius error vs exact `A^-1` | `1.21e-3` |
| extracted affine surrogate NMSE vs network | `1.22e-4` |
| mean Jacobian relative error vs exact `A^-1` | `4.39e-3` |
| neural network NMSE at 2x source amplitude | `4.01e-3` |
| extracted mathematical surrogate NMSE at 2x amplitude | `5.25e-6` |

The fun first observation is therefore:

> **The small neural network learned a fuzzy nonlinear approximation of a linear inverse, but a matrix decoded from its in-distribution behavior recovered the underlying operation closely enough to extrapolate dramatically better than the neural approximation itself at 2x amplitude.**

In this calibration, the extracted affine math has about **760x lower NMSE** than the neural network on the 2x-amplitude test.

That does **not** mean we solved mechanistic interpretability. Gate 0 is intentionally generous:

- the true problem class is linear;
- the decoder is handed the candidate family "affine operator";
- the task is supervised source recovery;
- least squares can exploit the network's I/O behavior directly;
- most hidden units are still required to reproduce the neural implementation at 1% fidelity.

The point is to establish the exact phenomenon we care about:

```text
fuzzy learned neural implementation
          ->
simpler mathematical description
          ->
better systematic extrapolation
```

Now we make the decoder progressively less informed.

See `results/gate0_summary.json`.

---

# Gate 1 — film the birth of the mathematics

`experiments/gate1_circuit_formation.py` keeps the same model organism but inspects checkpoints through training.

At every checkpoint it measures:

- task NMSE;
- error of the decoded affine operator against the known inverse;
- input-to-output Jacobians and how much they vary with input;
- each hidden unit's rank-1 contribution to the mean Jacobian;
- how many units are needed to reconstruct that Jacobian;
- neural versus decoded-math behavior at 2x amplitude.

Five-seed receipt:

| event | median training step |
| --- | ---: |
| network reaches <1% ID NMSE | `200` |
| decoded operator reaches <5% relative error | `200` |
| decoded operator reaches <1% relative error | `300` |
| decoded math beats neural 2x-OOD error by >10x | `200` |
| Jacobian becomes <5% input-variable | `750` |

At the end of training:

```text
network 2x-OOD NMSE             4.08e-3
decoded affine 2x-OOD NMSE      6.27e-6
mean advantage                  ~1072x

effective hidden-unit participation   10.26 / 16
units for 10% Jacobian reconstruction 11.2 / 16
units for 5% Jacobian reconstruction  12.8 / 16
```

So the first developmental picture is not:

```text
one special neuron becomes "the inverse matrix"
```

It is closer to:

```text
many nonlinear units co-adapt
        ->
a simple operator becomes visible in their collective effect
        ->
the collective implementation remains distributed
```

The compact mathematics becomes recoverable early, while the neural implementation never collapses to a tiny obvious subcircuit.

That is exactly the distinction this repo cares about.

See `results/gate1_summary.json`.

---

# Gate 2 — stop telling the decoder "fit a matrix"

Gate 0 and Gate 1 handed the decoder the answer family: affine maps.

`experiments/gate2_family_selection.py` weakens that assumption. The decoder is given a small candidate language of polynomial programs of degree 0, 1, 2, or 3 and sees only black-box `x -> neural output` queries.

It chooses the **smallest** candidate that reproduces held-out neural behavior within 1% NMSE.

Two positive-control organisms are trained:

```text
L: genuinely linear 2-D law
Q: genuinely quadratic 2-D law
```

The hidden true equations are used only after model-family selection for scoring.

Five seeds:

```text
linear organism:
  chosen degree     1 1 1 1 1

quadratic organism:
  chosen degree     2 2 2 2 2
```

OOD against the hidden generating law:

| organism | neural 2x-OOD NMSE | decoded program 2x-OOD NMSE | mean neural/decoded error ratio |
| --- | ---: | ---: | ---: |
| linear | `2.73e-3` | `8.01e-6` | `806x` |
| quadratic | `4.58e-2` | `1.30e-3` | `255x` |

This is still a calibration. The correct answer is already representable in the supplied polynomial language.

But we have moved one step:

```text
"fit this matrix"
      ->
"which compact family best explains this neural computation?"
```

See `results/gate2_summary.json`.

---

## A useful failure immediately after Gate 2

I also tried the obvious version of the future **active falsifier**:

> fit linear and quadratic rival explanations, then query the neural black box where they disagree most.

It failed.

On three development seeds with a 100x cumulative-error rejection criterion:

```text
random querying:
  rejected the wrong linear model in 1 query on all 3 seeds

naive maximum-disagreement querying:
  seed 0: 3 queries
  seed 1: not rejected within 100
  seed 2: not rejected within 100
```

A bootstrap uncertainty-aware variant improved this to `1, 1, >100`, but was still not robust.

The failure is informative: **maximum disagreement is not automatically maximum information.** It can seek points where one or both candidate models are poorly estimated rather than points that cleanly discriminate mechanisms.

So the active-experiment stage needs a proper reliability / expected-information criterion rather than the slogan "ask where they disagree."

That route is paused rather than tuned until it wins against random querying.

---

## What comes next

### G3 — blind demixing / coordinate discovery

Remove source labels from the learner. Train with independence or temporal structure, then ask whether the decoder can first discover a useful coordinate system and only then recover the learned demixing law.

This is where the Tuesday / ICA / IVA lineage should finally earn a direct role.

### G4 — temporal algorithm

Use a small recurrent network whose correct solution requires state: delay, carry, parity, or a simple source with memory.

Try to decode:

```text
continuous hidden state
      ->
small causal state variable
      ->
state transition equation / finite-state machine
```

### G5 — column addition

Train a recurrent or transformer model on addition. Attempt to recover an executable description containing the functional equivalents of:

```text
sum = a + b + carry
digit = sum mod 10
carry_next = floor(sum / 10)
```

The strong test is whether the extracted program systematically generalizes to lengths beyond the neural training regime.

### G6 — P-KAS as a known mechanistic animal

P-KAS already has a manual decomposition in `pkas_doors.py`: symmetric learned coupling, explicit twist, clause pull, pruning, and falsifiable ablations.

Hide the source code from the decoder and ask whether trajectories + interventions are enough to rediscover that decomposition.

---

# What our older repos contribute

This repo is not starting from zero.

- **Tuesday** — coordinate systems matter. ICA/IVA supplied a disciplined way to ask whether the same underlying cause can be recovered across different observed bases.
- **WhatIsI** — correlations are weak evidence; counterfactual state swapping can test whether an internal variable is actually used as the causal coordinate for downstream computation.
- **Sunday** — estimate the effective operator first, then attack exotic interpretations with linear/reservoir/symmetry nulls.
- **Twensday** — characterize reachable matrix families, rank, spectra, recurrence, and representation limits instead of inferring intelligence from a pretty matrix.
- **P-KAS Doors** — manually decompose a complicated solver into operators, derive mathematical consequences, then test those consequences by ablation.
- **Concrete / active-selection threads** — spend experiments where competing explanations actually disagree.

The new synthesis is:

> **coordinate discovery + causal intervention + operator identification + symbolic compression + active falsification.**

---

# Field map

This is a live research area. We should borrow aggressively and claim novelty only after attackers.

Closest existing lines:

- **Mechanistic interpretability / automated circuit discovery** — ACDC and later circuit-finding methods automate localization of causal subnetworks.
- **Causal abstraction / interchange interventions** — formalizes when a high-level causal model is a faithful abstraction of a neural model.
- **Sparse feature circuits / transcoders / attribution graphs** — replace polysemantic neuron-level stories with sparse feature-level causal graphs.
- **Circuit tracing** — attribution graphs expose chains of internal feature influence and permit intervention.
- **Sparse-circuit training** — weight-sparse transformers can be trained so their learned circuits are substantially easier to understand.
- **Deep Distilling** — trains a constrained neural representation that can be losslessly condensed into compact executable code.
- **MIPS / program synthesis from neural algorithms** — converts RNN behavior into discrete latent state and synthesizes executable programs.
- **Symbolic regression / SINDy / neural equation discovery** — discovers compact governing equations from observed or latent dynamical trajectories.
- **Formal mechanistic interpretability** — recent work adds verification-style guarantees for discovered circuits.
- **Neural algorithmic reasoning interpretability** — recent work studies circuit formation and reuse in networks trained to emulate classical algorithms.

Useful starting references:

- Conmy et al. (2023), *Towards Automated Circuit Discovery for Mechanistic Interpretability*  
  https://arxiv.org/abs/2304.14997
- Nanda et al. (2023), *Progress measures for grokking via mechanistic interpretability*  
  https://arxiv.org/abs/2301.05217
- Geiger et al. (2025), *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*  
  https://www.jmlr.org/papers/v26/23-0058.html
- Marks et al. (2025), *Sparse Feature Circuits*  
  https://arxiv.org/abs/2403.19647
- Blazek, Venkatesh & Lin (2024), *Automated discovery of algorithms from data* / Deep Distilling  
  https://doi.org/10.1038/s43588-024-00593-9
- Liu et al. (2024), *Opening the AI Black Box: Distilling Machine-Learned Algorithms into Code* / MIPS  
  https://doi.org/10.3390/e26121046
- OpenAI (2025), *Weight-sparse transformers have interpretable circuits*  
  https://arxiv.org/abs/2511.13653
- Anthropic (2025), open-source circuit tracing / attribution graphs  
  https://www.anthropic.com/research/open-source-circuit-tracing
- Hadad, Katz & Bassan (2026), *Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees*  
  https://arxiv.org/abs/2602.16823
- He et al. (2026), *MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning*  
  https://arxiv.org/abs/2602.21442

---

## Ground rules

1. **Do not confuse a correlated feature with a causal variable.**
2. **Do not assume raw neuron axes are the right coordinates.**
3. **A candidate explanation must make intervention predictions.**
4. **Always compare against a boring behavioral distillation baseline.**
5. **Prefer executable equations/programs to prose explanations.**
6. **Measure description length as well as fidelity.**
7. **Preserve failures.**
8. **Do not silently change the question when a new paper arrives.**

The long-term question is extravagant.

The first organisms should be tiny enough that we can know exactly when we are fooling ourselves.
