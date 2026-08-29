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

## What comes next

### G1 — circuit formation during training

Do not inspect only the final network.

Save checkpoints and ask when the mathematical operation becomes visible:

```text
random network
    ->
partial approximation
    ->
distributed circuit forms
    ->
effective operator stabilizes
    ->
cleanup / redundancy changes
```

Track:

- task loss;
- distance of decoded operator from the known mathematics;
- Jacobian spectrum;
- hidden-unit causal importance;
- smallest faithful circuit;
- OOD gap between neural approximation and decoded math.

This directly tests the intuition that a circuit gradually **forms around a computation** during learning.

### G2 — blind demixing

Remove source labels from the learner. Train with independence / temporal structure, then ask whether the decoder can recover the learned demixing law without being given the true matrix.

### G3 — temporal algorithm

Use a small recurrent network whose correct solution requires state: delay, carry, parity, or a simple source with memory.

Try to decode:

```text
continuous hidden state
      ->
small causal state variable
      ->
state transition equation / finite-state machine
```

### G4 — column addition

Train a recurrent or transformer model on addition. Attempt to recover an executable description containing the functional equivalents of:

```text
sum = a + b + carry
digit = sum mod 10
carry_next = floor(sum / 10)
```

The strong test is whether the extracted program systematically generalizes to lengths beyond the neural training regime.

### G5 — P-KAS as a known mechanistic animal

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
