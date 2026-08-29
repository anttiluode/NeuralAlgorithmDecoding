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

<details>
<summary><strong>Process note: why this slightly ridiculous repo exists</strong></summary>

This did not begin as a clean research program. It began with the recurring thought that
neural systems learn complicated transformations in distributed form, while a human may
later be able to describe the useful computation much more compactly.

So the working inversion became:

```text
let the neural machine learn however it learns
        ->
experiment on the trained thing
        ->
recover the smallest causal executable description we can earn
```

The route here has included bad ideas, over-strong interpretations, old repos unexpectedly
becoming useful again, and experiments that killed the prettier story. That mess is not a
methodological virtue; it is simply the provenance. The protection is to preserve receipts,
attackers and negative results.

A longer note on the human/AI process is in [`docs/PROCESS_NOTE.md`](docs/PROCESS_NOTE.md).

</details>

The target is:

> **A smaller executable model that agrees with the neural system on behavior and on declared interventions, generalizes where the neural approximation generalizes, and is simple enough for a human to inspect.**

There may be no unique "true algorithm" inside a neural network. The practical target is therefore the **simplest faithful causal abstraction we can earn under an explicit intervention/test family**.

## Current receipts

If you are arriving from outside the project, these are the useful checkpoints rather than the whole archaeology:

| gate | organism | what survived |
| --- | --- | --- |
| **5** | 16-D GRU doing decimal addition | counterfactual response classes -> 2-state transducer -> exact base-10 carry program; hidden-state swaps follow the decoded machine |
| **7/8** | same addition algorithm trained in different statistical worlds | the abstract program stays the same while causal robustness geometry changes; that geometry shows strong path dependence after convergence |
| **9** | addition learning through training | the pretty "geometry suddenly clicks at Aha" story fails here: causal geometry forms before sudden-looking complete-problem success |
| **10A** | labeled P-KAS core transitions | hidden phase/plasticity/pruning constants and equations recover from black-box experiments |
| **10B** | **unlabeled** mixed P-KAS transitions | transition families separate without FREE/GROW/PRUNE labels; three hidden growth targets and the same core equations are recovered |
| **10C** | P-KAS SAT intervention policy | black-box action traces recover the clause/variable/target policy and expose the historical target-sign mismatch by counterfactual tests |

The strongest current neural program-extraction result is Gate 5.  
The latest policy-recovery result is Gate 10C.  
Neither is evidence that arbitrary large neural networks can already be decompiled.

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

# Gate 3 — blind temporal demixing

This is the first gate where the network is **not given source labels**.

Two independent Gaussian AR processes have different temporal autocorrelation and are mixed by a fixed 2x2 matrix.

The neural organism is:

```text
mixture x(t)
    |
    v
2 -> tanh(16) -> 2 latent channels
    |
    v
linear decoder -> reconstruct x(t)
```

Training uses only the observations:

- reconstruct the mixture;
- keep both latent channels alive with a covariance-to-identity constraint;
- suppress off-diagonal lagged covariances at lags 1, 2, 5, and 10.

So the separation signal is essentially a differentiable neural version of the temporal structure exploited by AMUSE/SOBI.

No true source appears in the loss.

Five seeds, 2000 steps:

| quantity | mean |
| --- | ---: |
| neural latent vs true-source correlation | `0.99740` |
| reconstruction-only attacker correlation | `0.85502` |
| analytic AMUSE correlation | `0.99981` |
| decoded affine row-direction error vs exact inverse | `0.194°` |
| affine surrogate NMSE vs neural latent | `0.00380` |
| neural source correlation at 2x amplitude | `0.98929` |
| decoded-math source correlation at 2x amplitude | `0.999851` |

This one matters conceptually.

The nonlinear neural circuit learned blind source separation from temporal statistics. When we then treated that trained circuit as a black box and fitted its collective input->latent operation, the recovered matrix rows aligned with the exact demixing directions to about **0.2 degrees** on average.

So now the sequence is:

```text
mixed temporal data
      ->
nonlinear neural circuit learns separation without source labels
      ->
distributed neural transformation
      ->
read collective operator
      ->
recover the classical demixing geometry
```

The analytic AMUSE attacker is still slightly better. Good. We are not inventing a better source separator.

The point is stranger:

> **a neural circuit can learn a fuzzy implementation of a known mathematical separation operation from data, and that mathematics can be decoded back out after learning.**

The seed-0 developmental trace is also non-monotonic:

```text
step       source corr      decoded row error
0             .888              11.38°
100           .745              60.92°
200           .744              60.04°
500           .855              34.07°
1000          .998               0.44°
2000          .999               0.13°
```

So the circuit does not simply rotate smoothly toward the answer. It wanders through a worse mixed representation and then reorganizes sharply into the separated basis.

That is much closer to the "circuit forming around a computation" picture than Gate 0 was.

See `experiments/gate3_blind_temporal_demix.py` and `results/gate3_summary.json`.

---

## What comes next

# Gate 4 — continuous recurrent state -> minimal executable FSM

The next organism is an 8-dimensional GRU trained on running parity.

Training sequences are only 24 bits long.

The decoder is not given the parity state. It sees:

- input bits;
- continuous 8-D hidden states;
- the neural network's own output predictions.

A naive hidden-state clustering is **not** enough. On different seeds the same two-state computation appears as 2, 3, or 4 geometric clusters.

That turned into an important distinction:

> **geometric states are not necessarily computational states.**

So Gate 4 deliberately overclusters the hidden trajectory, builds an empirical transition graph, and then merges states that are behaviorally equivalent: same output and transitions into equivalent future states.

That is ordinary automaton minimization applied after neural-state observation.

Five seeds:

```text
raw hidden-state clusters selected:   4, 3, 2, 2, 4
minimal causal states recovered:      2, 2, 2, 2, 2
transition consistency:               1.0 on every seed
output consistency:                   1.0 on every seed
```

Every recovered machine, up to state-name permutation, is:

```text
             input 0     input 1
state 0   -> state 0   -> state 1
state 1   -> state 1   -> state 0
```

That is the parity algorithm.

Then the extracted FSM is run on **1024-step sequences**, more than 40x the neural training horizon:

```text
neural accuracy vs true parity      1.000
decoded FSM accuracy vs truth       1.000
decoded FSM fidelity vs neural      1.000
```

The stronger test is causal.

We take a running neural sequence, replace its 8-D hidden state with a representative hidden state from the **other decoded abstract state**, and continue the exact same suffix.

The neural network then follows the state-swapped FSM prediction:

```text
first-step intervention fidelity     1.000
full suffix intervention fidelity    1.000
```

across all five seeds.

So Gate 4 gets much closer to the target than a cluster plot would:

```text
8-D fuzzy recurrent dynamics
        ->
several geometric hidden clusters
        ->
behavioral equivalence / state minimization
        ->
2-state causal machine
        ->
counterfactual hidden-state intervention
        ->
same future behavior
```

The extracted object is not merely correlated with parity. Swapping the proposed state changes the neural computation exactly as the decoded state machine predicts.

See `experiments/gate4_parity_fsm.py` and `results/gate4_summary.json`.

---

## What comes next

# Gate 5 — from addition GRU to an actual arithmetic program

This is the first result in the repo that really looks like **neural computation -> causal abstraction -> executable code**.

A 16-D GRU receives two decimal digits at a time, least-significant column first, and emits the corresponding digit of their sum.

It is trained only on **8-column sequences**.

The true compact algorithm requires one hidden carry bit, but the decoder is never given carry labels.

## The first decoder failed

The obvious Gate-4 trick was tried first:

```text
hidden states
   ->
Euclidean k-means
   ->
cluster centroids
   ->
intervene on centroids
```

That failed badly even though the neural network itself was almost perfect.

The centroids were not legitimate causal states. The GRU retained nuisance geometry from recent digit inputs, so nearby / averaged hidden vectors did not mean "same computation."

That failure changes the definition of state.

## Causal response equivalence

For each actual observed hidden state `h`, Gate 5 asks the neural network a counterfactual question:

> **What would you output if each of the 100 possible digit pairs arrived next?**

So every hidden state gets a 100-input intervention signature:

```text
h
 |
 +-- (0,0) -> ?
 +-- (0,1) -> ?
 +-- ...
 +-- (9,9) -> ?
```

Hidden states are grouped by **what they do under interventions**, not by where they sit geometrically.

That immediately exposes two causal classes.

Five seeds:

```text
discovered causal state count:     2 2 2 2 2
mean signature consistency:        0.999909
mean output-table consistency:     0.999938
transition consistency:            1.000000
```

A small automatically chosen set of diagnostic interventions is then enough to classify successor states, producing a complete two-state Mealy transducer.

## Then compress the transducer into mathematics

At this point we have a 2-state × 100-input transition/output table.

The decoder is still not told "decimal carry."

It searches a tiny candidate language:

- base `B` from 2 through 16;
- either assignment of the two abstract states to carry values 0 and 1;
- candidate law:

```text
total = a + b + carry
digit = total % B
carry_next = int(total >= B)
```

Across all five seeds it discovers:

```text
B = 10
mismatches = 0
```

The abstract state names flip on different seeds, as they should. The inferred **carry semantics do not**.

So the final extracted program is:

```python
total = a + b + carry
digit = total % 10
carry = int(total >= 10)
```

## Run the extracted program beyond the neural training horizon

The GRU was trained on length 8.

Test length is 256.

```text
neural long-sequence accuracy      0.999890
decoded transducer accuracy        1.000000
decoded program/table fidelity     0.999890
```

The decoded discrete machine is perfect on the long test even where the fuzzy neural approximation makes a few errors.

## Causal intervention

Finally, replace the running GRU hidden state with an **actual observed hidden state from the opposite discovered causal class** and continue the same digit suffix.

The neural computation follows the injected carry-state prediction:

```text
first-step intervention fidelity   1.000000
full suffix fidelity               0.999951
```

That gives the complete chain:

```text
16-D recurrent neural circuit
       ->
counterfactual response signatures
       ->
2 causal equivalence classes
       ->
finite-state transducer
       ->
symbolic compression
       ->
decimal carry program
       ->
hidden-state intervention agrees
```

This is much closer to the long-term target than "neuron 7 correlates with carry."

It extracts **what the circuit computes**, and then compresses that computation into executable mathematics.

See `experiments/gate5_addition_causal_decode.py` and `results/gate5_summary.json`.

---

# Gate 6 — the algorithm has a geometry too

The splat lineage carried a more specific idea than "geometry computes":

```text
internal / data world
      ->
learned geometry
      ->
freeze geometry
      ->
geometry constrains later dynamics
```

In SplatField this was the explicit **data world -> Gram world** split: face data chose the
packet basis, then the frozen packet-overlap Gram matrix determined the autonomous field's
modes and decay rates.

Gate 6 asks whether the same distinction helps on the addition GRU.

Gate 5 already supplied two causal classes using only complete 100-input counterfactual
response signatures. No carry labels are used.

Now use those intervention-defined classes to inspect the raw 16-D hidden geometry.

Across five seeds:

```text
causal classes linearly separable in held-out hidden states       5/5
halfway toward fitted causal boundary: class flip                0.0%
25% past boundary normal: class flip                            93.9%
same Euclidean displacement orthogonal to boundary: flip         0.0%
```

So the raw neural geometry is not merely arbitrary decoration around the decoded program.
Learning has produced a state-space direction along which perturbation changes the
computation, while large matched orthogonal changes are almost computationally silent.

But the important caveat is just as useful. Crossing the linear boundary often produces
an **off-manifold fuzzy state**: the resulting full response signature is about `20.3%`
Hamming distance from the nearest canonical class on average.

Therefore:

> **The geometry constrains computation, but a geometric axis is not automatically a clean
> symbolic variable.**

The current picture has two legitimate outputs:

```text
ALGORITHMIC QUOTIENT
two causal states -> decimal carry program

REALIZATION GEOMETRY
thick 16-D neural regions
+ causally important separating direction
+ mostly silent nuisance directions
+ invalid/off-manifold regions
```

That is a much closer inheritance from the splat work than treating geometry as nuisance.

See `experiments/gate6_causal_geometry.py`,
`results/gate6_summary.json`, and
`docs/SPLAT_GEOMETRY_LINEAGE.md`.

---

# Gate 7 — same algorithm, different worlds, different geometry

Gate 6 showed that the decoded carry computation lives inside a structured neural
geometry. Gate 7 asks the stronger splat-shaped question:

> **Does the training world help choose that geometry even when the final algorithm is
> the same?**

For each seed, three GRUs start from **identical initial weights** and learn the same
decimal-addition task.

Only the training statistics differ:

```text
uniform   carry persists ~55%, flips ~45%
sticky    carry persists ~84%, flips ~16%
toggle    carry persists ~19%, flips ~81%
```

The biased worlds still draw 35% uniformly, so every digit pair retains support.

After training, all 15 networks decode to the same object:

```text
2 causal states
base = 10
program mismatches = 0
```

Uniform held-out accuracy stays above `0.997` mean in every world.

But the causal realization geometry differs.

Median intervention distance required to flip causal response class, normalized by RMS
hidden-state norm:

```text
uniform   0.2019
sticky    0.2731
toggle    0.2628
```

The sticky geometry has a **35.3% larger** causal flip radius than the paired uniform
model; toggle is **30.2% larger**. Both differences have the same sign on all 5 paired
seeds.

Even better, the ordinary hidden-space boundary predicts the actual intervention
threshold surprisingly well:

```text
corr(predicted geometric margin, actual causal flip radius)

uniform   0.888
sticky    0.807
toggle    0.831
```

And direction matters enormously. At a `0.3 x RMS-state-norm` perturbation:

```text
                 boundary-normal flip     matched random flip
uniform                 97.0%                    0.94%
sticky                  63.0%                    0.47%
toggle                  68.8%                    0.63%
```

So the result is not just "different networks use different coordinates."

Within the same architecture and task, training statistics alter the **causal robustness
geometry** while the decoded symbolic algorithm remains unchanged.

That is the cleanest current artificial analogue of the old SplatField picture:

```text
world statistics
      ->
geometry forms
      ->
freeze / run
      ->
geometry constrains which perturbations change computation
```

See `experiments/gate7_world_shapes_geometry.py` and
`results/gate7_summary.json`.

---

# Gate 8 — geometry has history

There is an immediate alternate explanation for Gate 7:

> perhaps the three worlds merely select different solutions during initial learning.

So Gate 8 starts with **one already-converged uniform-world network**, clones the exact
weights, and only then switches one clone to the sticky world and the other to the toggle
world for another 800 updates.

The base-10 program remains exact.

But the geometry barely moves:

```text
from-scratch Gate 7:
sticky vs uniform causal-radius difference    +35.3%
toggle vs uniform causal-radius difference    +30.2%

post-convergence Gate 8:
uniform -> sticky, 800 updates                 +0.4%
uniform -> toggle, 800 updates                 -2.3%
```

So in this tiny organism the same statistics that strongly affect **which geometry forms**
do not rapidly rewrite the realization after the algorithm has converged.

That suggests a developmental / hysteretic picture:

```text
world during formation
      ->
one neural realization basin is selected
      ->
algorithm stabilizes
      ->
later same-task statistics can change experience
without cheaply rebuilding the realization geometry
```

This is a negative result against the simplest "geometry just tracks current data
statistics" story, and it makes the splat/evolutionary intuition more interesting rather
than less.

See `experiments/gate8_geometry_hysteresis.py` and
`results/gate8_summary.json`.

---

# Gate 9 — artificial Aha microscope

The Aha / insight literature suggested a tempting stronger claim:

> perhaps the internal geometry itself **clicks** at the moment a solution becomes available.

Gate 9 tests that claim in the addition GRU because Gate 5 already gave us the final
causal abstraction: a two-state decimal carry machine.

The converged mechanism is used as a **progress microscope** over earlier checkpoints.

At each checkpoint:

- test ordinary length-8 digit accuracy;
- test length-128 digit accuracy;
- demand exact success on all 128 output digits;
- inject real hidden states and query all 100 possible next digit pairs;
- score each state's causal response signature against the final carry mechanism;
- measure when the two causal response classes become geometrically separable.

Five-seed median ordering:

```text
causal geometry >=90% separable        step 300
short-horizon digit accuracy >=90%     step 400
causal response error <=5%             step 450
>=10% exact carry response states      step 450
exact 128-column success >=10%         step 700
causal response error <=1%             step 800
exact 128-column success >=50%         step 875   (4/5 reached)
```

So the strict slogan **"Aha = geometry suddenly appears" fails in this organism**.

The useful geometry becomes visible first. The causal law then keeps sharpening, and only
later does complete long-horizon behavior look as though it has clicked.

That visible suddenness is partly a threshold effect: when 128 local decisions must all
be right, tiny improvements near 99-100% per-step reliability produce large changes in
complete-problem success.

The sharper current picture is:

```text
geometry / causal representation begins forming
          ->
system becomes increasingly close to the compact algorithm
          ->
external success criterion crosses a threshold
          ->
"Aha-like" behavioral transition
```

This is an artificial-network result only. It does not identify the mechanism of human
insight.

The accompanying paper review separates recent human evidence for representational
change, pre-Aha neural state, dynamic brain-state trajectories, sleep, structural
connectivity, reward and memory from the stronger geometry speculation.

See:

- `experiments/gate9_aha_geometry.py`
- `results/gate9_summary.json`
- `docs/GATE9_AHA_GEOMETRY_RESULT.md`
- `docs/AHA_SIGNAL_PAPER_REVIEW_2026-08-29.md`

---

# Gate 10A — turn the P-KAS autopsy into an automatic equation-recovery test

P-KAS was originally attractive as a system that seemed to "grow a solver."

The later audit weakened that interpretation because the problem adapters supplied much
of the solving grammar. But that makes the core a useful known organism for the decoder.

Gate 10A seals a reimplementation of the historical P-KAS core behind an experiment
interface. The decoder gets only before/after states, timestamps and permitted
interventions; it does not read the hidden constants.

It automatically recovers:

```text
FREE PHASE DYNAMICS

dphi_i/dt
  ~= K * sum_j W_ij sin(phi_j - phi_i)
     + bounded residual

true K               0.870000
decoded mean K       0.869945


PAIR GROWTH

delta W_ij = delta W_ji

delta_w
  ~= eta * exp(
       -0.5 * ((phase_error-target)/sigma)^2
     )

true eta             0.045
decoded eta          0.045

true sigma/pi        0.300
decoded sigma/pi     0.300

fit R^2              1.000


PRUNING

W <- (1-r) W
if W < threshold:
    W = 0

true r               0.005
decoded r            0.005

true threshold       0.005
decoded threshold    0.00500000000002
```

The prune threshold is found by **active binary-search intervention**, not by reading the
source constant.

So hundreds of observed transitions compress back into a few executable equations.

This is still deliberately easy:

- operation boundaries are labeled;
- the growth target relation is available as an intervention input;
- every relevant state variable is visible;
- the equation language is small.

But it converts the old manual P-KAS-Doors style decomposition into an automatic receipt.

See:

- `experiments/gate10a_pkas_blackbox_equations.py`
- `results/gate10a_pkas_blackbox_summary.json`
- `docs/GATE10A_PKAS_BLACKBOX_RESULT.md`

---

# Gate 10B — discover the P-KAS regimes before fitting equations

Gate 10A was still told which operation happened.

Gate 10B receives a shuffled stream containing only:

```text
phase_before
phase_after
W_before
W_after
dt
```

No FREE / GROW / PRUNE labels.

No changed pair.

No growth target relation.

The transition support itself reveals three regimes:

```text
phase changes, W fixed       -> phase-only flow
one reciprocal W pair moves  -> local pair plasticity
many W entries move          -> global pruning
```

Post-hoc truth scoring:

```text
regime recovery accuracy     1.000 on all 5 seeds
```

The phase-only regime again yields:

```text
true K                 0.870000
decoded mean K         0.869688

true noise width       0.030000
decoded mean           0.029957

mean R^2               0.99810
```

The pair-growth regime is more interesting because the decoder is **not told the hidden
target relation**.

From the strongest updates it discovers three preferred phase differences on every seed:

```text
~0.1
~pi/2
~2*pi/3
```

and recovers one shared law:

```text
delta W_ij = delta W_ji

delta_w
  ~= 0.045 *
     exp(
       -0.5 *
       ((phase_difference-target)/(0.3*pi))^2
     )
```

with fit `R^2 = 1.0` in this deterministic calibration.

The global-weight regime yields:

```text
W <- 0.995 W

if W < ~0.005:
    W = 0
```

with decoded mean threshold `0.004999856`.

So the progression is now:

```text
Gate 10A
named transition -> fit equation

Gate 10B
unlabeled transition
    -> identify transition family
    -> infer hidden relation modes
    -> fit equation
```

The large caveat remains: P-KAS makes this unusually easy because its three operations
touch visibly different parts/supports of the state. A realistic neural system will have
overlapping mechanisms.

See:

- `experiments/gate10b_pkas_unlabeled_regimes.py`
- `results/gate10b_pkas_unlabeled_summary.json`
- `docs/GATE10B_PKAS_UNLABELED_RESULT.md`

---

# Gate 10C — recover SAT policy and a historical bug from behavior

Gate 10C moves above low-level equations.

The decoder sees:

```text
unsatisfied-clause context
which clause was acted on
which variable was intervened on
which target phase was applied
```

It does not use the SAT adapter source.

First, the acted-on clauses behave like a uniform size-10 subset without replacement.
Inside each selected 3-literal clause, the chosen variable position is approximately
uniform:

```text
position 0   .3345
position 1   .3320
position 2   .3335
```

Then candidate target-sign programs are compared:

| candidate | mean accuracy |
| --- | ---: |
| **sign of clause[0]** | **1.0000** |
| majority sign | .7497 |
| sign of selected literal | .6687 |
| sign of clause[2] | .5011 |
| sign of clause[1] | .4999 |

That reproduces the odd historical behavior we had previously found by reading source:

```text
selected variable
    <- random literal in clause

target sign
    <- FIRST literal in clause
```

The causal test is stronger.

For trials where a non-first literal is selected:

```text
flip selected non-first literal sign
keep clause[0] fixed
    ->
target unchanged          100%

flip clause[0] sign
keep selected literal fixed
    ->
target flips              100%
```

So the decoder has crossed from:

```text
fit dynamics
```

into:

```text
recover an executable decision rule
and expose a behavioral bug
```

without source inspection.

This still has a generous interface because the acted-on clause identity is observed.

See:

- `experiments/gate10c_pkas_sat_policy.py`
- `results/gate10c_pkas_sat_policy_summary.json`
- `docs/GATE10C_PKAS_SAT_POLICY_RESULT.md`

---

## What comes next

### G10D — remove observed clause identity

Give the decoder only:

```text
all unsatisfied clauses
state before
state after
observed stamped node
observed target phase
downstream W changes
```

and ask which clause must have generated the transition.

If several clauses are equally compatible, the decoder should return:

```text
NOT IDENTIFIABLE
```

rather than inventing one.

That would join the P-KAS policy work to the old TransientWaveCompiler lesson:

> **an explanation is only earned to the level that the available experiment can
> distinguish it from alternatives.**

A second high-value branch remains genuine delayed-generalization / modular-addition
grokking: decode the final mechanism, then ask whether a causal "distance to
generalization" can be measured before the visible test-accuracy transition.

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
