# From Splat geometry to neural algorithm geometry

The Splat / Geometric-Neuron lineage is relevant here in a more precise way than
"internal representations can be non-unique."

The useful old picture was:

```text
DATA WORLD
statistics of the training world
        |
        v
learned geometry / basis
        |
      freeze
        |
        v
OPERATOR WORLD
that geometry constrains the dynamics available afterward
```

In SplatField the statement was literal.

The CelebA-trained decoder chose packet positions, scales, frequencies and orientations.
Once a keyframe froze those packets, the autonomous field no longer had access to the
dataset or encoder. It saw the packet overlap geometry through its Gram matrix. That Gram
operator selected eigenmodes and decay rates.

So the remembered idea is not merely:

> geometry computes.

It is closer to:

> **experience produces geometry; frozen geometry becomes an operator constraint on later
> computation.**

GeometricNeuronPlusField found the same division in another language:

```text
slow anatomy / operator G
        ->
geometry-defined modal resonator bank
        ->
moving field
```

and showed that a one-cell anatomical change could disturb global modal coordinates.
The body was therefore not decoration around a computation: it parameterized the family
of dynamics available to the field.

## Translation to NeuralAlgorithmDecoding

A trained artificial network also has two levels that should not be collapsed:

```text
training world / data / objective
          |
          v
weights + hidden-state geometry
          |
        freeze
          |
          v
allowed state transitions / response geometry
          |
          v
abstract computation
```

The **abstract algorithm** and its **neural realization geometry** are different decoding
targets.

Gate 5 extracted the invariant algorithmic quotient:

```text
many 16-D neural states
      ->
two causal response-equivalence classes
      ->
carry transducer
      ->
base-10 program
```

Gate 6 asks what the learned neural geometry contributes after that quotient is known.

For held-out hidden states, causal classes are defined only by their complete 100-input
counterfactual response signatures. No carry labels are used. In all five trained GRUs,
those independently discovered classes are perfectly linearly separable by an LDA
microscope.

Then perturbations are norm-matched:

```text
same hidden state
      |
      +-- move across learned causal-boundary normal
      |
      +-- move same Euclidean distance orthogonal to that normal
```

Five-seed result:

```text
halfway toward boundary: causal class flips       0.0%
25% beyond boundary: causal class flips          93.9%
matched orthogonal displacement: flips            0.0%
```

So in this organism the raw hidden geometry is not arbitrary decoration. Once learning is
over, there is a direction in state space along which perturbation changes the computation,
while large matched orthogonal perturbations are almost behaviorally silent.

But there is an important limit. Boundary-crossing states are often **off manifold**:
their full response signature is about 20% Hamming distance from the nearest canonical
causal class on average. The boundary normal is therefore a causal sensitivity direction,
not a magical clean "carry coordinate."

That distinction is useful:

```text
abstract algorithm:
    two clean causal states

neural realization:
    thick learned regions / trajectories in 16-D
    + a causally important separating geometry
    + nuisance directions
    + off-manifold states that no normal trajectory visits
```

## Why keep both levels?

If NeuralAlgorithmDecoding returns only the small program, it loses information about how
the trained system physically/neuronally realizes and protects that program.

If it returns only the neural circuit geometry, it can miss the simple computation shared
across many realizations.

The desired output should eventually contain both:

1. **Algorithmic quotient** — the smallest executable causal abstraction.
2. **Realization geometry** — how the learned network embeds that abstraction, which
   perturbations are silent, which cross computational boundaries, where the representation
   is fragile, and what dynamics restore or amplify perturbations.

That is the direct inheritance from the splat work.

The dataset/world can shape the geometry.

The geometry can then constrain the computation.

And the decoder's job is to read both without pretending they are the same thing.
