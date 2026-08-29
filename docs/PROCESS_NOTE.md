# Process note — why this repo exists

This file is not part of the method.

It is here because the method did not arrive as a method.

The project started from a very ordinary fascination: neural systems seem to solve
complicated transformations without carrying around the clean symbolic equations that a
human mathematician would normally write down.

A biological nervous system hears mixtures, sees motion, learns a word from repeated
context, predicts what comes next, and gradually turns experience into useful behavior.
Artificial neural networks do a related kind of thing in a much simpler engineered
setting: give them data, an objective, enough parameters, and they often construct a
distributed internal computation that works before anyone has a concise description of
what that computation is.

That leads to the inversion behind this repository:

```text
do not begin by asking:
"how should we write the algorithm?"

instead ask:
"what algorithm did this trained neural machine end up implementing?"
```

The dream version is extravagant:

```text
large fuzzy neural computation
        ->
experiments on the trained machine
        ->
causal states / operators / invariants
        ->
small executable mathematics
```

If the compact mathematics exists, it may be cheaper to run, easier to verify, easier to
transfer, and easier to understand than the neural approximation that discovered it.

But the process here is deliberately less grand than the dream.

We use tiny organisms whose answers are knowable. We let an experiment fail. We preserve
the failure. We try the boring attacker. We separate a pretty representation from a causal
one. We keep asking whether the thing we are calling an "algorithm" survives an
intervention.

This is why the repository contains odd steps such as:

- a nonlinear demixer that collapses back into a matrix;
- a GRU whose geometric clusters are *not* its computational states;
- a carry algorithm recovered only after grouping neural states by what they do under
  counterfactual queries;
- different training worlds producing different realization geometry while preserving the
  same executable algorithm;
- an "Aha" experiment that killed the nicest version of the geometry-click story.

That last one is representative of the process.

The exciting sentence was:

> perhaps the geometry clicks.

The experiment said:

> not like that, at least here.

Useful hidden geometry appeared before the sudden-looking behavioral transition.

That correction is worth more than keeping the prettier story.

## A note about AI-assisted work

This repository is heavily AI-assisted.

That is not hidden and it is not itself evidence for any result.

The practical workflow is closer to:

```text
human notices / asks / remembers / proposes
       ->
AI reads code and papers, proposes a precise test
       ->
code is built
       ->
the result attacks the idea
       ->
human reaction changes the question
       ->
repeat
```

Sometimes the human contribution is an awkward physical picture rather than formal
mathematics. Sometimes the AI contributes the language needed to turn that picture into a
test. Sometimes an older repository contains exactly the mathematical warning needed by a
new one. Sometimes both of us wander into fog.

The only protection against that is the ledger:

```text
QUESTION
PAYOFF
FALSIFIER
INVARIANT
RESULT
ATTACKER
WHAT DIED
WHAT SURVIVED
```

The repo should be judged by the experiments and receipts, not by whether the route to
them looked respectable.

## Why keep this note?

Because a polished README can accidentally make exploratory work look inevitable in
retrospect.

It was not.

The useful thread emerged through many failed architectures, overclaims that later had to
be narrowed, strange analogies, forgotten old experiments, and repeated attempts to
translate an intuition into something falsifiable.

That mess is part of the provenance.

The scientific claim, if any eventually survives, still has to stand without it.
