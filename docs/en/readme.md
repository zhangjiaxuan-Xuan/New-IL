# Progress-Aware Trajectory Cloud Supervision

This is an English archive version of the original Chinese idea note.

The proposed idea is called:

**Progress-Aware Trajectory Cloud Supervision for Event-Constrained Imitation
Learning**

or **PA-TCS**.

## Core Idea

The central claim is that action supervision in imitation learning should not
always force a model to match one fixed trajectory point at one fixed time
index.

For continuous motion stages, the learner should be allowed to be slightly
faster, slower, earlier, later, or spatially different as long as the predicted
motion remains inside a reasonable progress-aware trajectory cloud tube.

However, key events such as gripper open, gripper close, contact onset, release,
or subgoal boundary should remain precise. They should not be blurred into a
cloud or averaged away.

## Motivation

Existing distributional policies already model multimodal actions. Diffusion
Policy, BeT, and IBC show that actions may have multiple valid modes. But the
new point here is that distributional supervision should not apply uniformly to
every time step and every action dimension.

Action chunks have heterogeneous temporal semantics:

- continuous movement points can be progress-elastic
- key events must remain event-constrained and precise
- the whole sequence must preserve progress consistency

## Method Sketch

Given context $c_t=(o_t,l,h_t)$, a policy predicts an action chunk:

$$
\hat{\tau}_t=
\{(\hat{x}_{t+k},\hat{g}_{t+k})\}_{k=0}^{H-1}
$$

Instead of only minimizing fixed-time behavior cloning error, PA-TCS supervises
continuous actions against a progress-aware trajectory cloud tube while applying
stricter constraints at key transitions.

## Intended Use

This note is not a finished paper. It is an idea archive for preserving a
possible research direction and inviting discussion or collaboration.

The Chinese version in `docs/zh-CN/` contains the fuller original note.
