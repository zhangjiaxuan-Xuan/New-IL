# New-IL

New-IL is an idea archive about a supervision principle for imitation learning:
**Progress-Aware Trajectory Cloud Supervision**.

This is not a formal academic publication. It is a public idea archive for
preserving the direction, sharing it with interested readers, and potentially
finding collaborators.

## The Idea in Plain Words

In many robot imitation-learning systems, a model is trained to match the
demonstration at a fixed future time step. This can be too strict.

For continuous movement, being a little faster, slower, earlier, later, or
slightly different in path may still be perfectly fine. The robot should not be
punished heavily as long as it stays inside a reasonable path region for the
current phase.

But some things must be precise: closing the gripper, opening the gripper,
making contact, releasing an object, inserting something, or crossing from one
task phase to another.

So the core idea is simple:

**continuous motion should be flexible, but key events should be precise.**

The supervision target should reflect that difference. It should not treat every
time step and every action dimension in the same way.

## Why It Might Matter

- It separates harmless timing variation from real trajectory mistakes.
- It preserves multiple valid ways to move through a phase.
- It prevents key events from being blurred by distributional action modeling.
- It gives action chunks a more meaningful training target.
- It suggests better evaluation metrics than success rate alone, such as path
  violation, event timing error, event pose error, wrong phase crossing, and
  backward progress.

## Documents

- English: `docs/en/readme.md`
- Chinese: `docs/zh-CN/readme.md`
- OpenPI LIBERO success rollout data:
  `runs/openpi_libero_no90_0_1_2_3gpu_1s6w/DATA_README.md`

## Chinese README

See `README.zh-CN.md`.

## License

This project is released under the MIT License. See `LICENSE`.
