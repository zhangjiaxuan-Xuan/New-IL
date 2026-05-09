# New-IL

New-IL is an idea archive about **Progress-Aware Trajectory Cloud Supervision**
for event-constrained imitation learning.

> This is not a rigorous academic publication. It is an exploratory archive for
> preservation, discussion, inspiration, and possible collaboration.

## The Core Intuition

Many imitation-learning systems supervise an action chunk as if every future
action must match one fixed demonstration timestep. This is too rigid for robot
manipulation.

Continuous motion can be flexible:

> move a little faster, slower, earlier, later, or along a slightly different
> path, as long as the motion stays inside a feasible phase-consistent tube.

But key events should be precise:

> gripper close, gripper open, contact, release, insertion, and phase transition
> should not be blurred into a distribution.

So the supervision target should not be uniformly point-wise and should not be
uniformly distributional. It should respect the heterogeneous temporal semantics
of robot manipulation.

## The Main Formula

Instead of forcing:

$$
\hat{x}_{t+k} \rightarrow x^\star_{t+k}
$$

we allow the prediction to match a progress interval inside a trajectory cloud:

$$
d_{\mathrm{tube}}(\hat{x}_{t+k})
=
\min_{\rho\in I_{t,k}}
d_{\mathcal{C}_r}(\hat{x}_{t+k},\rho)
$$

where $I_{t,k}$ is the allowed progress interval and $\mathcal{C}_r$ is the
trajectory cloud for phase $r$.

The full training objective combines progress-elastic continuous supervision
with strict key-event constraints:

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{tube}}
+
\lambda_e\mathcal{L}_{\mathrm{event}}
+
\lambda_c\mathcal{L}_{\mathrm{cross}}
+
\lambda_m\mathcal{L}_{\mathrm{mono}}
+
\lambda_v\mathcal{L}_{\mathrm{speed}}
$$

## Why This Might Be Interesting

- It separates progress error from true trajectory violation.
- It preserves multi-solution continuous motion without blurring key events.
- It gives action chunks a phase-aware supervision structure.
- It provides evaluation hooks beyond success rate, such as tube violation rate,
  event timing error, event pose error, wrong crossing rate, and progress
  backward rate.

## Documents

- English: `docs/en/readme.md`
- Chinese: `docs/zh-CN/readme.md`

## Chinese README

See `README.zh-CN.md`.

## License

This project is released under the MIT License. See `LICENSE`.
