# 2026-06-23 PATCS Probability Tube Followups

This note preserves the latest algorithm discussion so it survives context compaction.

## Observed Issue

The first probabilistic tube visualization may show unexplained local contractions outside gripper-transition points.

Likely causes to inspect:

1. Same-task demos may naturally cluster tightly at some non-event timesteps, producing small covariance.
2. The current tube mean is the aligned cloud mean, not strictly target-demo centered; trajectory crossings or multimodal regions can create artificial narrow waists.
3. Covariance is estimated independently per timestep, with no temporal smoothing, so neighboring sections can abruptly change size.
4. The local frame is computed directly from target tangents; tangent jitter or frame flips can twist the surface.
5. The current surface is a chain of Gaussian elliptical sections, not a true continuous KDE/GMM density-volume isosurface.

## Required Future Optimization

The target design is a smooth-edged probabilistic blob/tube envelope, not a raw segmented Gaussian tube.

Before training integration, add diagnostics for every target demo:

- `event_scale[t]`
- `cov_det[t]`
- principal radii per section
- distance to nearest gripper transition
- non-event local minima of radius/covariance

Then improve the core with:

- temporal smoothing over covariance/radius;
- parallel-transport or minimum-rotation local frames;
- configurable target-centered mean versus cloud-mean blending;
- GMM/KDE for multimodal sections;
- black-hole contraction limited to transition neighborhoods;
- smoother isosurface generation for visualization.

Keep this as a known algorithm optimization point: current contraction is only an initial design, not the final desired smooth probability envelope.

## 2026-06-24 Core Direction Update

The current implementation is a prototype:

```text
phase-wise 2D normal-plane Gaussian sections
-> stitched into a 3D tube surface
```

This is useful for fast synthetic visualization, but it is not the desired final PATCS probability cloud. It can produce stitching artifacts, local-frame twisting, and non-smooth boundaries.

The next core algorithm should be:

```text
global 3D probability density
-> demo_i-conditioned contraction field
-> olive envelope
-> narrow precision channel
-> final black-hole anchor
```

The large probability cloud should be computed first from all same-task trajectory points, using one of:

- global 3D Gaussian;
- 3D KDE;
- GMM;
- voxel density + isosurface.

Then a contraction field should condition that cloud on the current target demo and its gripper transitions. The black-hole region should not be only a tiny point-like window. It should be split into:

```text
wide cloud -> olive narrowing -> narrow precision channel -> final black-hole anchor
```

Planned config fields:

```text
density_mode = "global_3d_gaussian" | "phase_2d"
precision_channel_window
precision_channel_radius
blackhole_window
event_radius
```

The desired radius logic is:

```text
radius(t) =
  base_global_density_radius
  * olive_scale(t)
  * channel_scale(t)
  + event_radius
```

`precision_channel_window` should be much longer than `blackhole_window` so the model enters a narrow but continuous fine-operation corridor before the final hard event anchor.
