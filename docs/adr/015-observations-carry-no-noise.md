# Observations carry no noise

Every task published its observations twice: an `actor` group with
`enable_corruption=True` over terms that each declared a uniform noise, and a
clean `critic` group over the same terms. The noise is gone, on both groups and
on every term, and no task declares any.

Observation noise is a robustness device for an on-policy run that collects its
own data. Training here is QC-FQL, offline from recorded demonstrations, and
the demonstrations came through the uncorrupted observation. A corrupted
training observation would therefore describe a state the operator never saw,
and the offline and online halves would stop coming from one distribution.

The asymmetry was the same device seen from the other side: a critic that reads
the clean state while the actor reads a noisy one is a PPO arrangement. QC-FQL
has no such split.

## What it actually changed

Nothing on either QC path. Both the adapter in `g3a_bench/qc/env.py` and the
`play` cfg the recorder builds already forced the flag off, so the noise had
never reached a demonstration or a training observation. It survived only as a
default that two callers had to undo, and as noise declared on terms where a
reader would reasonably assume it applied.

The values it removed: ±0.01 rad on joint positions, ±0.01 m and ±0.01 rad on
the end-effector and object poses, and **±1.5 rad/s on joint velocities**, which
is large enough to dominate the signal it was added to.

Observation widths are unchanged. Noise never touched dimensions, so a place
observation is still 42 wide and no recorded dataset is invalidated.

## Consequences

- PPO through mjlab's registry now trains on clean observations. mjlab requires
  an `rl_cfg` to register a task, which is why the RSL-RL config exists; it is
  not the training path (see ADR-002).
- The adapter keeps setting `enable_corruption = False`. It is redundant, and
  it stays as the seam's own guarantee (ADR-009).
- If sim-to-real robustness is ever wanted, put the noise back in one place
  (`tasks/common/env_cfg.make_manipulation_env_cfg`) and **record the
  demonstrations through it**. Turning it on for training alone reintroduces
  exactly the split this record closes.
