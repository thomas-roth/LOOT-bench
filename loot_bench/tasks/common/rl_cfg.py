"""Runner config required by mjlab's task registry.

Training is QC-FQL in the upstream ``qc`` checkout, not RSL-RL PPO, so no field
here is read. ``register_mjlab_task`` requires an ``rl_cfg``, and this supplies
one. See ``docs/adr/002-training-lives-in-the-upstream-qc-checkout.md``.

``experiment_name`` stays per task because ``loot-play`` resolves a checkpoint
under ``logs/rsl_rl/<experiment_name>/``; one shared default would put every
task in the same directory.
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg


def manipulation_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(experiment_name=experiment_name)
