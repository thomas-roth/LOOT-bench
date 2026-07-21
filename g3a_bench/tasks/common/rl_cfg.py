"""Registry-required RL runner config stub -- not a PPO runner anything uses.

Not PPO residue: mjlab's task registry hard-requires an ``rl_cfg`` object for
every registered task (``mjlab.tasks.registry.register_mjlab_task`` has no
default for it), so some valid object must be passed regardless of training
algorithm. Training goes through the upstream ``qc`` repo (QC-FQL), never
mjlab's RSL-RL trainer -- see ``g3a_bench/qc.py`` and the README's "Training"
section -- so only two fields on this object are ever actually read:
``experiment_name`` (checkpoint-path naming, read by mjlab's own
``--agent trained`` play mode) and ``clip_actions`` (read by mjlab's generic
``RslRlVecEnvWrapper`` and ``tests/viewer/measure_viewer_fps.py``). Everything
else uses mjlab's own defaults.
"""

from mjlab.rl import RslRlOnPolicyRunnerCfg


def manipulation_rl_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(experiment_name=experiment_name)
