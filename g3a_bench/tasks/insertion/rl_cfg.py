"""PPO runner configuration for the object-insertion task category."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from g3a_bench.tasks.common.rl_cfg import manipulation_ppo_runner_cfg


def insertion_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    return manipulation_ppo_runner_cfg(experiment_name, max_iterations=15_000)
