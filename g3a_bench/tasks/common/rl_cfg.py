from mjlab.rl import RslRlOnPolicyRunnerCfg


def manipulation_rl_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(experiment_name=experiment_name)
