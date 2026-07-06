"""PPO runner configuration for the object-insertion task category."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

# CNN encoder for camera observations (mirrors the mjlab YAM vision reference).
_VISION_CNN_CFG = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}
_VISION_MODEL_CLS = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"


def insertion_ppo_runner_cfg(
    experiment_name: str, vision: bool = False
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner cfg. With ``vision=True`` the actor/critic gain a CNN encoder
    and consume the "camera" observation group (for ``obs_mode="camera"``)."""
    cnn_cfg = _VISION_CNN_CFG if vision else None
    model_cls = _VISION_MODEL_CLS if vision else "MLPModel"
    obs_groups = (
        {"actor": ("actor", "camera"), "critic": ("critic", "camera")}
        if vision
        else {"actor": ("actor",), "critic": ("critic",)}
    )
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=cnn_cfg,
            class_name=model_cls,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=cnn_cfg,
            class_name=model_cls,
        ),
        obs_groups=obs_groups,
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=15_000,
    )
