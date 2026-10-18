from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ForkliftNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 10 Hz control: 48 steps = 4.8 s of driving per env per iteration
    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 50
    experiment_name = "forklift_nav"
    # actor gets only deployable observations, critic also gets the privileged state
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    # Small actor: it runs in numpy inside the ROS node (no torch on the ROS side).
    # obs_normalization stays off so the exported weights are the whole policy.
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
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
    )
