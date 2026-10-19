"""Forklift task registrations (discovered by Isaac Lab through the ``isaaclab.tasks`` entry point)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Forklift-Nav-Direct",
    entry_point=f"{__name__}.forklift_nav_env:ForkliftNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forklift_nav_env_cfg:ForkliftNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ForkliftNavPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Isaac-Forklift-Insert-Direct",
    entry_point=f"{__name__}.forklift_insert_env:ForkliftInsertEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forklift_insert_env_cfg:ForkliftInsertEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_insert_cfg:ForkliftInsertPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
