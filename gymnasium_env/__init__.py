from gymnasium.envs.registration import register

register(
    id="gymnasium_env/fjsp-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/ContainerStackingEnv-v0",
    entry_point="gymnasium_env.envs:ContainerStackingEnv",
)
