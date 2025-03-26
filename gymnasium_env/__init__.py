from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorldEnv",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

register(
    id="gymnasium_env/ContainerStackingEnv-v0",
    entry_point="gymnasium_env.envs:ContainerStackingEnv",
)

# 预测装船顺序
register(
    id="gymnasium_env/ContainerSeqEnv-v0",
    entry_point="gymnasium_env.envs:ContainerSeqEnv",
)
