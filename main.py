import pdb

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import gymnasium_env
from gymnasium_env.envs.ContainerStackingEnv import ContainerStackingEnv
from gymnasium_env.envs.grid_world import GridWorldEnv

from module.TrmEncoder import TransformerFeaturesExtractor

def train():
    env = gymnasium.make('gymnasium_env/ContainerStackingEnv-v0')
    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        n_steps = 512,
        tensorboard_log="./tensorboard/",
        policy_kwargs=policy_kwargs
    )

    

    model.learn(total_timesteps=20000, progress_bar=False) 
    model_path = "./checkpoints/ContainerStackingEnv-v0_ppo_model"
    model.save(model_path)
    


def test():
    env = gymnasium.make('gymnasium_env/ContainerStackingEnv-v0')
    model_path = "./checkpoints/ContainerStackingEnv-v0_ppo_model"
    loaded_model = PPO.load(model_path, env=env)

    num_episodes = 1
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = loaded_model.predict(obs)
            action = action.item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    env.close()

def main():
    train()
    test()


if __name__ == "__main__":
    main()
    
    

