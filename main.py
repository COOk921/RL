import pdb

import gymnasium
from stable_baselines3 import PPO,DQN,A2C,TD3
from stable_baselines3.common.env_checker import check_env

import gymnasium_env
from gymnasium_env.envs.ContainerStackingEnv import ContainerStackingEnv
from gymnasium_env.envs.grid_world import GridWorldEnv

from module.TrmEncoder import TransformerFeaturesExtractor


model_path = "./checkpoints/ContainerStackingEnv-v0_ppo_model"
base_model_path = "./checkpoints/base/ContainerStackingEnv-v0_ppo_model"

def train():
    env = gymnasium.make('gymnasium_env/ContainerStackingEnv-v0')
    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=0,
        n_steps = 256,
        learning_rate=0.0001,
        tensorboard_log="./tensorboard/learn_setting", #tensorboard --logdir=./tensorboard
        device="cuda"   
        # policy_kwargs=policy_kwargs
    )


    #model = PPO.load(model_path, env=env)

    model.learn(total_timesteps=80000, progress_bar=False) 
    
    model.save(model_path)
    


def test():
    env = gymnasium.make('gymnasium_env/ContainerStackingEnv-v0')
    
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
    
    

