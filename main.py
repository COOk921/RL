import pdb
import numpy as np
import gymnasium
from stable_baselines3 import PPO,DQN,A2C,TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecNormalize
import torch

import gymnasium_env
from gymnasium_env.envs.ContainerStackingEnv import ContainerStackingEnv
from gymnasium_env.envs.grid_world import GridWorldEnv
from gymnasium_env.envs.ContainerSeqEnv import ContainerSeqEnv

from module.TrmEncoder import TransformerFeaturesExtractor
from model.network import CustomActorCriticPolicy,CustomFeaturesExtractor

from callback import TensorboardCallback


model_path = "./checkpoints/ContainerStackingEnv-v0_ppo_model"
base_model_path = "./checkpoints/base/ContainerStackingEnv-v0_ppo_model"


def train():
    #env = gymnasium.make('gymnasium_env/ContainerStackingEnv-v0')
    env = gymnasium.make("gymnasium_env/GridWorldEnv")

    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256)
    )
    
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        n_steps = 1024,
        learning_rate=0.0001,
        tensorboard_log="./tensorboard/learn_setting", # tensorboard --logdir=./tensorboard/learn_setting
        device="cuda"  ,
        policy_kwargs=policy_kwargs
    )

    
    #model = PPO.load(base_model_path, env=env)

    model.learn(total_timesteps=5000, progress_bar=False) 
    
    model.save(model_path)
def test():
    env = gymnasium.make('gymnasium_env/GridWorldEnv', render_mode="human")
    
    loaded_model = PPO.load(model_path, env=env)

    num_episodes = 50
    avg_reward  = 0
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
            if episode == num_episodes - 1:
                env.render()
        avg_reward += total_reward
        
       # print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    avg_reward /= num_episodes
    print(f"Average Reward = {avg_reward}")
    
    env.close()


from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_masks()

def train1(env):
    env = ActionMasker(env, mask_fn)
    #check_env(env)
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        net_arch=[
            dict(
                pi=[512, 256,256],  # 策略网络
                vf=[512, 256,256]   # 价值网络
                # pi=[256, 128,64],  # 策略网络
                # vf=[256, 128,64]   # 价值网络
            )
        ],
        activation_fn=torch.nn.ReLU
    )
    

    model = MaskablePPO(
        "MultiInputPolicy",  #CustomActorCriticPolicy
        env, 
        verbose = 0,
        n_steps = 1024,
        learning_rate=5e-4,
        batch_size=512,
        clip_range=0.20,  #
        ent_coef= 0.01,

        tensorboard_log="./tensorboard/learn_setting", # tensorboard --logdir=./tensorboard/learn_setting
        device="cuda" ,
        policy_kwargs=policy_kwargs
    )
   
    #model = PPO.load(base_model_path, env=env)

    model.learn(total_timesteps=500000, progress_bar=True ,  callback=TensorboardCallback())   #
    
    model.save(model_path)

def test1(env):
    env = ActionMasker(env, mask_fn)
    
    loaded_model = MaskablePPO.load(model_path, env=env)

    num_episodes = 1
    avg_reward  = 0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            
            masks = env.action_masks()
            action, _ = loaded_model.predict(obs,action_masks= masks)
            action = action.item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if episode == num_episodes - 1:
                env.render()
        avg_reward += total_reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    # avg_reward /= num_episodes
    # print(f"Average Reward = {avg_reward}")
    
    env.close()

def main():
    # without action mask
    #train()
    #test()

    env = gymnasium.make('gymnasium_env/ContainerSeqEnv-v0')
    # with action mask
    train1(env)
    test1(env)


if __name__ == "__main__":
    main()
    
    

