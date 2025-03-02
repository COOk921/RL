import gymnasium
import gymnasium_env
from agent import PPOAgent
from ppo import PPO_discrete
from tqdm import tqdm
import torch
from utils.my_utils import save_model
from utils.config import parse_args
from utils.replayBuffer import ReplayBuffer
from utils.normalization import Normalization,RewardScaling
from utils.my_utils import dict_2_tensor
import numpy as np

import pdb
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

env = gymnasium.make('gymnasium_env/fjsp-v0')


state_dim = 5 
action_dim = env.action_space.n

# agent = PPOAgent(state_dim * state_dim, action_dim, device)

total_steps = 0 

agent  = PPO_discrete(args)
replay_buffer = ReplayBuffer(args)
state_norm = Normalization(shape=args.state_dim)

if args.use_reward_norm: 
    reward_norm = Normalization(shape=1)
elif args.use_reward_scaling: 
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

max_steps = 50
episode = 0
while total_steps < args.max_train_steps:
    state, _ = env.reset()
    # turn state into tensor
    state = dict_2_tensor(state, args)
   
    # if args.use_state_norm:
    #     state = state_norm(state)

    if args.use_reward_scaling:
        reward_scaling.reset()

    done = False
    episode_steps = 0
    episode_reward = 0

    while not done:
        episode_steps += 1    #累计步数

        action, log_prob = agent.choose_action(np.array(state))
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = dict_2_tensor(next_state, args)

        done = terminated or truncated
      
        # if args.use_state_norm:
        #     next_state = state_norm(next_state)
        
        if args.use_reward_norm:
            reward = reward_norm(reward)
        elif args.use_reward_scaling:
            reward = reward_scaling(reward)
        
        if done and episode_steps != max_steps:
            dw = 1  # dead
        else:
            dw = 0  # not dead

       
        replay_buffer.store(state, action, log_prob, reward, next_state, dw, done)
        
        state = next_state
        total_steps += 1
        episode_reward += reward
        
        # Update
        if replay_buffer.count == args.batch_size:
            agent.update(replay_buffer, total_steps)
            replay_buffer.count = 0
        
        if episode_steps == max_steps:
            break
    
    episode += 1
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")
   

save_model(agent.actor, agent.critic, agent.optimizer_actor, agent.optimizer_critic, "./model/ppo_fjsp.pth")


env.close()





