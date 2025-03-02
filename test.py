from utils.my_utils import load_model
from agent import PolicyNetwork
from utils.config import parse_args
from utils.my_utils import dict_2_tensor
from ppo import Actor
import gymnasium
import gymnasium_env

import torch
import pdb

load_path = './model/ppo_fjsp.pth'

env = gymnasium.make('gymnasium_env/fjsp-v0', render_mode='human')

args = parse_args()

#初始化策略网络
policy_net = Actor(args)

#加载模型
policy_net = load_model(policy_net, load_path)


state_dict, _ = env.reset()
done = False
episode_reward = 0

while not done:
    state = dict_2_tensor(state_dict, args)
    action_prob = policy_net(state)
    action = torch.argmax(action_prob).item()
  

    if action == 0:
        print('right')
    elif action == 1:
        print('up')
    elif action == 2:
        print('left')
    elif action == 3:
        print('down')

    _, reward, done, _, _ = env.step(action)
    print(reward)
    episode_reward += reward
    env.render()

print(f"Episode Reward: {episode_reward}")

env.close()