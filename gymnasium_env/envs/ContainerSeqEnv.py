import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from module.TrmEncoder import TrmEncoder
import torch
import pdb
from utils.my_utils import weight_count,port_count

class ContainerSeqEnv(gym.Env):
    def __init__(self):
        super(ContainerSeqEnv, self).__init__()
        
        self.bay_width = 17
        self.bay_height = 17
        self.cont_num = self.bay_width*self.bay_height
        self.now_reward = 0

        self.observation_space = spaces.Dict({
            "cont_weights": Box(low=0, high=10, shape=(self.cont_num,), dtype=np.int32),
            "cont_port": Box(low=0, high=3, shape=(self.cont_num,), dtype=np.int32),

            # "top_ports": Box(low=0, high=2, shape=(self.bay_width,), dtype=np.int32),
            # "top_weights": Box(low=0, high=10, shape=(self.bay_width,), dtype=np.int32),
            "bay_weight": Box(low=0, high=10, shape=(self.bay_height,self.bay_width), dtype=np.int32),
            "bay_port": Box(low=0, high=2, shape=(self.bay_height,self.bay_width), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(self.cont_num)
        self.now_action = 0
        self.total_weight = 0

        self.reset()
    
        
    def _get_obs(self):
        observation = {
          "cont_weights": self.cont_weights,
          "cont_port": self.cont_port,

        #   "top_ports": self.top_ports,
        #   "top_weights": self.top_weights,
          "bay_weight":  self.bay_weight,
          "bay_port": self.bay_port
        }

        return observation
    
    def action_masks(self) -> np.ndarray:
        return self.mask

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
       
        super().reset(seed=seed)
        self.generate_containers()

        self.total_weight = np.sum(self.cont_weights)
        self.now_cont_index = 0
        self.bay_weight = np.zeros((self.bay_height,self.bay_width), dtype=np.int32)
        self.bay_port = np.zeros((self.bay_height,self.bay_width), dtype=np.int32)
        self.top_weights = np.zeros((self.bay_width,), dtype=np.int32)
        self.top_ports = np.zeros((self.bay_width,), dtype=np.int32)
        self.mask = np.ones((self.cont_num,), dtype=np.int32)
        self.now_reward = 0

        observation = self._get_obs()   
        info = self._get_info()
       
        return observation, info
        
    def step(self, action): 
        terminated = False 
        truncated = False
        self.now_action = action
        reward = 0
        
       
        current_weight = self.cont_weights[action]
        current_port = self.cont_port[action]

        self.now_cont_index += 1  #[0,self.cont_num-1]
        
        # row:[0,self.bay_height-1]
        # col:[0,self.bay_width-1]
        row,col = ((self.now_cont_index -1)  // self.bay_width, (self.now_cont_index -1) % self.bay_width) 
        self.bay_weight[row][col] = current_weight
        self.bay_port[row][col] = current_port

        self.top_weights[col] = current_weight
        self.top_ports[col] = current_port

        self.mask[action] = 0

        
        reward = self.calculate_reward(row, col, current_weight, current_port)

        if self.now_cont_index == self.cont_num :
            terminated = True

        
    
        self.now_reward += reward

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def calculate_reward(self, row, col, current_weight, current_port):
        reward = 0
        
        # 基础放置奖励
        #reward += (current_weight)* (self.bay_height - row) * 0.1 
        
        weight_reward, port_reward = 0, 0
        # 稳定性奖励
        if row > 0:
            if self.bay_weight[row-1][col] <= current_weight:
                weight_reward -= 1
            else:
                weight_reward += 0

        if row > 0:
            if current_port >= self.bay_port[row - 1 ][col]:
                port_reward += 0
            else:
                port_reward -= 1
      
          
        reward = weight_reward  + port_reward 
        return reward

    def generate_containers(self):
        
        #随机集装箱
        # mean_weight = 50
        # std_weight = 10
        # self.cont_weights = np.random.normal(mean_weight, std_weight, size=self.cont_num).astype(int)

        #固定集装箱
        #self.cont_weights = sorted([i for i in range(40,50)]) 

       # 从文件读取集装箱
        with open("ContainerData/mixed/container_levels_300_mixed.txt", "r") as file:
            data = file.read()
            self.cont_weights = np.array([int(x) for x in data.split()], dtype=np.int32)
            self.cont_weights = self.cont_weights[:self.cont_num]

        with open("ContainerData/port.txt", "r") as file:
            self.cont_port = np.array([int(x) for x in file.read().split()], dtype=np.int32)
            self.cont_port = self.cont_port[:self.cont_num]
        
       

    def render(self, mode="human"):
       
        print(f"当前集装箱索引: {self.now_cont_index}, 当前重量: {self.cont_weights[self.now_action]}, 当前港口: {self.cont_port[self.now_action]}")
        print("Bay weight:")

        for i in range(self.bay_height):
            for j in range(self.bay_width):
                print(f"{self.bay_weight[self.bay_height - i -1 ][j ]:3.0f}", end=" ")
            print()

        print("Bay port:")
        for i in range(self.bay_height):
            for j in range(self.bay_width):
                print(f"{self.bay_port[self.bay_height - i -1 ][j]:3.0f}", end=" ")
            print()
        
        print(f"Bay内倒序: {weight_count(self.bay_weight)}")
        # print(f"元素序列倒序: {count_ascending_containers(self.containers)}")

        print(self.top_weights)
        print(f"当前奖励: {self.now_reward}")

        print("=" * (self.bay_width * 4 - 1))

    def if_callback(self):
        if self.now_cont_index == self.cont_num -1 :
            return 1
        return  0


    


