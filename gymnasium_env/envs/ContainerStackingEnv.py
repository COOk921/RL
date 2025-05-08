import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from module.TrmEncoder import TrmEncoder
import torch
import pdb
from utils.my_utils import weight_count,port_count

class ContainerStackingEnv(gym.Env):
    def __init__(self):
        super(ContainerStackingEnv, self).__init__()
        
        # 船舶 Bay 的尺寸
        self.bay_width = 7
        self.bay_height = 7
        # self.encoder = TrmEncoder(input_size=128, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1)
        self.now_reward = 0
     
        
        self.observation_space = spaces.Dict({
            "next_container_weight":  Box(low=0, high=100, shape=(1,), dtype=np.int32),
            "bay_state": Box(low=0, high=100, shape=(self.bay_width,self.bay_height), dtype=np.int32),
            #"heights": Box(low=0, high=self.bay_height, shape=(self.bay_width,), dtype=np.int32),
            # "top_weights": Box(low=0, high=3, shape=(self.bay_width,), dtype=np.int32)
        })
       
        self.action_space = spaces.Discrete(self.bay_width*self.bay_height)

        self.reset()
    
        
    def _get_obs(self):
        
        observation = {
            "next_container_weight": np.array([self.containers[self.current_container_index]], dtype=np.int32),
            "bay_state": self.weights,
            #"heights": self.heights,
            # "top_weights": self.top_weights
        }
        
        #pdb.set_trace()
        return observation
    
    def action_masks(self) -> np.ndarray:
        #mask = np.where(self.heights < self.bay_width, 1, 0)   # 0:屏蔽
        #mask = np.ones(self.bay_width*self.bay_height, dtype=np.int32)
        mask = np.where(self.weights == 0, 1, 0).flatten()
        return mask

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
       
        super().reset(seed=seed)
        # 状态矩阵
        self.weights = np.zeros((self.bay_height,self.bay_width), dtype=np.float32)  #[height,width]
        self.unload_port = np.zeros((self.bay_height,self.bay_width), dtype=np.int32)  #[height,width]
        self.heights = np.zeros(self.bay_width, dtype=np.int32)
        self.top_weights = np.zeros(self.bay_width, dtype=np.int32)
        self.total_reward = 0

        self.generate_containers()
        self.event_port = np.random.randint(0, 4, size=(self.bay_height,self.bay_width ))
        
        self.current_container_index = 0

        observation = self._get_obs()   
        info = self._get_info()
       

        # TODO: 可视化
        # if self.render_mode == "human":
        #     self._render_frame()
       
        return observation, info
        
    def step(self, action): 
        terminated = False 
        truncated = False
        reward = 0
    
        current_weight = self.containers[self.current_container_index]
        # [0,35] -> [0,5]
        action = (action // self.bay_width, action % self.bay_width)  # 转化为坐标 (row, col)

      
        
        reward = self.calculate_reward(action, current_weight)
        
        self.weights[action[0]][action[1]] = current_weight
        self.unload_port[action[0]][action[1]] = self.event_port[action[0]][action[1]]
        
        self.current_container_index += 1


        if self.current_container_index == self.bay_width*self.bay_height:
            terminated = True
       
        self.now_reward = reward
       
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def calculate_reward(self, action, current_weight):
        reward = 0
        row, col = action
        
        # 基础放置奖励
        reward += current_weight * (self.bay_height - row) * 0.1
        
        # 稳定性奖励
        if row > 0:
            if self.weights[row-1][col] < current_weight:
                reward -= 50  # 重的压轻的惩罚
        
        # 高度平衡奖励
        # height_diff = max(self.heights) - min(self.heights)
        # reward -= height_diff * 0.5
        
        # 终止状态奖励
        # if self.current_container_index == self.bay_width*self.bay_height:
        #     reward += 100
            
        return reward

    def generate_containers(self):

        #随机集装箱
        mean_weight = 50
        std_weight = 10
        self.containers = np.random.normal(mean_weight, std_weight, size=self.bay_width*self.bay_height + 1)
        #self.containers = np.clip( int(self.containers), 30, 70)

        #固定集装箱
        #self.containers = sorted([i for i in range(40,50)]) 

       # 从文件读取集装箱
        with open("weight.txt", "r") as file:
            data = file.read()
            self.containers = np.array([int(x) for x in data.split()], dtype=np.int32)
            
       

    def render(self, mode="human"):
       
        print(f"当前集装箱索引: {self.current_container_index}, 当前重量: {self.containers[self.current_container_index-1]}")
        print("Bay 状态:")
        for i in range(self.bay_width):
            for j in range(self.bay_height):
                print(f"{self.weights[self.bay_width - i - 1][j]:3.0f}", end=" ")
            print()
        
        # print(f"Bay内倒序: {count_ascending_order(self.weights)}")
        # print(f"元素序列倒序: {count_ascending_containers(self.containers)}")

        print(self.top_weights)
        print(f"当前奖励: {self.now_reward}")

        print("=" * (self.bay_width * 4 - 1))
