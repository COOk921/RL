import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from module.TrmEncoder import TrmEncoder
import torch
import pdb


#定义自己的gym环境
class ContainerStackingEnv(gym.Env):
    def __init__(self):
        super(ContainerStackingEnv, self).__init__()
        
        # 船舶 Bay 的尺寸
        self.bay_width = 6
        self.bay_height = 6
        self.encoder = TrmEncoder(input_size=128, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1)

       
        #离散动作空间
        self.observation_space = Box(
            low=0, high=10, shape=(self.bay_width,), dtype=np.float32
        )
        
        
        # self.observation_space = Box(
        #     low=0, high=1, shape=(self.bay_width*self.bay_height,128), dtype=np.float32
        #     )
       
        self.action_space = spaces.Discrete(self.bay_width)
        self.reset()
    
        
    def _get_obs(self):
        weights = self.weights
        unload_port = self.unload_port
       
        weights = torch.tensor(weights, dtype=torch.float32).view(1, -1)
        unload_port = torch.tensor(unload_port, dtype=torch.int32).view(1, -1)
        
        observation = torch.cat((weights, unload_port), dim=0)
        
        
        # 转化为浮点数组
    
        return self.heights.astype(np.float64)#[2, bay_width*bay_height]

        # [batch, bay_width*bay_height, d_k]
        # features = self.encoder(weights, unload_port) 
        # return features.detach().numpy() #[2, bay_width*bay_height]

  
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
       
        super().reset(seed=seed)
        # 状态矩阵
        self.weights = np.zeros((self.bay_height,self.bay_width), dtype=np.float32)  #[height,width]
        self.unload_port = np.zeros((self.bay_height,self.bay_width), dtype=np.int32)  #[height,width]
        self.heights = np.zeros(self.bay_width, dtype=np.int32)  #[width]

        #self.containers = np.loadtxt("weight.txt")  # 读取weight.txt文件作为self.containers
        #集装箱总重量固定为2500
        self.containers = np.random.rand(50)
        self.containers = self.containers / np.sum(self.containers) * 2500
        self.containers = np.round(self.containers)
        
        self.event_port = np.random.randint(0, 4, size=(self.bay_height,self.bay_width ))
        
        self.current_container_index = 0
        self.g = 0 
        observation = self._get_obs()   
        info = self._get_info()
       

        # TODO: 可视化
        # if self.render_mode == "human":
        #     self._render_frame()
       
        return observation, info
        
    def step(self, action): 
    
        observation = self._get_obs()
        info = self._get_info()
        terminated = False 
        truncated = False

        # 计算奖励
        reward = 0

        if self.heights[action] >= self.bay_height:
            self.current_container_index += 1

            return observation, -20, False, True, {"message": "超出堆放高度"} 

        current_weight = self.containers[self.current_container_index]
    
        reward -= current_weight*self.heights[action]*0.05

        # 更新状态
        self.weights[self.heights[action] ][action] = current_weight
        self.unload_port[self.heights[action] ][action] = self.event_port[self.heights[action] ][action]
        self.heights[action] += 1 

        # 更新当前集装箱索引
        self.current_container_index += 1
        
        # reward = self.current_container_index  
        

        # 检查是否所有位置堆满
        if np.argmin(self.heights) == self.bay_height:
            reward += 100 
            terminated = True

        return observation, reward, terminated, truncated, info

    # def step(self, action): # action 表示选择在哪一列堆放
    
    #     observation = self._get_obs()
    #     info = self._get_info()
    #     terminated = False
    #     truncated = False

        
    #     # 计算奖励
    #     reward = 0
    #     # if action == np.argmin(self.heights):
    #     #     reward += 10
    #     # elif action == np.argmax(self.heights):
    #     #     reward -= 5

    #     if self.current_container_index == len(self.containers) -3 :
    #         reward += 300
    #         return observation, reward, True, True, info

    #     if self.heights[action] >= self.bay_height:
    #         self.current_container_index += 1
    #         return observation, -100, False, True, {"message": "超出堆放高度"}

    #     current_weight = self.containers[self.current_container_index]

    
    #     # 更新状态
    #     self.weights[self.heights[action] ][action] = current_weight
    #     self.unload_port[self.heights[action] ][action] = self.event_port[self.heights[action] ][action]
    #     self.heights[action] += 1 

    #     # 更新当前集装箱索引
    #     self.current_container_index += 1
    #     return observation, reward, terminated, truncated,info

    def render(self, mode="human"):
       
        print(f"当前集装箱索引: {self.current_container_index}")
        print("Bay 状态:")
        for i in range(self.bay_width):
            for j in range(self.bay_height):
                print(f"{self.weights[self.bay_width - i - 1][j]:.0f}", end=" ")
            print()
       
        
        print("卸货港口:")
        for i in range(self.bay_width):
            for j in range(self.bay_height):
                print(f"{self.unload_port[self.bay_width - i - 1][j]:.0f}", end=" ")
            print()
        print("=" * (self.bay_width * 4 - 1))

