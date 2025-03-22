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
        self.now_reward = 0
       
        #离散动作空间
        # self.observation_space = Box(
        #     low=0, high=10, shape=(2,self.bay_width), dtype=np.float32
        # )
        self.observation_space = spaces.Dict({
            "next_container_weight": Box(low=0, high=100, shape=(1,), dtype=np.float32),
            #"heights": Box(low=0, high=self.bay_height, shape=(self.bay_width,), dtype=np.float32),
            "top_weights": Box(low=0, high=100, shape=(self.bay_width,), dtype=np.float32)
        })
       
        self.action_space = spaces.Discrete(self.bay_width)
        self.reset()
    
        
    def _get_obs(self):
        # weights = self.weights
        # unload_port = self.unload_port
        # weights = torch.tensor(weights, dtype=torch.float32).view(1, -1)
        # unload_port = torch.tensor(unload_port, dtype=torch.int32).view(1, -1)
        # observation = torch.cat((weights, unload_port), dim=0)
        
        observation = {
            "next_container_weight": [self.containers[self.current_container_index]],
            #"heights": self.heights,
            "top_weights": self.top_weights
        }
       
        return observation

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
        self.heights = np.zeros(self.bay_width, dtype=np.int32)
        self.top_weights = np.zeros(self.bay_width, dtype=np.int32)
        self.total_reward = 0

        #self.containers = np.loadtxt("weight.txt")  # 读取weight.txt文件作为self.containers
        #集装箱总重量固定为2500
        self.containers = np.random.rand(50)
        self.containers = self.containers / np.sum(self.containers) * 2500
        self.containers = np.round(self.containers)
        
        self.event_port = np.random.randint(0, 4, size=(self.bay_height,self.bay_width ))
        
        self.current_container_index = 0

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

        top_weight = 0
       
        # 垂直平衡约束
        if self.heights[action] == 0:
            top_weight = 0             
        else:
            top_weight = self.weights[self.heights[action] - 1][action] 
            
        
        if current_weight <= top_weight or top_weight == 0:
            reward += 10
        
        

        # 更新状态
        self.weights[self.heights[action] ][action] = current_weight
        self.unload_port[self.heights[action] ][action] = self.event_port[self.heights[action] ][action]

        self.heights[action] += 1 
        self.top_weights[action] = current_weight

        # 更新当前集装箱索引
        self.current_container_index += 1
        reward += self.current_container_index  
        

        # 检查是否所有位置堆满
        if np.argmin(self.heights) == self.bay_height:
            reward += 100 
            terminated = True

        self.now_reward = reward
        return observation, reward, terminated, truncated, info


    def render(self, mode="human"):
       
        print(f"当前集装箱索引: {self.current_container_index}")
        print("Bay 状态:")
        for i in range(self.bay_width):
            for j in range(self.bay_height):
                print(f"{self.weights[self.bay_width - i - 1][j]:.0f}", end=" ")
            print()
        print("=" * (self.bay_width * 4 - 1))
        
        print(self.top_weights)
        print(f"当前奖励: {self.now_reward}")
