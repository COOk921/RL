![image-20250323222601527](C:\Users\ytn30\AppData\Roaming\Typora\typora-user-images\image-20250323222601527.png)

- 蓝色的表示是在full-bay基础上训练的，发现Reward没有上升趋势
- 粉色的表示重新训练，Reward上升，但是并没有突破蓝色线，表明重量因素并不是很好训练？



```python
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
from module.TrmEncoder import TrmEncoder
import torch
import pdb

class ContainerStackingEnv(gym.Env):
    def __init__(self):
        super(ContainerStackingEnv, self).__init__()
        
        # 船舶 Bay 的尺寸
        self.bay_width = 6
        self.bay_height = 6
        # self.encoder = TrmEncoder(input_size=128, d_k=128, d_v=128, n_heads=4, is_layer_norm=True, attn_dropout=0.1)
        self.now_reward = 0
        #self.containers = np.loadtxt("weight.txt")

       
        self.observation_space = spaces.Dict({
            "next_container_weight": Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "heights": Box(low=0, high=self.bay_height, shape=(self.bay_width,), dtype=np.float32),
            "top_weights": Box(low=0, high=100, shape=(self.bay_width,), dtype=np.float32)
        })
       
        #self.action_space = spaces.Discrete(self.bay_width)
        self.action_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        self.reset()
    
        
    def _get_obs(self):
    
        observation = {
            "next_container_weight": np.array([self.containers[self.current_container_index]], dtype=np.float32),
            "heights": self.heights,
            "top_weights": self.top_weights
        }
       
        return observation
  
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

       
        
        #集装箱总重量固定为3600
        self.containers = np.random.rand(37)
        self.containers = self.containers / np.sum(self.containers) * 1400
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
       
      
        terminated = False 
        truncated = False
       
        if 0 <= action < 1/6:
            action = 0
        elif 1/6 <= action < 2/6:
            action = 1
        elif 2/6 <= action < 3/6:
            action = 2
        elif 3/6 <= action < 4/6:
            action = 3
        elif 4/6 <= action < 5/6:
            action = 4
        else:
            action = 5


        # 随机action
        #action = np.random.randint(0, self.bay_width)
        
        
        reward = 0
        if self.current_container_index >= len(self.containers) - 2:
            truncated = True
        if self.heights[action] >= self.bay_height:     #Bay 溢出
            self.current_container_index += 1
            if np.min(self.heights) == self.bay_height:  #Bay满 溢出
                terminated = True
                reward += 100
            else:               #Bay未满 溢出
                #terminated = True                
                reward -= 20
            self.now_reward = reward

            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info 

        current_weight = self.containers[self.current_container_index]

        top_weight = 0

        
        # 垂直平衡约束
        if self.heights[action] == 0:
            top_weight = 0             
        else:
            top_weight = self.weights[self.heights[action] - 1][action] 
            
        if current_weight <= top_weight:
            reward += 10 - abs(current_weight - top_weight) / 10
        elif  top_weight == 0:
            reward += 10
        else:
            reward -= abs(current_weight - top_weight) / 10
        

        # 更新状态
        self.weights[self.heights[action] ][action] = current_weight
        self.unload_port[self.heights[action] ][action] = self.event_port[self.heights[action] ][action]

        self.heights[action] += 1 
        self.top_weights[action] = current_weight

        # 更新当前集装箱索引
        self.current_container_index += 1
        #reward += self.current_container_index  
        
        # 如果Bay满，则终止
        if np.min(self.heights) == self.bay_height:
            reward += 100 
            terminated = True
        
        self.now_reward = reward
       
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info


    def render(self, mode="human"):
       
        print(f"当前集装箱索引: {self.current_container_index}, 当前重量: {self.containers[self.current_container_index-1]}")
        print("Bay 状态:")
        for i in range(self.bay_width):
            for j in range(self.bay_height):
                print(f"{self.weights[self.bay_width - i - 1][j]:3.0f}", end=" ")
            print()
        
        
        print(self.top_weights)
        print(f"当前奖励: {self.now_reward}")
        print("=" * (self.bay_width * 4 - 1))

```

