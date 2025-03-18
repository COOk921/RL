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
    
        self.observation_space = Box(
            low=0, high=1, shape=(2,self.bay_width*self.bay_height), dtype=np.float32
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
       
        return observation #[2, bay_width*bay_height]

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
        #随机的集装箱满足正太分布，均值为50，标准差为10
        self.containers = np.random.normal(50, 10, size=36)
        self.containers = np.clip(self.containers, 10, 100)
        self.containers = np.round(self.containers)

        self.event_port = np.random.randint(0, 4, size=(self.bay_height,self.bay_width ))
        
        self.current_container_index = 0
        observation = self._get_obs()   
        info = self._get_info()
       

        # TODO: 可视化
        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info
        

    def step(self, action): # action 表示选择在哪一列堆放
    
        observation = self._get_obs()
        info = self._get_info()

        if self.heights[action] >= self.bay_height:
            return observation, -100, True, False, {"message": "超出堆放高度"}
        
        current_weight = self.containers[self.current_container_index]
        
        # 计算奖励
        reward = 0
        
        # if action == np.argmin(self.heights):
        #     reward += 10
        # elif action == np.argmax(self.heights):
        #     reward -= 10

       
        # 1. 重量差奖励
        top_weight = self.weights[self.heights[action]][action]  
        if top_weight == 0:         # 如果当前列为空，给予5点奖励
            reward += 50  
        else:                      # 如果当前列不为空，根据重量差给予奖励
            weight_diff = abs(current_weight - top_weight)
            reward += max(0, 50 - weight_diff)  # 权重差小得分高
        
        # 2. 左右平衡奖励
        left_weight = sum(self.weights[:,:self.bay_width // 2]) # 左侧重量
        right_weight = sum(self.weights[:,self.bay_width // 2:]) # 右侧重量
        balance_penalty = abs(sum(left_weight) - sum(right_weight))   # 左右平衡惩罚
        reward -= balance_penalty * 0.1  # 平衡性惩罚
        

        # 3. 动作完成奖励
        if self.current_container_index == len(self.containers) - 1:
            reward += 300

        
        terminated = False
        truncated = False
    
        # 更新状态
        self.weights[self.heights[action] ][action] = current_weight
        self.unload_port[self.heights[action] ][action] = self.event_port[self.heights[action] ][action]
        self.heights[action] += 1 

        # 更新当前集装箱索引
        self.current_container_index += 1
        
        return observation, reward, terminated, truncated,info

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

