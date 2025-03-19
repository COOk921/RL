**Agent填满位置**

每个集装箱都给予奖励，前面的奖励小，后面逐渐增大，鼓励合理位置填充

```python
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

        # 更新状态
        self.weights[self.heights[action] ][action] = current_weight
        self.unload_port[self.heights[action] ][action] = self.event_port[self.heights[action] ][action]
        self.heights[action] += 1 

        # 更新当前集装箱索引
        self.current_container_index += 1
        
        reward = self.current_container_index  

        # 检查是否所有位置堆满
        if np.argmin(self.heights) == self.bay_height:
            reward += 100 
            terminated = True

        return observation, reward, terminated, truncated, info
```

