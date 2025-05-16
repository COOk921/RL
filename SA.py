import numpy as np
import random
import math
import copy
import sys
from pathlib import Path



# 定义集装箱类
class Container:
    def __init__(self, id, weight, destination):
        self.id = id
        self.weight = weight  # 重量（越大越重）
        self.destination = destination  # 目的港口（越大越远）

# 初始化集装箱和槽位顺序
def initialize_sequence(num_containers, rows, cols, type):
    containers = []

    with open(f"ContainerData/{type}/container_levels_{num_containers}_{type}.txt", "r") as file:
        data = file.read()
        cont_weights = np.array([int(x) for x in data.split()], dtype=np.int32)
        cont_weights = cont_weights[:num_containers]

    with open("ContainerData/port.txt", "r") as file:
        cont_port = np.array([int(x) for x in file.read().split()], dtype=np.int32)
        cont_port = cont_port[:num_containers]

    # 创建集装箱列表
    for i in range(num_containers):
        weight = cont_weights[i]
        destination = cont_port[i]
        containers.append(Container(i, weight, destination))

    
    # 定义槽位填充顺序：从下到上，从左到右
    slot_order = [(i, j) for i in range(rows) for j in range(cols)]
    
    return containers, slot_order

# 将集装箱顺序映射到Bay布局
def sequence_to_bay(sequence, slot_order, rows, cols):
    bay = [[None for _ in range(cols)] for _ in range(rows)]
    for idx, (i, j) in enumerate(slot_order):
        if idx < len(sequence):
            bay[i][j] = sequence[idx]
        else:
            break  # 如果集装箱数量少于槽位数量
    return bay

# 计算目标函数（惩罚）
def calculate_penalty(sequence, slot_order, rows=7, cols=7):
    bay = sequence_to_bay(sequence, slot_order, rows, cols)
    weight_penalty = 0
    destination_penalty = 0
    
    # 检查每一列
    for j in range(cols):
        # 获取该列的集装箱（从下到上，行号0到6）
        column = [(i, bay[i][j]) for i in range(rows) if bay[i][j] is not None]
        if not column:
            continue
        
        # 重量约束：上轻下重（重量随行号递增）
        for k in range(1, len(column)):
            i_curr, cont_curr = column[k]
            i_prev, cont_prev = column[k-1]
            if i_curr > i_prev and cont_curr.weight > cont_prev.weight:
                weight_penalty +=1 # (cont_curr.weight - cont_prev.weight) * abs(i_curr - i_prev)
        
        # 目的港口约束：上近下远（目的港口随行号递减）
        for k in range(1, len(column)):
            i_curr, cont_curr = column[k]
            i_prev, cont_prev = column[k-1]
            if i_curr > i_prev and cont_curr.destination < cont_prev.destination:
                destination_penalty +=1 # (cont_prev.destination - cont_curr.destination) * abs(i_curr - i_prev)
    
    # 总惩罚（可调整权重）
    return weight_penalty + destination_penalty

# 生成邻域解（交换两个集装箱的顺序）
def get_neighbor(sequence):
    new_sequence = copy.deepcopy(sequence)
    # 随机选择两个不同位置
    idx1, idx2 = random.sample(range(len(sequence)), 2)
    new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
    return new_sequence

# 模拟退火算法
def simulated_annealing(num_containers, rows, cols, type):
    # 初始化
    sequence, slot_order = initialize_sequence(num_containers, rows, cols,type)
    current_penalty = calculate_penalty(sequence, slot_order)
    best_sequence = copy.deepcopy(sequence)
    best_penalty = current_penalty
    
    # 模拟退火参数
    initial_temp = 1000
    final_temp = 0.1
    alpha = 0.995
    max_iterations = 50000
    
    temp = initial_temp
    iteration = 0
    
    while temp > final_temp and iteration < max_iterations:
        # 生成邻域解
        neighbor_sequence = get_neighbor(sequence)
        neighbor_penalty = calculate_penalty(neighbor_sequence, slot_order)
        
        # 计算能量差
        delta = neighbor_penalty - current_penalty
        
        # 接受准则
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            sequence = neighbor_sequence
            current_penalty = neighbor_penalty
            
            # 更新最优解
            if current_penalty < best_penalty:
                best_sequence = copy.deepcopy(sequence)
                best_penalty = current_penalty
        
        #print(f"Iteration {iteration}: Current Penalty = {current_penalty}, Best Penalty = {best_penalty}")
        # 降低温度
        temp *= alpha
        iteration += 1
    
    # 将最优顺序转换为Bay布局
    best_bay = sequence_to_bay(best_sequence, slot_order, rows, cols)
    return best_bay, best_penalty



def print_bay(bay, rows , cols):
    

    weight_count = 0
    port_count = 0

    print("Bay weight:")
    for i in range(rows):
        for j in range(cols):
            if bay[rows-1-i][j] is None:
                print(f"{0:3.0f}",end=" ")
            else:
                print(f"{bay[rows-1-i][j].weight:5.0f}", end=" ")

                if bay[rows-i-1][j] is not  None and rows-i-1 > 0:
                    if  bay[rows-1-i][j].weight > bay[rows-i-2][j].weight:
                        weight_count += 1
                        print("*" ,end=" ")


        print()

    print("重量倒箱:", weight_count) 
    print("重量倒箱率：", weight_count / (100) * 100, "%")
   
    print("Bay port:")
    for i in range(rows):
        for j in range(cols):
            if bay[rows-1-i][j] is None:
                print(f"{0:3.0f}",end=" ")
              
            else:
                print(f"{bay[rows-1-i][j].destination:5.0f}", end=" ")
                if bay[rows-i-1][j] is not  None and rows-i-1 > 0:
                    if  bay[rows-1-i][j].destination < bay[rows-i-2][j].destination:
                        port_count += 1
                        print("*" ,end=" ")
           
        print()
    print("港口倒箱:", port_count)
    print("港口倒箱率：", port_count / (100) * 100, "%")




# 测试
if __name__ == "__main__":
    num_containers = 100  
    rows = 8
    cols = 8
    type = "uniform"  # "uniform", "heavy", "light", "mixed"

    best_bay, best_penalty = simulated_annealing(num_containers, rows, cols,type=type)
    print("Best Penalty:", best_penalty)
    print("Best Bay Layout:")
    print_bay(best_bay, rows, cols)