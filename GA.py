import numpy as np
import random
import copy
import sys
from pathlib import Path
import pdb

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
def calculate_penalty(sequence, slot_order, rows, cols):
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
                weight_penalty += 1  # 惩罚计数
        
        # 目的港口约束：上近下远（目的港口随行号递减）
        for k in range(1, len(column)):
            i_curr, cont_curr = column[k]
            i_prev, cont_prev = column[k-1]
            if i_curr > i_prev and cont_curr.destination < cont_prev.destination:
                destination_penalty += 1  # 惩罚计数
    
    # 总惩罚
    return weight_penalty + destination_penalty

# 遗传算法
def genetic_algorithm(num_containers, rows, cols, type):
    # 初始化
    containers, slot_order = initialize_sequence(num_containers, rows, cols, type)
    
    # GA 参数
    population_size = 100
    generations = 500
    tournament_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.1
    
    # 初始化种群
    population = [copy.deepcopy(containers) for _ in range(population_size)]
    for individual in population:
        random.shuffle(individual)
    
    best_sequence = copy.deepcopy(population[0])
    best_penalty = calculate_penalty(best_sequence, slot_order, rows, cols)
    
    # 主循环
    for generation in range(generations):
        new_population = []
        
        # 精英保留：保留最优个体
        current_best = min(population, key=lambda seq: calculate_penalty(seq, slot_order, rows, cols))
        current_best_penalty = calculate_penalty(current_best, slot_order, rows, cols)
        if current_best_penalty < best_penalty:
            best_sequence = copy.deepcopy(current_best)
            best_penalty = current_best_penalty
        new_population.append(copy.deepcopy(current_best))
        
        # 选择、交叉、变异
        while len(new_population) < population_size:
            # 锦标赛选择
            def tournament_select():
                tournament = random.sample(population, tournament_size)
                return min(tournament, key=lambda seq: calculate_penalty(seq, slot_order, rows, cols))
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # 交叉
            if random.random() < crossover_rate:
                child1, child2 = ordered_crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # 变异
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        # 更新种群
        population = new_population[:population_size]
        
        #print(f"Generation {generation}: Best Penalty = {best_penalty}")
    
    # 将最优顺序转换为Bay布局
    best_bay = sequence_to_bay(best_sequence, slot_order, rows, cols)
    return best_bay, best_penalty

# 交叉操作：有序交叉（Ordered Crossover, OX）
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [None] * size, [None] * size
    
    # 随机选择交叉点
    start, end = sorted(random.sample(range(size), 2))
    
    # 复制父代的交叉片段
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]
    
    # 填充剩余位置
    def fill_remaining(child, parent, start, end):
        pos = (end + 1) % size
        parent_pos = (end + 1) % size
        while None in child:
            while parent[parent_pos] in child[start:end+1]:
                parent_pos = (parent_pos + 1) % size
            if child[pos] is None:
                child[pos] = parent[parent_pos]
                parent_pos = (parent_pos + 1) % size
            pos = (pos + 1) % size
        return child
    
    child1 = fill_remaining(child1, parent2, start, end)
    child2 = fill_remaining(child2, parent1, start, end)
    
    return child1, child2

# 变异操作：交换两个位置
def mutate(sequence):
    new_sequence = copy.deepcopy(sequence)
    idx1, idx2 = random.sample(range(len(sequence)), 2)
    new_sequence[idx1], new_sequence[idx2] = new_sequence[idx2], new_sequence[idx1]
    return new_sequence

# 打印Bay布局
def print_bay(bay, rows, cols):
    weight_count = 0
    port_count = 0

    print("Bay weight:")
    for i in range(rows):
        for j in range(cols):
            if bay[rows-1-i][j] is None:
                print(f"{0:3.0f}", end=" ")
            else:
                print(f"{bay[rows-1-i][j].weight:5.0f}", end=" ")
                if bay[rows-i-1][j] is not None and rows-i-1 > 0:
                    if bay[rows-1-i][j].weight > bay[rows-i-2][j].weight:
                        weight_count += 1
        print()

    print("重量倒箱:", weight_count)
    print("重量倒箱率：", weight_count / (300) * 100, "%")
   
    print("Bay port:")
    for i in range(rows):
        for j in range(cols):
            if bay[rows-1-i][j] is None:
                print(f"{0:3.0f}", end=" ")
            else:
                print(f"{bay[rows-1-i][j].destination:5.0f}", end=" ")
                if bay[rows-i-1][j] is not None and rows-i-1 > 0:
                    if bay[rows-1-i][j].destination < bay[rows-i-2][j].destination:
                        port_count += 1
        print()
    print("港口倒箱:", port_count)
    print("港口倒箱率：", port_count / (300) * 100, "%")

# 测试
if __name__ == "__main__":
    num_containers = 300
    rows = 17
    cols = 17
    type = "uniform"  # "uniform", "heavy", "light", "mixed"

    best_bay, best_penalty = genetic_algorithm(num_containers, rows, cols, type=type)
    print("Best Penalty:", best_penalty)
    print("Best Bay Layout:")
    print_bay(best_bay, rows, cols)