import numpy as np
from my_utils import weight_count,port_count

class random_rule:
    def __init__(self, rule):

        self.type = "mixed" #  light heavy,mixed  uniform
        self.rule = rule
        self.bay_width = 17
        self.bay_height = 17
        self.cont_num = self.bay_width * self.bay_height
        
        self.cont_weights = None
        self.cont_port = None
       
        self.bay_weight = np.zeros((self.bay_height,self.bay_width), dtype=np.int32)
        self.bay_port = np.zeros((self.bay_height,self.bay_width), dtype=np.int32)
        self.data()


        if self.rule == "random":
            self.random_fill()
        elif self.rule == "weight":
            self.weight_fill()
        elif self.rule == "port":
            self.port_fill()


    def data(self):
        with open(f"ContainerData/{self.type}/container_levels_300_{self.type}.txt", "r") as file:
            data = file.read()
            self.cont_weights = np.array([int(x) for x in data.split()], dtype=np.int32)
            self.cont_weights = self.cont_weights[:self.cont_num]

        with open("ContainerData/port.txt", "r") as file:
            self.cont_port = np.array([int(x) for x in file.read().split()], dtype=np.int32)
            self.cont_port = self.cont_port[:self.cont_num]


    def sort_weight(self,arr):
        indices = np.argsort(arr)[::-1]
        sorted_arr = arr[indices]
        
        return sorted_arr, indices
    def sort_port(self,arr):
        indices = np.argsort(arr)
        sorted_arr = arr[indices]
        
        return sorted_arr, indices
    
    # 随机填充
    def random_fill(self):
        now_index = 0
        for j in range(self.bay_height):
            for k in range(self.bay_width):
                self.bay_weight[j][k] = self.cont_weights[now_index]
                self.bay_port[j][k] = self.cont_port[now_index]
                now_index += 1
    # weight
    def weight_fill(self):
        sorted_weights, indices = self.sort_weight(self.cont_weights)
        sorted_ports = self.cont_port[indices]
        
        now_index = 0
        for j in range(self.bay_height):
            for k in range(self.bay_width):
                self.bay_weight[j][k] = sorted_weights[now_index]
                self.bay_port[j][k] = sorted_ports[now_index]
                now_index += 1

    # port
    def port_fill(self):
        sorted_ports, indices = self.sort_port(self.cont_port)
        sorted_weights = self.cont_weights[indices]
        
        now_index = 0
        for j in range(self.bay_height):
            for k in range(self.bay_width):
                self.bay_weight[j][k] = sorted_weights[now_index]
                self.bay_port[j][k] = sorted_ports[now_index]
                now_index += 1

    def show(self):
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
        
        print(f"重量倒序: {weight_count(self.bay_weight)}")
        print(f"港口倒序: {port_count(self.bay_port)}")


random_rule = random_rule("random")
random_rule.show()