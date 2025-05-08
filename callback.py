from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from utils.my_utils import count_ascending_order
import pdb

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.loggeinit_callbackdom_value", value)
        
        done =  self.training_env.env_method('if_callback')[0]
        
        #done = self.locals['dones']
        count = 0
       
        if done == 1:
           
            bay_weight = self.training_env.get_attr('bay_weight')[0]
            bay_port = self.training_env.get_attr('bay_port')[0]

            count1 = count_ascending_order(bay_weight)
            count2 = count_ascending_order(bay_port)

            self.logger.record("count/weight_count", count1)
            self.logger.record("count/port_count", count2)
        
        return True