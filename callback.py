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
        # self.logger.record("random_value", value)

        done = self.locals['dones']
        count = 0
       
        if done == 1:
            bay_state = self.locals['infos'][0]['terminal_observation']['bay_state']
            count = count_ascending_order(bay_state)
            self.logger.record("count", count)
        
        return True