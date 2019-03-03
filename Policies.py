import numpy as np
class RandomPolicy(object):
    def __init__(self, env):
        self.env = env
        self.action_space = np.array(range(self.env.action_space.n))
    
    def sample_action(self, observation):
        return np.random.choice(self.action_space)