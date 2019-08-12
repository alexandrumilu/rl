import numpy as np
class Value_iteration(object):
    def __init__(self,env):
        """Initializes an agent that will apply the value iteration algorithm to solve a discrete environment
        with known dynamics and rewards.
        Args:
        env: the environment - an object in Grid class. 
        """
        self.env = env
        self.lake = env.lake
        
    def train(self, epsilon = 0.0001, discount_factor = 0.99, verbose = False):
        """Given a navigable lake grid, trains the value iteration algorithm until convergence.
        Algorithm is considered to have converged when |V_(k+1)-V_k|_(\infty) < epsilon.
        """
        transitions = self.env.transitions
        terminal = self.env.terminal_states
        rewards = self.env.rewards
        num_states = self.lake.size
        values = np.zeros(num_states)
        done = False
        while not done:
            new_values = np.max((rewards+
                                discount_factor*np.dot(np.transpose(transitions,axes=(1,2,0)),values*(1-terminal))),
                                axis=1)
            if verbose:
                print(new_values)
            if np.max(new_values-values)<epsilon:
                policy = np.argmax((rewards+
                                discount_factor*np.dot(np.transpose(transitions,axes=(1,2,0)),values*(1-terminal))),
                                axis=1)
                return policy,values
            values = new_values
    def _readable_state(self,state, width):
        i = state//width
        j = state%width
        return (i,j)
    def compute_shortest_path_using_policy(self, policy, verbose = False):
        """Given a deterministic policy, computes how long it will take to reach the goal.
        Args:
        policy: a numpy array of shape (num_states,) that gives the recommended action at each state.
        verbose: if True prints every grid cell we visit. 
        """
        height, width = self.lake.shape
        state = self.env.start_state
        end_state = self.env.end_state
        transitions = self.env.transitions
        count = 0
        while 1:
            count+=1
            if verbose:
                print(self._readable_state(state,width))
            next_state = np.argmax(transitions[:,state,policy[state]])
            if next_state==end_state:
                return count
            state=next_state