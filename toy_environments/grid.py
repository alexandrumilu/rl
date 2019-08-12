import numpy as np
import queue
class Grid(object):
    def __init__(self,
                 shape = (4,4),
                 random = True,
                 prob_frozen = 0.8,
                 given_grid = None,
                 start = (0,0),
                 end = None,
                 reward_every_step = -1,
                 reward_fall = -100,
                 reward_reach_goal=100,
                ):
        """Initializes grid environment.
        
        Args:
        shape: tuple representing the shape of the grid.
        random: boolean that represents if grid to be generated should be random
        prob_frozen: probabiliy the square is frozen, i.e. navigable. If not frozen we die - bad.
        given_grid: if random=False the grid will be equal to given_grid.
        start: start position of our agent.
        end: goal of the agent.
        reward_every_step: if step on a frozen surface get this reward
        reward_fall: if step on an unfrozen surface get this reward
        reward_reach_goal: if reach goal get this reward
        
        """
        self.height = shape[0]
        self.width = shape[1]
        if random:
            self.lake = self.create_random_grid_environment(self.height, self.width, prob_frozen)
        else:
            assert(given_grid!=None) 
            self.lake = given_grid
        self.shortest_path = self.get_length_smallest_path(self.lake,start=start,end=end)
        self.transitions = self.get_transitions_tensor(lake=self.lake)
        self.terminal_states = self.get_done_states(lake=self.lake,end=end)
        self.rewards = self.set_rewards(lake=self.lake,
                                   end=end,
                                   reward_every_step=reward_every_step,
                                   reward_fall=reward_fall,
                                   reward_reach_goal=reward_reach_goal)
        self.start = start
        if end==None:
            self.end = (self.height-1,self.width-1)
        else:
            self.end = end
        self.start_state = self._state_from_readable(self.start[0],self.start[1],self.width)
        self.end_state = self._state_from_readable(self.end[0],self.end[1],self.width)
    def create_random_grid_environment(self,height=4,width=4, prob_frozen = 0.8):
        """Creates a random height X width frozen lake environment.
        The goal of the agent is to navigate from (0,0) to (height-1,width-1).
        This function only returns navigable environments. If the first random
        one does not generate an environment with a path from start to finish we
        try again. 

        Args:
        height: number of rows
        width: number of columns
        prob_frozen: probabiliy the square is frozen, i.e. navigable. If not frozen we die - bad.

        Returns:
        A numpy array of shape (height,width), with 2 for the starting space,
        3 for the finish, 1 for every frozen space and 0 for every non frozen space. 
        """
        navigable = False
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        while not navigable:
            lake = np.random.binomial(n=1,p=prob_frozen,size = (height,width))
            lake[0,0] = 1
            lake[-1,-1] = 1
            stack = [(0,0)]
            visited = np.zeros((height,width))
            visited[0,0] =1
            while len(stack)>0:
                (i,j) = stack.pop()
                for v,h in directions:
                    if i+v>=0 and i+v<height and j+h>=0 and j+h<width:
                        if lake[i+v,j+h]==1 and visited[i+v,j+h]==0:
                            stack.append((i+v,j+h))
                            visited[i+v,j+h] = 1
                        if i+v==height-1 and j+h==width-1:
                            navigable = True
                            lake[0,0] = 2
                            lake[height-1,width-1] = 3
                            return lake

    def get_length_smallest_path(self,lake,start=(0,0),end=None):
        """Runs bfs to get the optimal path in the lake environment.

        Args:
        lake - numpy array.
        start - tuple with starting coordinates
        end - tuple with ending coordinates

        Returns:
        integer that represents the length of the smallest path
        """

        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        height,width = lake.shape
        if end==None:
            end =  (height-1,width-1)
        q = queue.Queue()
        q.put(start)
        visited = -np.ones((height,width))
        visited[start] = 0
        while not q.empty():
            (i,j) = q.get()
            for v,h in directions:
                if (i+v,j+h)==end:
                    return visited[i,j]+1
                if i+v>=0 and i+v<height and j+h>=0 and j+h<width:
                    if lake[i+v,j+h]>0 and visited[i+v,j+h]<0:
                        q.put((i+v,j+h))
                        visited[i+v,j+h] = visited[i,j]+1
                    

    def _readable_state(self,state, width):
        i = state//width
        j = state%width
        return (i,j)

    def _state_from_readable(self,i,j,width):
        return i*width+j

    def get_transitions_tensor(self,lake):
        """Given a lake environment, computes a SXSXA tensor that represents the
        transition probabilities

        Args:
        lake - numpy array.

        Returns:
        SXSXA numpy array.
        """
        height, width = lake.shape
        states = np.arange(0,height*width)
        actions = np.arange(0,4)
        transitions = np.zeros((height*width,height*width,4))
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        for s in range(height*width):
            for a in range(4):
                i,j = self._readable_state(s,width)
                v,h = directions[a]
                if i+v>=0 and i+v<height and j+h>=0 and j+h<width:
                    s_prime = self._state_from_readable(i+v,j+h,width)
                    transitions[s_prime,s,a] = 1
        return transitions

    def get_done_states(self,lake, end=None):
        """Given a lake grid, return which states are terminal.
        The terminal states are the holes and the end state. 
        The environment should reset after reaching a terminal state. 

        Args:
        lake - numpy array.
        end - end position if different than bottom right corner. 

        Retruns:
        terminal_states - boolean array of shape (states,1)

        """
        height, width = lake.shape
        if end==None:
            end =  (height-1,width-1)
        terminal_states = np.zeros(height*width, dtype=bool)
        for s in range(height*width):
            i,j = self._readable_state(s,width)
            if lake[i,j]==0 or (i,j)==end:
                terminal_states[s] = True
        return terminal_states

    def set_rewards(self, lake, reward_every_step = -1, reward_fall = -100, reward_reach_goal = 100, end=None):
        """Given a lake grid, return a reward depending on the state you reach s.
        Args:
        lake: numpy array
        reward_every_step: if step on a frozen surface get this reward
        reward_fall: if step on an unfrozen surface get this reward
        reward_reach_goal: if reach goal get this reward
        Returns:
        rewards: numpy array of shape (states, actions)
        """
        num_states = lake.size
        height, width = lake.shape
        if end==None:
            end =  (height-1,width-1)

        rewards = np.zeros((num_states,4))
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        for s in range(num_states):
            i1,j1 = self._readable_state(s,width)
            for a in range(4):
                v,h = directions[a]
                if i1+v>=0 and i1+v<height and j1+h>=0 and j1+h<width:
                    i = i1+v
                    j = j1+h
                    if (i,j)==end:
                        rewards[s,a] = reward_reach_goal
                    elif lake[i,j]==0:
                        rewards[s,a] = reward_fall
                    else:
                        rewards[s,a] = reward_every_step
        return rewards