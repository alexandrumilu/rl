import numpy as np
import queue
def create_random_grid_environment(height=4,width=4, prob_frozen = 0.8):
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
    
def get_length_smallest_path(lake,start=(0,0),end=None):
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
                    

def _readable_state(state, width):
    i = state//width
    j = state%width
    return (i,j)

def _state_from_readable(i,j,width):
    return i*width+j

def get_transitions_tensor(lake):
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
            i,j = _readable_state(s,width)
            v,h = directions[a]
            if i+v>=0 and i+v<height and j+h>=0 and j+h<width:
                s_prime = _state_from_readable(i+v,j+h,width)
                transitions[s_prime,s,a] = 1
    return transitions

def get_done_states(lake, end=None):
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
        i,j = _readable_state(s,width)
        if lake[i,j]==0 or (i,j)==end:
            terminal_states[s] = True
    return terminal_states

def set_rewards(lake, reward_every_step = 0.1, reward_fall = 0, reward_reach_goal = 1, end=None):
    """Given a lake grid, return a reward depending on the state you reach s.
    Args:
    lake - numpy array
    reward_every_step - if step on a frozen surface get this reward
    reward_fall - if step on an unfrozen surface get this reward
    reward_reach_goal - if reach goal get this reward
    Returns:
    rewards - numpy array of shape (states, actions)
    """
    num_states = lake.size
    height, width = lake.shape
    if end==None:
        end =  (height-1,width-1)
    
    rewards = np.zeros((num_states,4))
    directions = [(0,1),(1,0),(0,-1),(-1,0)]
    for s in range(num_states):
        i1,j1 = _readable_state(s,width)
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
    
def value_iteration_algorithm(lake, epsilon = 0.0001, discount_factor = 0.8):
    """Given a navigable lake grid, trains the value iteration algorithm until convergence.
    Algorithm is considered to have converged when |V_(k+1)-V_k|_(\infty) < epsilon.
    """
    transitions = get_transitions_tensor(lake)
    terminal = get_done_states(lake)
    rewards = set_rewards(lake)
    num_states = lake.size
    values = np.zeros(num_states)
    done = False
    while not done:
        new_values = np.max((rewards+
                            discount_factor*np.dot(np.transpose(transitions,axes=(1,2,0)),values*(1-terminal))),
                            axis=1)
        if np.max(new_values-values)<epsilon:
            policy = np.argmax((rewards+
                            discount_factor*np.dot(np.transpose(transitions,axes=(1,2,0)),values*(1-terminal))),
                            axis=1)
            return policy,values
        values = new_values
def compute_shortest_path_using_policy(lake, policy,start=(0,0),end=None):
    height, width = lake.shape
    if end==None:
        end =  (height-1,width-1)
    state = _state_from_readable(start[0],start[1],width)
    end_state = _state_from_readable(end[0],end[1],width)
    transitions = get_transitions_tensor(lake)
    count = 0
    while 1:
        count+=1
        next_state = np.argmax(transitions[:,state,policy[state]])
        if next_state==end_state:
            return count
        state=next_state