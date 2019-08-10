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
    