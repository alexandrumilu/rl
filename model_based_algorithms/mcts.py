import numpy as np

class MCTSNode(object):
    def __init__(
        self,
        parent = None,
        value = 0,
        count = 0,
    ):
        self.children = []
        self.parent = parent
        self.value = 0
        self.count = count
    
    def is_root(self):
        return (self.parent == None)
    
    def update_value(self, new_value):
        self.value+=new_value
        self.count+=1
        
    def update_parent(self, parent):
        self.parent = parent
        
    def add_child(self, new_child):
        self.children.append(new_child)
        
class MCTS_agent(object):
    def __init__(self, policy, env, c_parameter, max_iterations):
        self.policy = policy
        self.env = env
        self.c = c_parameter
        self.max_iterations = max_iterations
        
        self.action_space = np.array(range(self.env.action_space.n))
        
    def uct_search(self, state):
        root = MCTSNode()
        done = False
        iteration = 0
        
        while iteration<self.max_iterations:
            self.env.reset()
            self.env.env.state = state
            node, new_state, value_so_far, done = self.tree_policy(root, state)
            if not done:
                value = value_so_far + self.run_default_policy(new_state)
            else:
                value = value_so_far
            self.update_parents(node,value)
            iteration+=1
            
        return self.best_child(root)[1]  
    
    def update_parents(self, node, value):
        while node:
            node.update_value(value)
            node = node.parent

    def run_default_policy(self,initial_state):
        self.env.env.state = initial_state
        obs = initial_state
        done = False
        total_reward = 0
        while not done:
            action = self.policy.sample_action(obs.reshape(len(obs),1)) 
            obs, reward, done, info = self.env.step(action)
            total_reward+=reward
        return total_reward
    
    def best_child(self, node):
        #might need a way to break ties that is different than by index
        max_score = self.compute_score_uct(node, node.children[0])
        child_with_max_score = node.children[0]
        action_with_max_score = 0
        for i in range(1,len(node.children)):
            score = self.compute_score_uct(node, node.children[i])
            if score>max_score:
                max_score = score
                child_with_max_score = node.children[i]    #if actions to chidlren map is trivial
                action_with_max_score = i
                
        return child_with_max_score, action_with_max_score
                        
    def compute_score_uct(self, node, child):
        if node.count==0:
            print(child.count)
        score = child.value/child.count + self.c*np.sqrt(2*np.log(node.count)/child.count)
        return score
    
    def expand(self, node):
        action = len(node.children)
        node.add_child(MCTSNode())
        child = node.children[action]
        child.update_parent(node)
        return child, action
     
    def tree_policy(self, root, state):
        self.env.env.state = state
        total_reward = 0
        done = False
        expanded = False
        node = root
        while not done:
            if len(node.children)<len(self.action_space):
                node, action = self.expand(node)
                expanded = True
            else:
                node, action = self.best_child(node)
            state, reward, done, info = self.env.step(action)
            total_reward+=reward
            if expanded:
                return node, state, total_reward, done
           
        return node, state, total_reward, done
        
    def select_action(self, state):
        return self.uct_search(state)