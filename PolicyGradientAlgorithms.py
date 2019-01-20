
# coding: utf-8

# In[7]:

import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt





# In[8]:

def mlp_relu_layer(X,weight_shape,bias_shape,seed):
    W = tf.get_variable("W",shape=weight_shape,initializer=tf.contrib.layers.xavier_initializer(seed = seed))
    b = tf.get_variable("b",shape = bias_shape,initializer=tf.zeros_initializer())
    return tf.nn.relu(tf.matmul(W,X)+b)
def mlp_no_activation_layer(X,weight_shape,bias_shape,seed):
    W = tf.get_variable("W",shape=weight_shape,initializer=tf.contrib.layers.xavier_initializer(seed = seed))
    b = tf.get_variable("b",shape = bias_shape,initializer=tf.zeros_initializer())
    return (tf.matmul(W,X)+b)
def mlp(X,num_of_neurons_per_layer,scope_name):
    layer_input = X
    for i in range(1,len(num_of_neurons_per_layer)):
        scope = scope_name+str(i)
        with tf.variable_scope(scope):
            if i==len(num_of_neurons_per_layer)-1:
                layer_output = mlp_no_activation_layer(layer_input,
                                          weight_shape = (num_of_neurons_per_layer[i],num_of_neurons_per_layer[i-1]),
                                          bias_shape = (num_of_neurons_per_layer[i],1),
                                         seed = 3)
            else:
                layer_output = mlp_relu_layer(layer_input,
                                          weight_shape = (num_of_neurons_per_layer[i],num_of_neurons_per_layer[i-1]),
                                          bias_shape = (num_of_neurons_per_layer[i],1),
                                         seed = 3)
        layer_input = layer_output
    return layer_output


# In[14]:



class PolicyGradient(object):
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
    ):
        tf.reset_default_graph()
        
        # Set hyperparameters.
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        
        # Define environment.
        self.env = gym.make(env_name)
        self.action_space = np.array(range(self.env.action_space.n))
        
        # Build computation graph.
        self.optimizer = optimizer
        self.state_placeholder = state_placeholder = tf.placeholder(tf.float32)
        policy_generator_output = policy_generator(state_placeholder)
        self.actions_placeholder = actions_placeholder = tf.placeholder(tf.int32)
        self.weights_placeholder = weights_placeholder = tf.placeholder(tf.float32)
        
        # Define loss function.
        self.log_probability = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels = tf.one_hot(actions_placeholder,depth=len(self.action_space)),
            logits = tf.transpose(policy_generator_output)
        )
        self.loss = loss = tf.reduce_mean(weights_placeholder * self.log_probability)
        self.update = self.optimizer.minimize(loss)
        
        self.policy = tf.nn.softmax(policy_generator_output,axis = 0)
        
        # Instantiate the TF session.
        self.sess = tf.Session()
        
        # Initialize storage for logging.
        self.rewards = []
        
    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        for i in range(self.num_episodes):
            self.train_one_epoch()
        return
    
    def train_one_epoch(self):
        
        states,actions,rewards,state_is_terminal = self.collect_experience()
        weights = self.calculate_weights(rewards,state_is_terminal)
        
        _ = self.sess.run(self.update,feed_dict={
            self.state_placeholder:states, 
            self.actions_placeholder:actions,
            self.weights_placeholder:weights
        })
        
    def collect_experience(self):
        obs = self.env.reset()
        stopping = False
        observations = []
        actions = []
        rewards = []
        dones = []
        rewards_per_episode = []
        while not stopping:
            action_probs = self.sess.run(self.policy, feed_dict={self.state_placeholder: obs.reshape(obs.shape[0],1)})
            action = self.sample(action_probs)
            
            # Store values needed for policy update.
            observations.append(obs)
            actions.append(action)
            
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            dones.append(done)
            rewards_per_episode.append(reward)
            
            if done:
                obs = self.env.reset()
                self.rewards.append(sum(rewards_per_episode))
                rewards_per_episode = []
            
            stopping = done and len(observations) > self.batch_size
        
        observations_array = np.array(observations)
        actions_array = np.array(actions)
        
        
        
        return observations_array.T,actions_array,rewards,dones
    
    def calculate_weights(self, rewards,state_is_terminal_flag):
        #probably not the best way but this will do the job
        assert(len(rewards)==len(state_is_terminal_flag))
        weights = []
        last_index = 0
        for i in range(len(rewards)):
            if state_is_terminal_flag[i]:
                weights = weights +  [sum(rewards[last_index:i+1])]*(i+1-last_index)
                last_index = i+1
        weights_array = np.array(weights)
        weights_array = weights_array.reshape(weights_array.shape[0],1)
        return weights_array.T
    
    def sample(self, policy):
        return np.random.choice(self.action_space, p = policy.reshape(self.action_space.shape))


# In[10]:

class PolicyGradientRewardToGo(PolicyGradient):
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
    ):
        super(PolicyGradientRewardToGo, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env_name,
        )
        
    def calculate_weights(self, rewards,state_is_terminal_flag):
        last_index = 0
        weights = []
        for i in range(len(rewards)):
            if state_is_terminal_flag[i]:
                list_to_add = (list(np.cumsum(np.array(rewards[last_index:i+1]))[-1] - np.cumsum(np.array(rewards[last_index:i+1]))))
                weights+=list_to_add
                last_index = i+1
        weights_array = np.array(weights)
        weights_array = weights_array.reshape(weights_array.shape[0],1)
        return weights_array.T


# In[11]:

class PolicyGradientValue(PolicyGradient):
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
        value_optimizer,
        value_generator,
    ):
        super(PolicyGradientValue, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env_name,
        )
        self.value_optimizer = value_optimizer
        self.value_generator = value_generator
        #initialize what we need for value function update
        self.value_function = value_generator(self.state_placeholder)
        self.target_placeholder = target_placeholder = tf.placeholder(tf.float32)
        self.value_loss =value_loss= tf.reduce_sum((self.value_function-target_placeholder)**2)/2.0
        self.update_value = self.value_optimizer.minimize(value_loss,var_list = tf.trainable_variables(scope="value"))
        
        #change loss for 
        self.loss = tf.reduce_mean(self.log_probability*(self.weights_placeholder-self.value_function))
        self.update = self.optimizer.minimize(
            self.loss,var_list = tf.trainable_variables(scope="policy")) 
    def train_one_epoch(self):
        
        states,actions,rewards,state_is_terminal = self.collect_experience()
        weights = self.calculate_weights(rewards,state_is_terminal)
        
        _ = self.sess.run(self.update,feed_dict={
            self.state_placeholder:states, 
            self.actions_placeholder:actions,
            self.weights_placeholder:weights
        })
        
        _ = self.sess.run(self.update_value,feed_dict = {
            self.state_placeholder:states,
            self.target_placeholder:weights
        })
    def calculate_weights(self, rewards,state_is_terminal_flag):
        last_index = 0
        weights = []
        for i in range(len(rewards)):
            if state_is_terminal_flag[i]:
                list_to_add = (list(np.cumsum(np.array(rewards[last_index:i+1]))[-1] - np.cumsum(np.array(rewards[last_index:i+1]))))
                weights+=list_to_add
                last_index = i+1
        weights_array = np.array(weights)
        weights_array = weights_array.reshape(weights_array.shape[0],1)
        return weights_array.T


# In[306]:

class PolicyGradientBootStrap(PolicyGradient):
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
        value_optimizer,
        value_generator,
    ):
        super(PolicyGradientBootStrap, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env_name,
        )
        self.value_optimizer = value_optimizer
        self.value_generator = value_generator
        #initialize what we need for value function update
        self.value_function = value_generator(self.state_placeholder)
        self.target_placeholder = target_placeholder = tf.placeholder(tf.float32)
        self.value_loss =value_loss= tf.reduce_sum((self.value_function-target_placeholder)**2)/2.0
        self.update_value = self.value_optimizer.minimize(value_loss,var_list = tf.trainable_variables(scope="value"))
        
        self.loss = tf.reduce_mean(self.log_probability*(self.weights_placeholder-self.value_function))
        self.update = self.optimizer.minimize(
            self.loss,var_list = tf.trainable_variables(scope="policy")) 
    def train_one_epoch(self):
        
        states,actions,rewards,state_is_terminal = self.collect_experience()
        targets = self.calculate_targets(rewards,state_is_terminal,states)
        
        _ = self.sess.run(self.update,feed_dict={
            self.state_placeholder:states, 
            self.actions_placeholder:actions,
            self.weights_placeholder:targets
        })
        
        _ = self.sess.run(self.update_value,feed_dict = {
            self.state_placeholder:states,
            self.target_placeholder:targets
        })
    def calculate_targets(self,rewards,state_is_terminal_flag,states):
        #write it vectorized
        m = len(rewards)
        assert(m==states.shape[1])
        #add a next state array:
        next_states = np.concatenate([states[:,1:],np.array([[0],[0],[0],[0]])],axis = 1)
        #shape the rewards and terminal flags
        rewards_np = np.array(rewards).reshape(m,1)
        done_np = np.array(state_is_terminal_flag).reshape(m,1)
        values = self.sess.run(self.value_function,feed_dict = {self.state_placeholder:next_states})
        #print(values.shape)
        targets = rewards_np+(1-done_np)*values.T
        #print(targets.shape)
        return targets.T


# In[12]:

class PolicyGradientBootStrapAdvOnly(PolicyGradient):
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
        value_optimizer,
        value_generator,
    ):
        super(PolicyGradientBootStrapAdvOnly, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env_name,
        )
        self.value_optimizer = value_optimizer
        self.value_generator = value_generator
        #initialize what we need for value function update
        self.value_function = value_generator(self.state_placeholder)
        self.target_placeholder = target_placeholder = tf.placeholder(tf.float32)
        self.value_loss =value_loss= tf.reduce_sum((self.value_function-target_placeholder)**2)/2.0
        self.update_value = self.value_optimizer.minimize(value_loss,var_list = tf.trainable_variables(scope="value"))
        
        self.loss = tf.reduce_mean(self.log_probability*(self.weights_placeholder-self.value_function))
        self.update = self.optimizer.minimize(
            self.loss,var_list = tf.trainable_variables(scope="policy")) 
        self.losses = []
    def train_one_epoch(self):
        
        states,actions,rewards,state_is_terminal = self.collect_experience()
        
        weights = self.calculate_weights(rewards,state_is_terminal)
        #print("Weights")
        #print(weights)
        _,loss_v = self.sess.run([self.update_value,self.value_loss],feed_dict = {
            self.state_placeholder:states,
            self.target_placeholder:weights
        })
        self.losses.append(loss_v)
        targets = self.calculate_targets(rewards,state_is_terminal,states)
        #print(targets)
        _ = self.sess.run(self.update,feed_dict={
            self.state_placeholder:states, 
            self.actions_placeholder:actions,
            self.weights_placeholder:targets
        })
        
        
    def calculate_targets(self,rewards,state_is_terminal_flag,states):
        #write it vectorized
        m = len(rewards)
        assert(m==states.shape[1])
        #add a next state array:
        next_states = np.concatenate([states[:,1:],np.array([[0],[0],[0],[0]])],axis = 1)
        assert(m==next_states.shape[1])
        #shape the rewards and terminal flags
        rewards_np = np.array(rewards).reshape(m,1)
        done_np = np.array(state_is_terminal_flag).reshape(m,1)
        values = self.sess.run(self.value_function,feed_dict = {self.state_placeholder:next_states})
        #print(values.shape)
        targets = rewards_np+(1-done_np)*values.T
        #print(targets.shape)
        return targets.T
    def calculate_weights(self, rewards,state_is_terminal_flag):
        last_index = 0
        weights = []
        for i in range(len(rewards)):
            if state_is_terminal_flag[i]:
                list_to_add = (list(np.cumsum(np.array(rewards[last_index:i+1]))[-1] - np.cumsum(np.array(rewards[last_index:i+1]))))
                weights+=list_to_add
                last_index = i+1
        weights_array = np.array(weights)
        weights_array = weights_array.reshape(weights_array.shape[0],1)
        return weights_array.T


# In[13]:

class PolicyGradientGAE(PolicyGradient):
    #everything as above except calculate targets. 
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
        value_optimizer,
        value_generator,
        gamma,
        lamda
    ):
        super(PolicyGradientGAE, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env_name,
        )
        self.value_optimizer = value_optimizer
        self.value_generator = value_generator
        #initialize what we need for value function update
        self.value_function = value_generator(self.state_placeholder)
        self.target_placeholder = target_placeholder = tf.placeholder(tf.float32)
        self.value_loss =value_loss= tf.reduce_sum((self.value_function-target_placeholder)**2)/2.0
        self.update_value = self.value_optimizer.minimize(value_loss,var_list = tf.trainable_variables(scope="value"))
        
        self.loss = tf.reduce_mean(self.log_probability*(self.weights_placeholder))
        self.update = self.optimizer.minimize(
            self.loss,var_list = tf.trainable_variables(scope="policy")) 
        self.losses = []
        self.gamma = gamma
        self.lamda = lamda
    def train_one_epoch(self):
        
        states,actions,rewards,state_is_terminal = self.collect_experience()
        
        weights = self.calculate_discounted_rew_to_go(rewards,state_is_terminal,self.gamma)
        #print("Weights")
        #print(weights)
        _,loss_v = self.sess.run([self.update_value,self.value_loss],feed_dict = {
            self.state_placeholder:states,
            self.target_placeholder:weights
        })
        self.losses.append(loss_v)
        targets = self.calculate_GAE(rewards,state_is_terminal,states)
        #print(targets)
        _ = self.sess.run(self.update,feed_dict={
            self.state_placeholder:states, 
            self.actions_placeholder:actions,
            self.weights_placeholder:targets
        })
        
    #this is different than above.     
    def calculate_GAE(self,rewards,state_is_terminal_flag,states):
        #write it vectorized
        m = len(rewards)
        assert(m==states.shape[1])
        #add a next state array:
        next_states = np.concatenate([states,np.array([[0],[0],[0],[0]])],axis = 1)
        assert(m+1==next_states.shape[1])
        #shape the rewards and terminal flags
        rewards_np = np.array(rewards).reshape(m,1)
        done_np = np.array(state_is_terminal_flag).reshape(m,1)
        values = self.sess.run(self.value_function,feed_dict = {self.state_placeholder:next_states})
        #calculates TD of shape (m,1)
        td = rewards_np+(1-done_np)*self.gamma*values[:,1:].T - values[:,:m].T
        
        targets = self.calculate_discounted_rew_to_go(list(td.squeeze()),state_is_terminal_flag,self.gamma*self.lamda)
        #print(targets.shape)
        return targets
    def calculate_discounted_rew_to_go(self,rewards,terminal_flag,discount):
        last_index = 0
        result = np.zeros_like(rewards)
        for i in range(len(rewards)):
            if terminal_flag[i]:
                current_episode_rew = rewards[last_index:i+1]
                running_sum = 0.0
                for j in reversed(range(len(current_episode_rew))):
                    running_sum = running_sum*discount+current_episode_rew[j]
                    result[last_index+j] = running_sum
                last_index = i+1
        result = result.reshape(result.shape[0],1)
        return result.T


# In[14]:

class PPO(PolicyGradient):
    def __init__(
        self,
        num_episodes,
        batch_size,
        optimizer,
        policy_generator,
        env_name,
        value_optimizer,
        value_generator,
        gamma,
        lamda,
        num_gradient_steps_policy,
        epsilon
    ):
        super(PPO, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env_name,
        )
        self.value_optimizer = value_optimizer
        self.value_generator = value_generator
        self.gamma = gamma
        self.lamda = lamda
        self.num_gradient_steps_policy = num_gradient_steps_policy
        self.epsilon = epsilon
        #initialize what we need for value function update
        self.value_function = value_generator(self.state_placeholder)
        self.target_placeholder = target_placeholder = tf.placeholder(tf.float32)
        self.value_loss =value_loss= tf.reduce_sum((self.value_function-target_placeholder)**2)/2.0
        self.update_value = self.value_optimizer.minimize(value_loss,var_list = tf.trainable_variables(scope="value"))
        #create placeholder for old policy
        self.old_probabilities = old_probabilities = tf.placeholder(tf.float32)
        #calculate probabilities of actions taken
        self.probabilities = tf.reduce_sum(self.policy*tf.one_hot(self.actions_placeholder,depth =len(self.action_space),axis = 0),axis =0,keepdims = True)
        #create L-clip
        term1 = self.probabilities*self.weights_placeholder/old_probabilities
        term2 = self.weights_placeholder*(1+self.epsilon*tf.sign(self.weights_placeholder))
        minim = tf.minimum(term1,term2)
        self.loss = -tf.reduce_mean(minim)
        self.update = self.optimizer.minimize(
            self.loss,var_list = tf.trainable_variables(scope="policy")) 
        self.losses = []
        
    def train_one_epoch(self):
        
        states,actions,rewards,state_is_terminal = self.collect_experience()
        #print("Shape of one hot:")
        #print(self.sess.run(tf.one_hot(self.actions_placeholder,depth=len(self.action_space)),feed_dict ={self.actions_placeholder:actions}).shape)
        #print("Shape of policy:")
        #print(self.sess.run(self.policy,feed_dict = {self.state_placeholder:states}).shape)
        weights = self.calculate_discounted_rew_to_go(rewards,state_is_terminal,self.gamma)
        #print("Weights")
        #print(weights)
        _,loss_v = self.sess.run([self.update_value,self.value_loss],feed_dict = {
            self.state_placeholder:states,
            self.target_placeholder:weights
        })
        self.losses.append(loss_v)
        targets = self.calculate_GAE(rewards,state_is_terminal,states)
        old_probs = self.sess.run(self.probabilities,feed_dict = {
            self.state_placeholder:states,
            self.actions_placeholder:actions
        })
        for i in range(self.num_gradient_steps_policy):
            _ = self.sess.run(self.update,feed_dict={
                self.state_placeholder:states, 
                self.actions_placeholder:actions,
                self.weights_placeholder:targets,
                self.old_probabilities:old_probs
            })     
    def calculate_GAE(self,rewards,state_is_terminal_flag,states):
        #write it vectorized
        m = len(rewards)
        assert(m==states.shape[1])
        #add a next state array:
        next_states = np.concatenate([states,np.array([[0],[0],[0],[0]])],axis = 1)
        assert(m+1==next_states.shape[1])
        #shape the rewards and terminal flags
        rewards_np = np.array(rewards).reshape(m,1)
        done_np = np.array(state_is_terminal_flag).reshape(m,1)
        values = self.sess.run(self.value_function,feed_dict = {self.state_placeholder:next_states})
        #calculates TD of shape (m,1)
        td = rewards_np+(1-done_np)*self.gamma*values[:,1:].T - values[:,:m].T
        
        targets = self.calculate_discounted_rew_to_go(list(td.squeeze()),state_is_terminal_flag,self.gamma*self.lamda)
        #print(targets.shape)
        return targets
    def calculate_discounted_rew_to_go(self,rewards,terminal_flag,discount):
        last_index = 0
        result = np.zeros_like(rewards)
        for i in range(len(rewards)):
            if terminal_flag[i]:
                current_episode_rew = rewards[last_index:i+1]
                running_sum = 0.0
                for j in reversed(range(len(current_episode_rew))):
                    running_sum = running_sum*discount+current_episode_rew[j]
                    result[last_index+j] = running_sum
                last_index = i+1
        result = result.reshape(result.shape[0],1)
        return result.T



