
import numpy as np
import tensorflow as tf
import gym



class ReplayBuffer():
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.is_done = []
        
        self.index_to_remove = 0
    
    @property
    def is_full(self):
        return len(self.rewards) >= self.replay_buffer_size 
    
    def add_data(self, state, action, next_state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.is_done.append(done)
        
        if self.is_full:
            self.actions = np.array(self.actions)
            self.states = np.array(self.states)
            self.next_states = np.array(self.next_states)
            self.rewards = np.array(self.rewards)
            self.is_done = np.array(self.is_done)
        
    def update_data(self, state, action, next_state, reward, done):
        self.states[self.index_to_remove] = state
        self.actions[self.index_to_remove] = action
        self.next_states[self.index_to_remove] = next_state
        self.rewards[self.index_to_remove] = reward
        self.is_done[self.index_to_remove] = done
        self.index_to_remove = (self.index_to_remove + 1) % self.replay_buffer_size
        
    def sample(self, sample_size):
        '''
        sample_states, sample_next_states: (sample_size x feature_dims)
        sample_actions, sample_rewards, sample_is_done: (sample_size x ,)
        '''
        sample_indices = np.random.choice(self.replay_buffer_size, sample_size)
        sample_states = self.states[sample_indices]
        sample_actions = self.actions[sample_indices]
        sample_next_states = self.next_states[sample_indices]
        sample_rewards = self.rewards[sample_indices]
        sample_is_done = self.is_done[sample_indices]
        #print(sample_actions.shape)
        return sample_states, sample_actions, sample_next_states, sample_rewards, sample_is_done
        




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




class DQN(object):
    def __init__(
        self,
        env_name,
        num_episodes,
        gamma, #discount factor
        q_function,
        target_function, #TODO: Maybe it's better to replicate q-function's type inside __init__. 
        optimizer,
        replay_buffer_size,
        batch_size,
        epsilon,
        frequency_of_target_updates,
    ):
        
        self.env = gym.make(env_name)
        self.action_space_size = self.env.action_space.n
        tf.reset_default_graph()
        self.num_episodes = num_episodes
        self.gamma = tf.constant(gamma)
        self.epsilon = epsilon
        
        self.q_function = q_function
        self.target_function = target_function
        
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.frequency_of_target_updates = frequency_of_target_updates
        
        # Make the TF graph
        self.states_placeholder = tf.placeholder(name = "sph",dtype=tf.float32)
        self.actions_placeholder = tf.placeholder(name = 'aph',dtype=tf.int32)
        self.next_state_placeholder = tf.placeholder(name = 'nsph',dtype=tf.float32)
        self.episode_done_placeholder = tf.placeholder(name = 'dph',dtype=tf.float32)
        self.reward_placeholder = reward_placeholder = tf.placeholder(name = 'rph',dtype=tf.float32)
        
        # Policy is argmax(q_values)
        self.q_values = self.q_function(self.states_placeholder) # shape = (num_actions x num_examples)
        self.best_actions = tf.argmax(self.q_values, axis=0)
        
        self.next_state_q = self.target_function(self.next_state_placeholder)
        # min[R + gamma * max q(next_state) - q(state)]
        self.target_values = self.reward_placeholder + (
            (1 - self.episode_done_placeholder) * 
            self.gamma * 
            tf.reduce_max(self.next_state_q, axis=0, keepdims=True)
        )
        
        self.actions_one_hot = actions_one_hot = tf.one_hot(self.actions_placeholder, depth=self.action_space_size, axis = 0)
        self.q_value_of_actions_taken = q_value_of_actions_taken = tf.reduce_sum(self.q_values*actions_one_hot,axis=0, keepdims = True)

        self.loss = tf.reduce_mean((self.target_values -  q_value_of_actions_taken)**2)/(2.0) #they say huber loss works better
        
        #update q-value
        self.update = self.optimizer.minimize(self.loss, var_list = tf.trainable_variables(scope='q_value'))
        self.assign_ops = tf.group(*self.update_targets())
        
        #Monitoring stuff
        self.value_function = tf.reduce_max(self.q_values,axis = 0)
        
        self.value = []
        self.rewards = []
        self.q_value_first_state = []
        self.target_value_first = []
        self.target_value_last = []
        self.q_value_last_state = []
        self.losses = []
        self.sess = tf.Session()
        
    def epsilon_greedy_action_selection(self, state, epsilon=None):
        epsilon = epsilon or self.epsilon
        e = np.random.random_sample()    
        if e < epsilon:
            # Random action
            action = self.env.action_space.sample()
        else:
            # Policy_based action
            action = self.sess.run(
                self.best_actions, 
                feed_dict={self.states_placeholder: state.reshape(state.shape[0], 1)}
            )[0]
        return action

    def construct_replay_buffer(self):
        state = self.env.reset()
        while not self.replay_buffer.is_full:
            action = self.epsilon_greedy_action_selection(state, epsilon=1.0)
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.add_data(state, action, next_state, reward, done)
            state = next_state
            if done:
                state = self.env.reset()
                
    def update_targets(self):
        sorted_q_vals = sorted(tf.trainable_variables(scope='q_value'), key=lambda v: v.name)
        sorted_targets = sorted(tf.trainable_variables(scope='target_value'), key=lambda v: v.name)
        assign_ops = []
        for q_val, target in zip(sorted_q_vals, sorted_targets):
            assign_ops.append(target.assign(q_val))
        return assign_ops
            
    def train(self):
        self.construct_replay_buffer()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.assign_ops)
        actions_taken = 0

        for i in range(self.num_episodes):
            done = False
            state = self.env.reset()
            y = self.sess.run(self.q_values,feed_dict = {self.states_placeholder:state.reshape(4,1)})
            self.q_value_first_state.append(np.max(y))
            z  = self.sess.run(self.target_values,feed_dict = {self.next_state_placeholder:state.reshape(4,1),
                                                              self.reward_placeholder:np.array([1]).reshape(1,1),
                                                              self.episode_done_placeholder:np.array([False]).reshape(1,1)})
            self.target_value_first.append(np.max(z))
            reward_for_episode = 0
            render = False
            if i%100==0:
                render = True
                
            while not done:
                actions_taken += 1
                #if render:
                #    self.env.render()
                # Take an action, store values in replay buffer. 
                action = self.epsilon_greedy_action_selection(state,epsilon=self.epsilon)
                self.value.append(self.sess.run(self.value_function,feed_dict = {self.states_placeholder:state.reshape(4,1)})[0])
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.update_data(state, action, next_state, reward, done)
                if done:
                    self.q_value_last_state.append(self.sess.run(self.value_function,feed_dict = {self.states_placeholder:state.reshape(4,1)})[0])
                state = next_state
                
                # Sample from replay buffer. 
                (sample_states, 
                 sample_actions,
                 sample_next_states,
                 sample_rewards,
                 sample_is_done) = self.replay_buffer.sample(self.batch_size)
                
                # Update q-function once. 
                _,loss = self.sess.run([self.update,self.loss], feed_dict={
                    self.actions_placeholder:sample_actions,
                    self.states_placeholder:sample_states.T,
                    self.next_state_placeholder:sample_next_states.T,
                    self.episode_done_placeholder:sample_is_done.reshape(self.batch_size, 1).T,
                    self.reward_placeholder:sample_rewards.reshape(self.batch_size, 1).T, 
                })
                self.losses.append(loss)
                
                
                # Every n times, set target = q_values
                if actions_taken % self.frequency_of_target_updates == 0:
                    self.sess.run(self.assign_ops)
                    
                reward_for_episode += reward
            self.rewards.append(reward_for_episode)
    




class DDQN(DQN):
    def __init__(
        self,
        env_name,
        num_episodes,
        gamma, #discount factor
        q_function,
        target_function, #TODO: Maybe it's better to replicate q-function's type inside __init__. 
        optimizer,
        replay_buffer_size,
        batch_size,
        epsilon,
        frequency_of_target_updates,
    ):
        super(DDQN,self).__init__(
        env_name,
        num_episodes,
        gamma, 
        q_function,
        target_function, 
        optimizer,
        replay_buffer_size,
        batch_size,
        epsilon,
        frequency_of_target_updates,
        )
        self.target_placeholder = tf.placeholder(name = 'tph',dtype = tf.float32)
        self.q_targets_value = tf.reduce_sum(self.next_state_q*tf.one_hot(self.best_actions,depth = self.action_space_size,axis = 0),
                                           axis = 0,keepdims = True)
        self.target_values = self.reward_placeholder + (
            (1 - self.episode_done_placeholder) * self.gamma * self.q_targets_value
        )
        self.loss_ddqn = tf.reduce_mean((self.target_placeholder -  self.q_value_of_actions_taken)**2)/(2.0)
        self.update_ddqn = self.optimizer.minimize(self.loss_ddqn, var_list = tf.trainable_variables(scope='q_value'))
    def train(self,initialize_variables):
        self.construct_replay_buffer()
        if initialize_variables:
            self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.assign_ops)
        actions_taken = 0

        for i in range(self.num_episodes):
            done = False
            state = self.env.reset()
            y = self.sess.run(self.q_values,feed_dict = {self.states_placeholder:state.reshape(4,1)})
            self.q_value_first_state.append(np.max(y))
#             z  = self.sess.run(self.target_values,feed_dict = {self.next_state_placeholder:state.reshape(4,1),
#                                                               self.reward_placeholder:np.array([1]).reshape(1,1),
#                                                               self.episode_done_placeholder:np.array([False]).reshape(1,1)})
#             self.target_value_first.append(np.max(z))
            reward_for_episode = 0
            render = False
            if i%100==0:
                render = True
                
            while not done:
                actions_taken += 1
                #if render:
                #    self.env.render()
                # Take an action, store values in replay buffer. 
                action = self.epsilon_greedy_action_selection(state,epsilon=self.epsilon)
                self.value.append(self.sess.run(self.value_function,feed_dict = {self.states_placeholder:state.reshape(4,1)})[0])
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.update_data(state, action, next_state, reward, done)
                if done:
                    self.q_value_last_state.append(self.sess.run(self.value_function,feed_dict = {self.states_placeholder:state.reshape(4,1)})[0])
                state = next_state
                
                # Sample from replay buffer. 
                (sample_states, 
                 sample_actions,
                 sample_next_states,
                 sample_rewards,
                 sample_is_done) = self.replay_buffer.sample(self.batch_size)
                
                targets = self.sess.run(self.target_values, feed_dict={
                    self.states_placeholder:sample_next_states.T,
                    self.next_state_placeholder:sample_next_states.T,
                    self.reward_placeholder:sample_rewards.reshape(self.batch_size, 1).T,
                    self.episode_done_placeholder:sample_is_done.reshape(self.batch_size, 1).T
                })
                # Update q-function once. 
                _,loss = self.sess.run([self.update_ddqn,self.loss_ddqn], feed_dict={
                    self.actions_placeholder:sample_actions,
                    self.states_placeholder:sample_states.T,
                    self.target_placeholder: targets 
                })
                self.losses.append(loss)
                
                
                # Every n times, set target = q_values
                if actions_taken % self.frequency_of_target_updates == 0:
                    self.sess.run(self.assign_ops)
                    
                reward_for_episode += reward
            self.rewards.append(reward_for_episode)
            if i%1000==0:
                print("Average reward over the last 200 episodes was "+str(np.mean(self.rewards[-200:])))



