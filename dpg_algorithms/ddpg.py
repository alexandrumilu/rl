from neural_network.neural_network import FeedForwardNeuralNetwork
import tensorflow as tf
import numpy as np
from ReplayBuffer import ReplayBuffer

class DDPG(object):
    def __init__(
        self,
        env,
        batch_size,
        replay_buffer_size,
        num_episodes,
        gamma,
        q_function,
        mu_function,
        optimizer_q,
        optimizer_mu,
        polyak_averaging_constant
    ):
        tf.reset_default_graph()
        #Set hyperparameters
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = tf.constant(gamma)
        self.polyak = polyak_averaging_constant
        
        #Define environment
        self.env = env
        self.action_shape = self.env.action_space.sample().shape
        
        #Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        #Build computation graph

        self.optimizer_q = optimizer_q
        self.optimizer_mu = optimizer_mu

        # Create placeholders.
        self.states_placeholder = tf.placeholder(name="sph", dtype=tf.float32)
        self.actions_placeholder = tf.placeholder(name='aph', dtype=tf.float32)
        self.next_state_placeholder = tf.placeholder(name='nsph', dtype=tf.float32)
        self.episode_done_placeholder = tf.placeholder(name='dph', dtype=tf.float32)
        self.reward_placeholder = tf.placeholder(name='rph', dtype=tf.float32)
        
        self.mu_function = mu_function
        self.q_function = q_function
        
        self.best_action_unbounded = self.mu_function.get_output_layer(self.states_placeholder)
        self.min_action = self.env.action_space.low.reshape(len(self.env.action_space.low),1)
        self.max_action = self.env.action_space.high.reshape(len(self.env.action_space.low),1)
        self.best_action = self.min_action+(self.max_action-self.min_action)*tf.sigmoid(self.best_action_unbounded)
        with tf.variable_scope("qvalue"):
            self.q_value = self.q_function.get_output_layer(tf.concat([self.states_placeholder, self.best_action], axis = 0))
        with tf.variable_scope("qvalue",reuse = True):
            self.q_value_aph = self.q_function.get_output_layer(
                tf.concat([self.states_placeholder, self.actions_placeholder], axis = 0)
            )
        
        self.target_q_function = FeedForwardNeuralNetwork(
            num_of_neurons_per_layer=self.q_function.num_of_neurons_per_layer,
            scope_name = "target_q",
            seed = self.q_function.seed
        )
        self.target_mu_function = FeedForwardNeuralNetwork(
            num_of_neurons_per_layer=self.mu_function.num_of_neurons_per_layer,
            scope_name = "target_mu",
            seed = self.mu_function.seed
        )
        self.best_action_next_state_unbounded = self.target_mu_function.get_output_layer(self.next_state_placeholder)
        self.best_action_next_state = self.min_action+(self.max_action-self.min_action)*tf.sigmoid(self.best_action_next_state_unbounded)
        
        self.q_value_next_state = self.target_q_function.get_output_layer(
            tf.concat([self.next_state_placeholder,self.best_action_next_state], axis = 0))
        self.target_value = self.reward_placeholder + (
            (1-self.episode_done_placeholder)*
            self.gamma*
            self.q_value_next_state
        )
        self.loss_critic = tf.reduce_mean((self.target_value - self.q_value_aph)**2)
        self.loss_actor = tf.reduce_mean(-self.q_value)
        
        self.update_critic = self.optimizer_q.minimize(
            self.loss_critic,
            var_list=tf.trainable_variables(scope="qvalue")
        )
        
        self.update_actor = self.optimizer_mu.minimize(
            self.loss_actor,
            var_list = tf.trainable_variables(scope = self.mu_function.scope_name)
        )
        
        self.assign_ops = tf.group(*self.update_targets_polyak())

        # Instantiate the TF session.
        self.sess = tf.Session()
        
        #Logging
        self.rewards = []
        self.q_values = []
        self.loss_q = []
        self.actions = []
        self.best_actions = []
        
    def update_targets_polyak(self):
        sorted_q_vals = sorted(tf.trainable_variables(scope="qvalue"), key=lambda v: v.name)
        sorted_targets_q = sorted(tf.trainable_variables(scope=self.target_q_function.scope_name), key=lambda v: v.name)
        sorted_mu = sorted(tf.trainable_variables(scope = self.mu_function.scope_name), key = lambda v: v.name)
        sorted_targets_mu = sorted(tf.trainable_variables(scope = self.target_mu_function.scope_name), key = lambda v: v.name)
        
        assign_ops = []
        for q_val, target in zip(sorted_q_vals, sorted_targets_q):
            assign_ops.append(target.assign(self.polyak*target+(1-self.polyak)*q_val))
        for mu, target in zip(sorted_mu, sorted_targets_mu):
            assign_ops.append(target.assign(self.polyak*target+(1-self.polyak)*mu))
            
        return assign_ops
    
    def construct_replay_buffer(self):
        state = self.env.reset()
        while not self.replay_buffer.is_full:
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.add_data(state, action, next_state, reward, done)
            state = next_state
            if done:
                state = self.env.reset()
                
    def explore(self, state, std):
        best_action = self.sess.run(self.best_action, feed_dict = {self.states_placeholder: state.reshape(state.shape[0], 1)})
        self.best_actions.append(best_action)
        noise = np.random.normal(loc = 0, scale = std, size = best_action.shape)
        action = best_action + noise
        return action
    
    def train(self, 
              start_from_scratch=True, 
              num_episodes = None, 
              end_standard_div = 0.1, 
              render = False, 
              freq_of_rendering = 100, 
              start_standard_div = None):
        num_episodes = num_episodes or self.num_episodes
        #start_from_scratch is a bool specifying if variables, replay buffer should be initialized.
        if start_from_scratch:
            self.construct_replay_buffer()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.assign_ops)
        if not start_standard_div:
            start_standard_div = (self.max_action - self.min_action)/4
        render_this_episode = False
        for i in range(num_episodes):
            done = False
            state = self.env.reset()
            reward_for_episode = 0
            standard_div = ((num_episodes - i)*start_standard_div + i*end_standard_div)/num_episodes
            
            render_this_episode = False
            if render and i%freq_of_rendering ==0:
                render_this_episode = True
            
            while not done:
                if render_this_episode:
                    self.env.render()
                
                action = self.explore(state, standard_div)
                self.actions.append(action)
                action = action.flatten() #some env do not work if action not flat
                next_state, reward, done, info = self.env.step(action)
            
                next_state = next_state.reshape(next_state.shape[0],)#for some reason next_state some times is (m,1) and not (m,)
                action = action.reshape(action.shape[0],)
                self.replay_buffer.update_data(state, action, next_state, reward, done)
                state = next_state

                # Sample from replay buffer.
                (sample_states,
                 sample_actions,
                 sample_next_states,
                 sample_rewards,
                 sample_is_done) = self.replay_buffer.sample(self.batch_size)
                
                _, loss_q = self.sess.run([self.update_critic, self.loss_critic], feed_dict={
                    self.actions_placeholder: sample_actions.T,
                    self.states_placeholder: sample_states.T,
                    self.next_state_placeholder: sample_next_states.T,
                    self.episode_done_placeholder: sample_is_done.reshape(self.batch_size, 1).T,
                    self.reward_placeholder: sample_rewards.reshape(self.batch_size, 1).T,
                })
                
                _, loss_mu = self.sess.run([self.update_actor, self.loss_actor], feed_dict={
                    self.states_placeholder: sample_states.T
                })
                
                self.loss_q.append(loss_q)
        
                self.sess.run(self.assign_ops)
                
                
                reward_for_episode += reward

            self.rewards.append(reward_for_episode)
            
    
    