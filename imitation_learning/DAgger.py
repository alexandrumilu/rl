import tensorflow as tf
import numpy as np


class DataStore(object):
    def __init__(self, observations, actions):
        self._observations = observations
        self._actions = actions

    #         self.observation_space_dim = None
    #         self.action_space_dim = None

    def add_data(self, observations, actions):

        self._observations += observations
        self._actions += actions

    @property
    def actions(self):
        return np.array(self._actions)  

    @property
    def observations(self):
        return np.array(self._observations)  

class DAgger(object):
    def __init__(
            self,
            expert,
            num_epochs,
            num_sample_trajectories,
            expert_observations,
            expert_actions,
            policy,
            num_steps_gradient_policy,
            env,
            
    ):
        self.expert = expert
        self.num_epochs = num_epochs
        self.num_sample_trajectories = num_sample_trajectories
        self.num_steps_gradient_policy = num_steps_gradient_policy
        self.env = env
        
        
        self.data_store = DataStore(observations=expert_observations, actions=expert_actions)
        self.policy = policy
        
        self.sess = tf.Session()
        
        self.observation_space_size = len(self.env.reset())
        self.action_space_size = len(self.policy.action_space)
        
        #logging
        self.rewards = []

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.num_epochs):
            policy.train(self.data_store.observations,
                         self.data_store.actions,
                         self.num_steps_gradient_policy,
                         self.sess
                        )
            observations = self.sample_trajectories()
            actions = self.expert.best_action(observations) #just samples from what the expert would have done
            self.data_store.add_data(observations, actions)

    def sample_trajectories(self):
        observations = []
        for t in range(self.num_sample_trajectories):
            obs = self.env.reset()
            
            done = False
            total_reward = 0
            while not done:
                observations.append(obs)
                action = self.policy.sample_action(obs.reshape(len(obs),1),self.sess)
                obs, reward, done, info = env.step(action)
                total_reward+=reward
                
            self.rewards.append(total_reward)
        return observations
class Policy(object):
    def __init__(
        self,
        function,
        env
    ):
        tf.reset_default_graph()
        
        self.function = function
        self.env = env
        self.action_space = np.array(range(self.env.action_space.n))
        
        self.obs_ph = tf.placeholder(tf.float32)
        
        self.logits = self.function(self.obs_ph)
        self.action_distribution = tf.nn.softmax(self.logits, axis = 0)
        
    def action_distrib(self,observations,sess):
        return sess.run(self.action_distribution,feed_dict = {self.obs_ph:observations})
    
    def sample_action(self,observations,sess):
        action_dis = self.action_distrib(observations,sess)
        return np.random.choice(self.action_space,p = action_dis.reshape(len(self.action_space)))
class TrainablePolicyDAgger(Policy):
    def __init__(
        self,
        function,
        env,
        optimizer,
        batch_size
    ):
        super(TrainablePolicyDAgger, self).__init__(
            function, 
            env
        )
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.act_ph = tf.placeholder(tf.int32)
        
        self.one_hot_actions = tf.one_hot(self.act_ph, depth=len(self.action_space))
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels = self.one_hot_actions,
            logits = tf.transpose(self.logits)))
        
        self.update = self.optimizer.minimize(self.loss)
        
        #logging
        self.losses = []
    def train_one_step(self, observations, actions, sess):
        _,loss = sess.run([self.update,self.loss],feed_dict = {self.obs_ph: observations.T,self.act_ph: actions})
        self.losses.append(loss)
    def train(self,observations,actions,num_steps,sess):
        for i in range(num_steps):
            sample_indices = np.random.choice(observations.shape[0],self.batch_size)
            batch_obs = observations[sample_indices]
            batch_actions = actions[sample_indices]
            self.train_one_step(batch_obs,batch_actions,sess)
            

class FeedForwardNeuralNetwork(object):
    def __init__(
        self,
        input_ph,
        num_of_neurons_per_layer,
        scope_name,
        seed
    ):
        
        
        self.num_of_neurons_per_layer = num_of_neurons_per_layer
        self.scope_name = scope_name
        self.input_ph = input_ph
        self.seed = seed
        
        self.evaluate = self.output()
        
    def mlp_relu_layer(self,X,weight_shape,bias_shape,seed):
        W = tf.get_variable("W",shape=weight_shape,initializer=tf.contrib.layers.xavier_initializer(seed = seed))
        b = tf.get_variable("b",shape = bias_shape,initializer=tf.zeros_initializer())
        return tf.nn.relu(tf.matmul(W,X)+b)
    def mlp_no_activation_layer(self,X,weight_shape,bias_shape,seed):
        W = tf.get_variable("W",shape=weight_shape,initializer=tf.contrib.layers.xavier_initializer(seed = seed))
        b = tf.get_variable("b",shape = bias_shape,initializer=tf.zeros_initializer())
        return (tf.matmul(W,X)+b)
    def output(self):
        layer_input = self.input_ph
        for i in range(1,len(self.num_of_neurons_per_layer)):
            scope = self.scope_name+str(i)
            with tf.variable_scope(scope):
                if i==len(self.num_of_neurons_per_layer)-1:
                    layer_output = self.mlp_no_activation_layer(layer_input,
                                              weight_shape = (self.num_of_neurons_per_layer[i],self.num_of_neurons_per_layer[i-1]),
                                              bias_shape = (self.num_of_neurons_per_layer[i],1),
                                             seed = self.seed)
                else:
                    layer_output = self.mlp_relu_layer(layer_input,
                                              weight_shape = (self.num_of_neurons_per_layer[i],self.num_of_neurons_per_layer[i-1]),
                                              bias_shape = (self.num_of_neurons_per_layer[i],1),
                                             seed = self.seed)
            layer_input = layer_output
        return layer_output

class Expert(object):
    def __init__(
        self,
        agent,
        expert_type
    ):
        self.expert_type = expert_type
        self.agent = agent
        
    def best_action(self,observations):
        observations = np.array(observations).reshape(len(observations[0]),len(observations))
        action_distribution = self.agent.sess.run(self.agent.policy,feed_dict = {self.agent.state_placeholder: observations})
        best_action = np.argmax(action_distribution, axis = 0)
        return list(np.argmax(action_distribution, axis = 0))
    
class HumanExpertCartPole(Expert):
    def __init__(self):
        pass
    def best_action(self, observations):
        actions = []
        for obs in observations:
            if obs[2]>0:
                action =1
                if obs[3]<(-1.5):
                    action =0
            else:
                action =0
                if obs[3]>1.5:
                    action =1
            actions.append(action)
        return actions
    
        