import tensorflow as tf
import gym

from Policies import *
from neural_network.neural_network import FeedForwardNeuralNetwork

class DataStore(object):
    def __init__(self, observations, actions):
        self._observations = observations
        self._actions = actions

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
            optimizer,
            batch_size,
            policy_function,
    ):
        #### Define Hyperparameters ####
        self.num_epochs = num_epochs
        self.num_sample_trajectories = num_sample_trajectories
        self.num_steps_gradient_policy = num_steps_gradient_policy
        self.optimizer = optimizer
        self.batch_size = batch_size

        #### Define Expert and Environment ####
        self.expert = expert
        self.env = env
        self.action_space = np.array(range(self.env.action_space.n))
        self.observation_space_size = len(self.env.reset())
        self.action_space_size = len(self.policy.action_space)

        self.data_store = DataStore(observations=expert_observations, actions=expert_actions)
        self.policy = policy

        #### Build TF graph ####
        tf.reset_default_graph()
        self.obs_ph = tf.placeholder(tf.float32)
        self.act_ph = tf.placeholder(tf.int32)
        self.one_hot_actions = tf.one_hot(self.act_ph, depth=len(self.action_space))
        self.logits = policy_function.get_output_layer(input_ph=self.obs_ph)
        self.action_distribution = tf.nn.softmax(self.logits, axis=0)
        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.one_hot_actions,
            logits=tf.transpose(self.logits))
        )
        self.update = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()

        #### Logging ####
        self.rewards = []
        self.losses = []

    def train_one_step(self, observations, actions):
        _,loss = self.sess.run([self.update,self.loss],feed_dict = {self.obs_ph: observations.T,self.act_ph: actions})
        self.losses.append(loss)

    def fit_to_data(self,observations,actions,num_steps):
        for i in range(num_steps):
            sample_indices = np.random.choice(observations.shape[0],self.batch_size)
            batch_obs = observations[sample_indices]
            batch_actions = actions[sample_indices]
            self.train_one_step(batch_obs,batch_actions)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.num_epochs):
            self.fit_to_data(self.data_store.observations,
                         self.data_store.actions,
                         self.num_steps_gradient_policy,
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
                action_distribution = self.sess.run(
                    self.action_distribution,
                    feed_dict={self.obs_ph: obs.reshape(len(obs),1)}
                )
                action = self.policy.sample_action(np.squeeze(action_distribution))
                obs, reward, done, info = self.env.step(action)
                total_reward+=reward
            self.rewards.append(total_reward)
        return observations


class Expert(object):
    def __init__(
        self,
        agent,
        expert_type
    ):
        self.expert_type = expert_type
        self.agent = agent
        
    def best_action(self, observations):
        observations = np.array(observations).reshape(len(observations[0]),len(observations))
        action_distribution = self.agent.sess.run(self.agent.policy,feed_dict = {self.agent.state_placeholder: observations})
        return list(np.argmax(action_distribution, axis = 0))
    
class HumanExpertCartPole(Expert):
    def __init__(self):
        super(HumanExpertCartPole, self).__init__(
            agent=None,
            expert_type='Human Expert'
        )

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

def main():
    env = gym.make('CartPole-v0')

    dagger = DAgger(
            expert=HumanExpertCartPole(),
            num_epochs=50,
            num_sample_trajectories=10,
            expert_observations=[env.reset()],
            expert_actions=[1],
            policy=ProbabilisticDiscretePolicy(env),
            num_steps_gradient_policy=100,
            env=env,
            optimizer=tf.train.AdamOptimizer(),
            batch_size=64,
            policy_function=FeedForwardNeuralNetwork(
                num_of_neurons_per_layer=[4, 16, 16, 2],
                scope_name='DAgger'
            ),
    )

    dagger.train()

if __name__ == '__main__':
    main()

    
        