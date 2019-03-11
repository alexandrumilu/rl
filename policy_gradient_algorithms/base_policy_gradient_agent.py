import numpy as np
import tensorflow as tf
from Policies import *

class PolicyGradient(object):
    def __init__(
            self,
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
    ):

        # Set hyperparameters.
        self.num_episodes = num_episodes
        self.batch_size = batch_size

        # Define environment.
        self.env = env
        self.action_space = np.array(range(self.env.action_space.n))

        # Build computation graph.
        tf.reset_default_graph()
        self.optimizer = optimizer
        self.state_placeholder = state_placeholder = tf.placeholder(tf.float32)

        policy_generator_output = policy_generator.get_output_layer(state_placeholder)
        self.actions_placeholder = actions_placeholder = tf.placeholder(tf.int32)
        self.weights_placeholder = weights_placeholder = tf.placeholder(tf.float32)

        # Define loss function.
        self.log_probability = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.one_hot(actions_placeholder, depth=len(self.action_space)),
            logits=tf.transpose(policy_generator_output)
        )
        self.loss = loss = tf.reduce_mean(weights_placeholder * self.log_probability)
        self.update = self.optimizer.minimize(loss)

        self.policy = tf.nn.softmax(policy_generator_output, axis=0)
        self.action_chooser = ProbabilisticDiscretePolicy(env)
        # Instantiate the TF session.
        self.sess = tf.Session()

        # Initialize storage for logging.
        self.rewards = []
        self.losses = []

    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        for i in range(self.num_episodes):
            self.train_one_epoch()
        return

    def train_one_epoch(self):

        states, actions, rewards, state_is_terminal = self.collect_experience()
        weights = self.calculate_weights(rewards, state_is_terminal)

        _ = self.sess.run(self.update, feed_dict={
            self.state_placeholder: states,
            self.actions_placeholder: actions,
            self.weights_placeholder: weights
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
            action_probs = self.sess.run(self.policy, feed_dict={self.state_placeholder: obs.reshape(obs.shape[0], 1)})
            action = self.action_chooser.sample_action(action_distribution=action_probs)

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

        return observations_array.T, actions_array, rewards, dones

    def calculate_weights(self, rewards, state_is_terminal_flag):
        # probably not the best way but this will do the job
        assert (len(rewards) == len(state_is_terminal_flag))
        weights = []
        last_index = 0
        for i in range(len(rewards)):
            if state_is_terminal_flag[i]:
                weights = weights + [sum(rewards[last_index:i + 1])] * (i + 1 - last_index)
                last_index = i + 1
        weights_array = np.array(weights)
        weights_array = weights_array.reshape(weights_array.shape[0], 1)
        return weights_array.T
