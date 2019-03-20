import tensorflow as tf
import numpy as np
from ReplayBuffer import ReplayBuffer
from neural_network.neural_network import FeedForwardNeuralNetwork
from Policies import *
class DQN(object):
    def __init__(
            self,
            env,
            num_episodes,
            gamma,  # discount factor
            q_function,
            optimizer,
            replay_buffer_size,
            batch_size,
            epsilon,
            frequency_of_target_updates,
    ):
        tf.reset_default_graph()
        #Set hyperparameters.
        self.num_episodes = num_episodes
        self.gamma = tf.constant(gamma)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.frequency_of_target_updates = frequency_of_target_updates

        #Define environment
        self.env = env
        self.action_space_size = self.env.action_space.n

        #Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        #Initialize action chooser
        self.action_chooser = EpsilonGreedyDiscretePolicy(self.env,self.epsilon)

        #Build computation graph

        self.optimizer = optimizer

        # Create placeholders.
        self.states_placeholder = tf.placeholder(name="sph", dtype=tf.float32)
        self.actions_placeholder = tf.placeholder(name='aph', dtype=tf.int32)
        self.next_state_placeholder = tf.placeholder(name='nsph', dtype=tf.float32)
        self.episode_done_placeholder = tf.placeholder(name='dph', dtype=tf.float32)
        self.reward_placeholder = tf.placeholder(name='rph', dtype=tf.float32)

        self.q_function = q_function

        self.q_values = self.q_function.get_output_layer(self.states_placeholder)  # shape = (num_actions x num_examples)
        self.best_actions = tf.argmax(self.q_values, axis=0)
        self.value_function = tf.reduce_max(self.q_values, axis=0)

        self.target_function = FeedForwardNeuralNetwork(
            num_of_neurons_per_layer=self.q_function.num_of_neurons_per_layer,
            scope_name = "target",
            seed = self.q_function.seed
        )

        self.next_state_q = self.target_function.get_output_layer(self.next_state_placeholder)
        self.target_values = self.reward_placeholder + (
                (1 - self.episode_done_placeholder) *
                self.gamma *
                tf.reduce_max(self.next_state_q, axis=0, keepdims=True)
        )

        self.actions_one_hot = actions_one_hot = tf.one_hot(self.actions_placeholder, depth=self.action_space_size,
                                                            axis=0)
        self.q_value_of_actions_taken = q_value_of_actions_taken = tf.reduce_sum(self.q_values * actions_one_hot,
                                                                                 axis=0, keepdims=True)

        self.loss = tf.reduce_mean((self.target_values - q_value_of_actions_taken) ** 2) / (
            2.0)

        # update q-value
        self.update = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables(scope=self.q_function.scope_name))
        self.assign_ops = tf.group(*self.update_targets())

        # Instantiate the TF session.
        self.sess = tf.Session()


        # Logging
        self.value = []
        self.rewards = []
        self.losses = []

    def construct_replay_buffer(self):
        state = self.env.reset()
        random_policy = RandomDiscretePolicy(self.env)
        while not self.replay_buffer.is_full:
            action = random_policy.sample_action()
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.add_data(state, action, next_state, reward, done)
            state = next_state
            if done:
                state = self.env.reset()

    def update_targets(self):
        sorted_q_vals = sorted(tf.trainable_variables(scope=self.q_function.scope_name), key=lambda v: v.name)
        sorted_targets = sorted(tf.trainable_variables(scope=self.target_function.scope_name), key=lambda v: v.name)
        assign_ops = []
        for q_val, target in zip(sorted_q_vals, sorted_targets):
            assign_ops.append(target.assign(q_val))
        return assign_ops

    def train(self, start_from_scratch=True, num_episodes = None):
        num_episodes = num_episodes or self.num_episodes
        #start_from_scratch is a bool specifying if variables, replay buffer should be initialized.
        if start_from_scratch:
            self.construct_replay_buffer()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.assign_ops)

        actions_taken = 0

        for i in range(num_episodes):
            done = False
            state = self.env.reset()
            reward_for_episode = 0
            while not done:
                actions_taken += 1
                # Take an action, store values in replay buffer.
                action = self.action_chooser.sample_action(
                    action_distribution=self.sess.run(
                        self.q_values,
                        feed_dict = {self.states_placeholder: state.reshape(state.shape[0], 1)}
                    )
                )
                self.value.append(
                    self.sess.run(self.value_function, feed_dict={self.states_placeholder: state.reshape(state.shape[0], 1)})[0])
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.update_data(state, action, next_state, reward, done)
                state = next_state

                # Sample from replay buffer.
                (sample_states,
                 sample_actions,
                 sample_next_states,
                 sample_rewards,
                 sample_is_done) = self.replay_buffer.sample(self.batch_size)

                # Update q-function once.
                _, loss = self.sess.run([self.update, self.loss], feed_dict={
                    self.actions_placeholder: sample_actions,
                    self.states_placeholder: sample_states.T,
                    self.next_state_placeholder: sample_next_states.T,
                    self.episode_done_placeholder: sample_is_done.reshape(self.batch_size, 1).T,
                    self.reward_placeholder: sample_rewards.reshape(self.batch_size, 1).T,
                })
                self.losses.append(loss)

                # Every n times, set target = q_values
                if actions_taken % self.frequency_of_target_updates == 0:
                    self.sess.run(self.assign_ops)

                reward_for_episode += reward
            self.rewards.append(reward_for_episode)


