import tensorflow as tf
import numpy as np


class DataStore(object):
    def __init__(self, observations, actions):
        self._observations = observations
        self._actions = actions

    #         self.observation_space_dim = None
    #         self.action_space_dim = None

    def add_data(self, observations, actions):
        #         if not self.action_space_dim:
        #             self.action_space_dim = (
        #                 len(actions[0])
        #                 if isinstance(actions[0], list) | isinstance(actions[0], np.ndarray)
        #                 else 1
        #             )
        #         if not self.observation_space_dim:
        #             self.observation_space_dim = (
        #                 len(observations[0])
        #                 if isinstance(observations[0], list) | isinstance(observations[0], np.ndarray)
        #                 else 1
        #             )

        self._observations += observations
        self._actions += actions

    @property
    def actions(self):
        return np.array(self._actions)  # .reshape(len(self._actions), self.action_space_dim)

    @property
    def observations(self):
        return np.array(self._observations)  # .reshape(len(self._observations), self.observation_space_dim)

class DAgger(object):
    def __init__(
            self,
            expert,
            num_epochs,
            num_sample_trajectories,
            expert_observations,
            expert_actions,
            policy
    ):
        self.expert = expert
        self.num_epochs = num_epochs
        self.num_sample_trajectories = num_sample_trajectories

        self.data_store = DataStore(observations=expert_observations, actions=expert_actions)
        self.policy = policy
        self.obs_placeholder = tf.placeholder(tf.float32)

        self.action_distribution = self.policy(self.obs_placeholder)

        self.sess = tf.Session()

    def train(self):
        for epoch in range(self.num_epochs):
            # TODO: We need to give it a policy.
            self.fit_policy()
            observations = self.sample_trajectories()
            actions = self.expert.act(observations)
            self.data_store.add_data(observations, actions)

    def sample_trajectories(self):
        observations = []
        for t in range(self.num_sample_trajectories):
            obs = env.reset()
            observations.append(obs)
            done = False
            while not done:
                action = self.sample(self.sess.run(
                    self.action_distribution,
                    feed_dict={self.obs_placeholder: obs.reshape(obs.shape[0], 1)}
                ))
                obs, reward, done, info = env.step(action)
                observations.append(obs)
        return observations

    def fit_policy(self):
        raise NotImplementedError()
