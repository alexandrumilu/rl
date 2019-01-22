'''
A set of environment classes for testing algorithm implementations.
'''
import numpy as np


class Environment(object):
    def action_space(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

class ActionSpace(object):
    def __init__(self, actions_list):
        self.actions = actions_list #TODO: What other properties are used?

    @property
    def n(self):
        return len(self.actions)

    def get(self, idx):
        return self.actions[idx]

class GaussianDistribution(object):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.stddev)


class MultiArmedBanditEnv(Environment):
    '''
    Implementation of a Multi Armed Bandit environment.
    Methods are compatible with most used methods of OpenAI gym environments.
    '''

    def __init__(
        self,
        reward_distributions,
    ):
        self.num_levers = len(reward_distributions)
        self.reward_distributions = reward_distributions
        self._action_space = ActionSpace(actions_list=range(self.num_levers))

    def action_space(self):
        return self._action_space

    def reset(self):
        return

    def step(self, action):
        reward_distribution = self.reward_distributions[action]

        reward = reward_distribution.sample()
        observation = None
        done = True
        info = {}

        return observation, reward, done, info

    @classmethod
    def get_stationary_gaussian_testbed(cls, num_levers):
        '''
        Returns a MultiArmedBandit test bed with n levers.
        Means of reward distributions - [q_n] - are selected from Gaussian dist. with mean=0 and unit variance.
        Then reward distribution of n levers are Gaussian with mean = q_n and unit variance.
        :param num_levers: int
        :return: MultiArmedBandit instance
        '''


        means_distribution = GaussianDistribution(mean=0, stddev=1)
        means = [means_distribution.sample() for i in range(num_levers)]
        rewards_distributions = [
            GaussianDistribution(mean=means[i], stddev=1.0)
            for i in range(num_levers)
        ]

        return MultiArmedBanditEnv(rewards_distributions)

    @classmethod
    def get_nonstationary_gaussian_testbed(cls, num_levers, reward_decay):
        raise NotImplementedError()
