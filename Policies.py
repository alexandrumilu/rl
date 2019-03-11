import numpy as np


class DiscretePolicy(object):
    def __init__(self, env):
        self.env = env

    def sample_action(self, action_distribution):
        raise NotImplementedError()


class RandomDiscretePolicy(DiscretePolicy):
    def __init__(self, env):
        super(RandomDiscretePolicy, self).__init__(env)

    def sample_action(self, action_distribution=None):
        return self.env.action_space.sample()


class MaxDiscretePolicy(DiscretePolicy):
    def __init__(self, env):
        super(MaxDiscretePolicy, self).__init__(env)

    def sample_action(self, action_distribution):
        return np.argmax(action_distribution)


class ProbabilisticDiscretePolicy(DiscretePolicy):
    def __init__(self, env):
        super(ProbabilisticDiscretePolicy, self).__init__(env)
        self.action_space = np.array(range(self.env.action_space.n))

    def sample_action(self, action_distribution):
        action_distribution = action_distribution.reshape(self.action_space.shape)
        return np.random.choice(a=self.action_space, p=action_distribution)


class EpsilonGreedyDiscretePolicy(DiscretePolicy):
    def __init__(self, env, epsilon):
        super(EpsilonGreedyDiscretePolicy, self).__init__(env)
        self.epsilon = epsilon
        self.random_policy = RandomDiscretePolicy(self.env)
        self.non_random_policy = MaxDiscretePolicy(self.env)

    def sample_action(self, action_distribution):
        random_number = np.random.uniform()
        if random_number > self.epsilon:
            return self.non_random_policy.sample_action(action_distribution)
        else:
            return self.random_policy.sample_action()
