import numpy as np
from policy_gradient_algorithms.base_policy_gradient_agent import PolicyGradient

class PolicyGradientRewardToGo(PolicyGradient):
    def __init__(
            self,
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
    ):
        super(PolicyGradientRewardToGo, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
        )

    def calculate_weights(self, rewards, state_is_terminal_flag):
        last_index = 0
        weights = []
        for i in range(len(rewards)):
            if state_is_terminal_flag[i]:
                list_to_add = (list(np.cumsum(np.array(rewards[last_index:i + 1]))[-1] - np.cumsum(
                    np.array(rewards[last_index:i + 1]))))
                weights += list_to_add
                last_index = i + 1
        weights_array = np.array(weights)
        weights_array = weights_array.reshape(weights_array.shape[0], 1)
        return weights_array.T
