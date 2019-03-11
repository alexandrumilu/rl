import numpy as np
from policy_gradient_algorithms.value_policy_gradient_agent import PolicyGradientValue 


class PolicyGradientBootStrap(PolicyGradientValue):
    def __init__(
            self,
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
    ):
        super(PolicyGradientBootStrap, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
        )

    def train_one_epoch(self):
        states, actions, rewards, state_is_terminal = self.collect_experience()
        targets = self.calculate_targets(rewards, state_is_terminal, states)

        _ = self.sess.run(self.update, feed_dict={
            self.state_placeholder: states,
            self.actions_placeholder: actions,
            self.weights_placeholder: targets
        })

        _ = self.sess.run(self.update_value, feed_dict={
            self.state_placeholder: states,
            self.target_placeholder: targets
        })

    def calculate_targets(self, rewards, state_is_terminal_flag, states):
        # write it vectorized
        m = len(rewards)
        assert (m == states.shape[1])
        # add a next state array:
        next_states = np.concatenate([states[:, 1:], np.array([[0], [0], [0], [0]])], axis=1)
        # shape the rewards and terminal flags
        rewards_np = np.array(rewards).reshape(m, 1)
        done_np = np.array(state_is_terminal_flag).reshape(m, 1)
        values = self.sess.run(self.value_function, feed_dict={self.state_placeholder: next_states})
        # print(values.shape)
        targets = rewards_np + (1 - done_np) * values.T
        # print(targets.shape)
        return targets.T