import tensorflow as tf
from policy_gradient_algorithms.value_policy_gradient_agent import PolicyGradientValue
import numpy as np

class PolicyGradientGAE(PolicyGradientValue):
    def __init__(
            self,
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
            number_of_gradient_steps_value,
            gamma,
            lamda
    ):
        super(PolicyGradientGAE, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
            number_of_gradient_steps_value
        )

        self.loss = tf.reduce_mean(self.log_probability * self.weights_placeholder)
        self.update = self.optimizer.minimize(
            self.loss, var_list=tf.trainable_variables(scope=policy_generator.scope_name))
        self.gamma = gamma
        self.lamda = lamda

    def train_one_epoch(self):

        states, actions, rewards, state_is_terminal = self.collect_experience()

        weights = self.calculate_discounted_reward_to_go(rewards, state_is_terminal, self.gamma)
        for _ in range(self.number_of_gradient_steps_value):
            _, loss_v = self.sess.run([self.update_value, self.value_loss], feed_dict={
                self.state_placeholder: states,
                self.target_placeholder: weights
            })
            self.losses.append(loss_v)

        targets = self.calculate_gae(rewards, state_is_terminal, states)
        _ = self.sess.run(self.update, feed_dict={
            self.state_placeholder: states,
            self.actions_placeholder: actions,
            self.weights_placeholder: targets
        })

    def calculate_gae(self, rewards, state_is_terminal_flag, states):
        m = len(rewards)
        assert (m == states.shape[1])
        # add a next state array:
        next_states = np.concatenate([states, np.array([[0], [0], [0], [0]])], axis=1)
        assert (m + 1 == next_states.shape[1])
        # shape the rewards and terminal flags
        rewards_np = np.array(rewards).reshape(m, 1)
        done_np = np.array(state_is_terminal_flag).reshape(m, 1)
        values = self.sess.run(self.value_function, feed_dict={self.state_placeholder: next_states})
        # calculates TD of shape (m,1)
        td = rewards_np + (1 - done_np) * self.gamma * values[:, 1:].T - values[:, :m].T

        targets = self.calculate_discounted_reward_to_go(list(td.squeeze()), state_is_terminal_flag,
                                                      self.gamma * self.lamda)
        return targets

    def calculate_discounted_reward_to_go(self, rewards, terminal_flag, discount):
        last_index = 0
        result = np.zeros_like(rewards)
        for i in range(len(rewards)):
            if terminal_flag[i]:
                current_episode_rew = rewards[last_index:i + 1]
                running_sum = 0.0
                for j in reversed(range(len(current_episode_rew))):
                    running_sum = running_sum * discount + current_episode_rew[j]
                    result[last_index + j] = running_sum
                last_index = i + 1
        result = result.reshape(result.shape[0], 1)
        return result.T
