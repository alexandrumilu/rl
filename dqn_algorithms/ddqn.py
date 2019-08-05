from dqn_algorithms.dqn import DQN
import tensorflow as tf
import numpy as np
class DDQN(DQN):
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
        super(DDQN, self).__init__(
            env,
            num_episodes,
            gamma,
            q_function,
            optimizer,
            replay_buffer_size,
            batch_size,
            epsilon,
            frequency_of_target_updates,
        )
        self.target_placeholder = tf.placeholder(name='tph', dtype=tf.float32)
        self.q_targets_value = tf.reduce_sum(
            self.next_state_q * tf.one_hot(self.best_actions, depth=self.action_space_size, axis=0),
            axis=0, keepdims=True)
        self.target_values = self.reward_placeholder + (
                (1 - self.episode_done_placeholder) * self.gamma * self.q_targets_value
        )
        self.loss_ddqn = tf.reduce_mean((self.target_placeholder - self.q_value_of_actions_taken) ** 2) / (2.0)
        self.update_ddqn = self.optimizer.minimize(
            self.loss_ddqn,
            var_list=tf.trainable_variables(scope=self.q_function.scope_name)
        )

    def train(self, start_from_scratch=True, num_episodes=None):
        num_episodes = num_episodes or self.num_episodes
        # start_from_scratch is a bool specifying if variables, replay buffer should be initialized.
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

                action = self.action_chooser.sample_action(
                    action_distribution=self.sess.run(
                        self.q_values,
                        feed_dict={self.states_placeholder: state.reshape(state.shape[0], 1)}
                    )
                )

                
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.update_data(state, action, next_state, reward, done)
                state = next_state

                # Sample from replay buffer.
                (sample_states,
                 sample_actions,
                 sample_next_states,
                 sample_rewards,
                 sample_is_done) = self.replay_buffer.sample(self.batch_size)

                targets = self.sess.run(self.target_values, feed_dict={
                    self.states_placeholder: sample_next_states.T,
                    self.next_state_placeholder: sample_next_states.T,
                    self.reward_placeholder: sample_rewards.reshape(self.batch_size, 1).T,
                    self.episode_done_placeholder: sample_is_done.reshape(self.batch_size, 1).T
                })
                # Update q-function once.
                _, loss = self.sess.run([self.update_ddqn, self.loss_ddqn], feed_dict={
                    self.actions_placeholder: sample_actions,
                    self.states_placeholder: sample_states.T,
                    self.target_placeholder: targets
                })
                self.losses.append(loss)

                # Every n times, set target = q_values
                if actions_taken % self.frequency_of_target_updates == 0:
                    self.sess.run(self.assign_ops)

                reward_for_episode += reward
            self.rewards.append(reward_for_episode)
