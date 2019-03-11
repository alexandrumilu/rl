import tensorflow as tf
from policy_gradient_algorithms.policy_gradient_reward_to_go_agent import PolicyGradientRewardToGo

class PolicyGradientValue(PolicyGradientRewardToGo):
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
        super(PolicyGradientValue, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
        )
        self.value_optimizer = value_optimizer
        self.value_generator = value_generator

        # initialize what we need for value function update
        self.value_function = value_generator.get_output_layer(self.state_placeholder)
        self.target_placeholder = target_placeholder = tf.placeholder(tf.float32)
        self.value_loss = value_loss = tf.reduce_sum((self.value_function - target_placeholder) ** 2) / 2.0
        self.update_value = self.value_optimizer.minimize(value_loss, var_list=tf.trainable_variables(scope=value_generator.scope_name))

        # change loss for
        self.loss = tf.reduce_mean(self.log_probability * (self.weights_placeholder - self.value_function))
        self.update = self.optimizer.minimize(
            self.loss,
            var_list=tf.trainable_variables(scope=policy_generator.scope_name)
        )

    def train_one_epoch(self):
        states, actions, rewards, state_is_terminal = self.collect_experience()
        weights = self.calculate_weights(rewards, state_is_terminal)

        _ = self.sess.run(self.update, feed_dict={
            self.state_placeholder: states,
            self.actions_placeholder: actions,
            self.weights_placeholder: weights
        })

        _ = self.sess.run(self.update_value, feed_dict={
            self.state_placeholder: states,
            self.target_placeholder: weights
        })
