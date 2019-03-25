from policy_gradient_algorithms.bootstrap_policy_gradient_agent import PolicyGradientBootStrap

class PolicyGradientBootStrapAdvOnly(PolicyGradientBootStrap):
    def __init__(
            self,
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
            number_of_gradient_steps_value
    ):
        super(PolicyGradientBootStrapAdvOnly, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
            number_of_gradient_steps_value
        )

    def train_one_epoch(self):
        states, actions, rewards, state_is_terminal = self.collect_experience()

        weights = self.calculate_weights(rewards, state_is_terminal)
        for _ in range(self.number_of_gradient_steps_value):
            _, loss_v = self.sess.run([self.update_value, self.value_loss], feed_dict={
                self.state_placeholder: states,
                self.target_placeholder: weights
            })
            self.losses.append(loss_v)

        targets = self.calculate_targets(rewards, state_is_terminal, states)

        _ = self.sess.run(self.update, feed_dict={
            self.state_placeholder: states,
            self.actions_placeholder: actions,
            self.weights_placeholder: targets
        })
