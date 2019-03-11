import tensorflow as tf
from policy_gradient_algorithms.policy_gradient_GAE import PolicyGradientGAE

class PPO(PolicyGradientGAE):
    def __init__(
            self,
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
            gamma,
            lamda,
            num_gradient_steps_policy,
            epsilon
    ):
        super(PPO, self).__init__(
            num_episodes,
            batch_size,
            optimizer,
            policy_generator,
            env,
            value_optimizer,
            value_generator,
            gamma,
            lamda,
        )
        self.num_gradient_steps_policy = num_gradient_steps_policy
        self.epsilon = epsilon
        # create placeholder for old policy
        self.old_probabilities = old_probabilities = tf.placeholder(tf.float32)
        # calculate probabilities of actions taken
        self.probabilities = tf.reduce_sum(
            self.policy * tf.one_hot(self.actions_placeholder, depth=len(self.action_space), axis=0), axis=0,
            keepdims=True)
        # create L-clip
        term1 = self.probabilities * self.weights_placeholder / old_probabilities
        term2 = self.weights_placeholder * (1 + self.epsilon * tf.sign(self.weights_placeholder))
        minim = tf.minimum(term1, term2)
        self.loss = -tf.reduce_mean(minim)
        self.update = self.optimizer.minimize(
            self.loss, var_list=tf.trainable_variables(scope=policy_generator.scope_name))
        self.losses = []

    def train_one_epoch(self):

        states, actions, rewards, state_is_terminal = self.collect_experience()
        weights = self.calculate_discounted_reward_to_go(rewards, state_is_terminal, self.gamma)
        _, loss_v = self.sess.run([self.update_value, self.value_loss], feed_dict={
            self.state_placeholder: states,
            self.target_placeholder: weights
        })
        self.losses.append(loss_v)
        targets = self.calculate_gae(rewards, state_is_terminal, states)
        old_probs = self.sess.run(self.probabilities, feed_dict={
            self.state_placeholder: states,
            self.actions_placeholder: actions
        })
        for _ in range(self.num_gradient_steps_policy):
            _ = self.sess.run(self.update, feed_dict={
                self.state_placeholder: states,
                self.actions_placeholder: actions,
                self.weights_placeholder: targets,
                self.old_probabilities: old_probs
            })


