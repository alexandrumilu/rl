import tensorflow as tf
SEED = 0
DEFAULT_SCOPE_NAME = 'NN'

class FeedForwardNeuralNetwork(object):
    def __init__(
            self,
            num_of_neurons_per_layer,
            scope_name=DEFAULT_SCOPE_NAME,
            seed=SEED
    ):

        self.num_of_neurons_per_layer = num_of_neurons_per_layer
        self.scope_name = scope_name
        self.seed = seed

    def _mlp_relu_layer(self, X, weight_shape, bias_shape, seed):
        W = tf.get_variable("W", shape=weight_shape, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b = tf.get_variable("b", shape=bias_shape, initializer=tf.zeros_initializer())
        return tf.nn.relu(tf.matmul(W, X) + b)

    def _mlp_no_activation_layer(self, X, weight_shape, bias_shape, seed):
        W = tf.get_variable("W", shape=weight_shape, initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b = tf.get_variable("b", shape=bias_shape, initializer=tf.zeros_initializer())
        return (tf.matmul(W, X) + b)

    def get_output_layer(self, input_ph):
        layer_input = input_ph
        layer_output = None
        for i in range(1, len(self.num_of_neurons_per_layer)):
            scope = self.scope_name + str(i)
            with tf.variable_scope(scope):
                if i == len(self.num_of_neurons_per_layer) - 1:
                    layer_output = self._mlp_no_activation_layer(
                        X=layer_input,
                        weight_shape=(
                            self.num_of_neurons_per_layer[i],
                            self.num_of_neurons_per_layer[i - 1]
                        ),
                        bias_shape=(self.num_of_neurons_per_layer[i], 1),
                        seed=self.seed
                    )
                else:
                    layer_output = self._mlp_relu_layer(
                        X=layer_input,
                        weight_shape=(
                            self.num_of_neurons_per_layer[i],
                            self.num_of_neurons_per_layer[i - 1]
                        ),
                        bias_shape=(self.num_of_neurons_per_layer[i], 1),
                        seed=self.seed
                    )
            layer_input = layer_output
        return layer_output