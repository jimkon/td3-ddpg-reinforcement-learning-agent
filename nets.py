import numpy as np
import tensorflow as tf


def nn_layer(x, size, activation=tf.nn.relu, drop_out=0.3, use_bias=True, return_vars=True):
    # x*W+b
    if drop_out:
        x = tf.nn.dropout(x, rate=drop_out)

    W = tf.Variable(np.random.random((x.shape[1], size)) * (1. / (int(x.shape[1]) * size)))

    if use_bias:
        b = tf.Variable(np.random.random((1, size)) * (1. / size))
        line = tf.matmul(x, W) + b
    else:
        b = None
        line = tf.matmul(x, W)

    if activation is None:
        y = line
    else:
        y = activation(line)

    if return_vars:
        return y, W, b
    else:
        return y


class FullyConnectedDNN:

    def __init__(self, input_dims, output_dims, hidden_layers=(200, 100), activations=(tf.nn.relu, tf.nn.relu), use_biases=(True, True),
                 drop_out=.3, output_activation=None, output_use_bias=False, x=None, lr=1e-2):

        self.init_args = locals()
        del self.init_args['self']

        self.input_dims = input_dims
        self.output_dims = output_dims

        self.input_shape = tuple([self.input_dims])
        self.output_shape = tuple([self.output_dims])

        layers = np.append(hidden_layers, output_dims).astype(np.int) if hidden_layers is not None else np.array([output_dims])
        all_activations = list(activations) if activations is not None else []
        all_activations.append(output_activation)
        all_use_biases = list(use_biases) if use_biases is not None else []
        all_use_biases.append(output_use_bias)

        print("NN: layers:{}, activations:{}".format(layers, all_activations, all_use_biases))

        self.ys = []

        self.weights_and_biases = {}

        # tf.compat.v1.reset_default_graph()
        self.x = tf.compat.v1.placeholder(tf.float64, shape=(None, input_dims))
        x = self.x if x is None else x
        for i, layer in enumerate(layers):
            y, W, b = nn_layer(x, layer, all_activations[i], drop_out=drop_out if i > 0 else 0., use_bias=all_use_biases[i], return_vars=True)

            self.ys.append(y)
            self.weights_and_biases['W{}'.format(i)] = W
            if all_use_biases[i]:
                self.weights_and_biases['b{}'.format(i)] = b

            x = y

        self.y = y

        self.y_ = tf.compat.v1.placeholder(tf.float64, shape=(None, output_dims))

        self.loss = tf.compat.v1.losses.mean_squared_error(self.y_, self.y)
        # self.loss = tf.reduce_mean(tf.squared_difference(self.y_, self.y))

        self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

        self.init_op = tf.compat.v1.global_variables_initializer()

        self.sess = tf.compat.v1.Session()
        self.sess.run(self.init_op)

    def predict(self, X):

        X = np.atleast_2d(X)

        result = self.sess.run(self.y, feed_dict={
                self.x: X
        })

        return result

    def fit(self, X, y):

        assert X.shape[0] == y.shape[0],\
            'X.shape[0] != y.shape[0], {} != {}'.format(X.shape[0], y.shape[0])

        assert X.shape[1] == self.input_shape[0],\
            'X.shape[1] = {}, it should be {}'.format(X.shape, self.input_shape)
        assert y.shape[1] == self.output_shape[0],\
            'y.shape[1] = {}, it should be {}'.format(y.shape, self.output_shape)

        self.sess.run(self.train, feed_dict={
                self.x : X,
                self.y_: y
        })

    def partial_fit(self, X, y):
        self.fit(np.atleast_2d(X), np.atleast_2d(y))

    def target_update(self, model, tau=5e-3):
        for key in self.weights_and_biases.keys():
            var_target, var = self.weights_and_biases[key], model.weights_and_biases[key]
            if var_target is None and var is None:
                continue
            self.sess.run(var_target.assign(tau*var+(1-tau)*var_target))

    def copy(self, **replace_args):
        args = dict(list(self.init_args.items())+list(replace_args.items()))
        self_copy = FullyConnectedDNN(**args)
        self_copy.target_update(self, tau=1.)
        return self_copy


class CriticModel(FullyConnectedDNN):

    def __init__(self, state_dims,
                 action_dims,
                 hidden_layers=(400, 300),
                 activations=(tf.nn.relu, tf.nn.relu),
                 output_activation=None,
                 output_use_bias=False,
                 **kwargs):

        self.state_dims = state_dims
        self.action_dims = action_dims

        self.states = tf.compat.v1.placeholder(tf.float64, shape=(None, state_dims))
        self.actions = tf.compat.v1.placeholder(tf.float64, shape=(None, action_dims))

        x_input = tf.compat.v1.concat([self.states, self.actions], axis=1)

        super().__init__(input_dims=state_dims + action_dims, output_dims=1, hidden_layers=hidden_layers,
                         activations=activations, output_activation=output_activation, output_use_bias=output_use_bias,
                         x=x_input, **kwargs)

    def Q(self, states, actions):
        states = np.atleast_2d(states)
        actions = np.atleast_2d(actions)

        assert states.shape[0] == actions.shape[0]
        assert states.shape[1] == self.state_dims
        assert actions.shape[1] == self.action_dims

        # state_action_pairs = np.hstack([states, actions])

        res = self.sess.run(self.y, feed_dict={self.states: states,
                                               self.actions: actions})

        return res.flatten()


class ActorModel(FullyConnectedDNN):

    def __init__(self, state_dims, action_dims, hidden_layers=(400, 300), activations=(tf.nn.relu, tf.nn.relu),
                 output_activation=tf.nn.tanh, output_use_bias=False, **kwargs):
        super().__init__(input_dims=state_dims, output_dims=action_dims, hidden_layers=hidden_layers,
                         activations=activations, output_activation=output_activation, output_use_bias=output_use_bias, **kwargs)

        self.states = self.x

    def policy(self, states):
        states = np.atleast_2d(states)

        assert states.shape[1] == self.input_dims

        result = self.predict(states)

        assert result.shape[1] == self.output_dims
        assert result.shape[0] == states.shape[0]

        return result
