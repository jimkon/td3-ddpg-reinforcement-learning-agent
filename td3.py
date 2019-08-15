import numpy as np
import tensorflow as tf

from nets import CriticModel, ActorModel


class Critic:

    def __init__(self, state_dims, action_dims, gamma=.99, lr=1e-3, tau=5e-3, sess=None, **nn_args):
        self.gamma = gamma
        self.tau = tau

        self.model1 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.model2 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.target_model1 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.target_model2 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)

        self.min_Q_sa = tf.compat.v1.placeholder(tf.float64, shape=(None, 1))

        self.loss = tf.losses.mean_squared_error(self.min_Q_sa, self.model1.y) +\
                    tf.losses.mean_squared_error(self.min_Q_sa, self.model2.y)
        self.train = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(self.loss)

        self.sess = tf.Session() if sess is None else sess
        self.model1.sess = self.sess
        self.target_model1.sess = self.sess
        self.model2.sess = self.sess
        self.target_model2.sess = self.sess

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def update(self, states, actions, rewards, states_, actions_):
        min_q_s_a_ = self.Q(states_, actions_)
        q_sa = np.reshape(rewards+self.gamma*min_q_s_a_, (-1, 1))

        self.sess.run(self.train, feed_dict={self.model1.states: states,
                                             self.model1.actions: actions,
                                             self.model2.states: states,
                                             self.model2.actions: actions,
                                             self.min_Q_sa: q_sa})

    def Q(self, states, actions):
        return np.min([self.target_model1.Q(states, actions), self.target_model2.Q(states, actions)], axis=0)

    def update_targets(self):
        self.target_model1.target_update(self.model1, self.tau)
        self.target_model2.target_update(self.model2, self.tau)


class Actor:

    def __init__(self, state_dims, action_dims, critic_model, lr=1e-3, tau=5e-3, sess=None, **nn_args):
        self.tau = tau

        self.model = ActorModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.target_model = ActorModel(state_dims=state_dims, action_dims=action_dims, **nn_args)

        self.critic_model = critic_model

        self.states = tf.compat.v1.placeholder(tf.float64, shape=(None, state_dims))
        self.actions = self.model.y

        critic_input = tf.compat.v1.concat([self.states, self.actions], axis=1)
        self.critic_graph = self.critic_model.model1.copy(x=critic_input)

        self.loss = -self.critic_graph.y
        self.train = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(self.loss,
                                                                              var_list=list(self.model.weights_and_biases.values()))

        self.sess = tf.Session() if sess is None else sess
        self.model.sess = self.sess
        self.target_model.sess = self.sess

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def pi(self, states):
        return self.target_model.policy(states)

    def update(self, states):

        self.critic_graph.target_update(self.critic_model.model1, tau=1.)

        self.sess.run(self.train, feed_dict={self.states: states,
                                             self.model.states: states})

    def update_target(self):
        self.target_model.target_update(self.model, self.tau)
