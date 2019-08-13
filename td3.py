import numpy as np
import tensorflow as tf

from nets import CriticModel


class Critic:

    def __init__(self, state_dims, action_dims, gamma=.99, lr=1e-3, tau=5e-3, **nn_args):
        self.gamma = gamma
        self.tau = tau

        self.model1 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.model2 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.target_model1 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.target_model2 = CriticModel(state_dims=state_dims, action_dims=action_dims, **nn_args)

        self.min_Q_s_a = tf.compat.v1.placeholder(tf.float64, shape=(None, 1))

        self.loss = tf.losses.mean_squared_error(self.min_Q_s_a, self.model1.y) +\
                    tf.losses.mean_squared_error(self.min_Q_s_a, self.model2.y)
        self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

        self.sess = tf.Session()
        self.model1.sess = self.sess
        self.target_model1.sess = self.sess
        self.model2.sess = self.sess
        self.target_model2.sess = self.sess

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def update(self, states, actions, rewards, states_, actions_, update_targets=False):
        min_q_s_a_ = np.min([self.model1.Q(states_, actions_), self.model2.Q(states_, actions_)], axis=0)
        q_sa = np.reshape(rewards+self.gamma*min_q_s_a_, (-1, 1))

        states_actions = np.hstack([states, actions])

        self.sess.run(self.train, feed_dict={self.model1.x: states_actions,
                                             self.model2.x: states_actions,
                                             self.min_Q_s_a: q_sa})

        if update_targets:
            self.target_model1.target_update(self.model1, self.tau)
            self.target_model2.target_update(self.model2, self.tau)

    def Q(self, states, actions):
        return np.min([self.model1.Q(states, actions), self.model2.Q(states, actions)], axis=0)

Critic(3, 1)