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

    def update(self, states, actions, rewards, states_, actions_, dones, noise_std=.2, noise_clip=.5, action_lims=None):

        noise = np.random.standard_normal(actions_.shape)*noise_std
        noise = np.clip(noise, -noise_clip, noise_clip)
        actions_ += noise
        actions_ = np.clip(actions_, action_lims[0], action_lims[1])

        min_q_s_a_ = self.Q(states_, actions_)
        q_sa = np.reshape(rewards + self.gamma * min_q_s_a_ * dones, (-1, 1))

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

    def __init__(self, state_dims, action_dims, critic, lr=1e-3, tau=5e-3, sess=None, **nn_args):
        self.tau = tau

        self.model = ActorModel(state_dims=state_dims, action_dims=action_dims, **nn_args)
        self.target_model = ActorModel(state_dims=state_dims, action_dims=action_dims, **nn_args)

        self.critic = critic

        self.states = tf.compat.v1.placeholder(tf.float64, shape=(None, state_dims))
        self.actions = self.model.y

        critic_input = tf.compat.v1.concat([self.states, self.actions], axis=1)
        self.critic_graph = self.critic.model1.copy(x=critic_input)

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

        self.critic_graph.target_update(self.critic.model1, tau=1.)

        self.sess.run(self.train, feed_dict={self.states: states,
                                             self.model.states: states})

    def update_target(self):
        self.target_model.target_update(self.model, self.tau)


class ExperienceReplay:

        def __init__(self, maxsize):
            self.__maxsize = maxsize
            self.__states = []
            self.__actions = []
            self.__rewards = []
            self.__states_ = []
            self.__dones = []

        def push(self, state, action, reward, state_, done):
            self.__states.append(state)
            self.__actions.append(action)
            self.__rewards.append(reward)
            self.__states_.append(state_)
            self.__dones.append(done)
            self.__pop_extra()

        def __pop_extra(self):
            extra = self.length() - self.__maxsize
            for _ in range(extra):
                self.__states.pop(0)
                self.__actions.pop(0)
                self.__rewards.pop(0)
                self.__states_.pop(0)
                self.__dones.pop(0)

        def get_random(self, count=1):
            indexes = np.random.randint(0, self.length(), count)
            states = np.array([self.__states[index] for index in indexes])
            actions = np.array([self.__actions[index] for index in indexes])
            rewards = np.array([self.__rewards[index] for index in indexes])
            states_ = np.array([self.__states_[index] for index in indexes])
            dones = np.array([self.__dones[index] for index in indexes])
            return states, actions, rewards, states_, dones

        def length(self):
            return len(self.__states)


class TD3:

    def __init__(self, state_dims, action_dims, action_low, action_high, batch_size=100, target_update_rate=2):

        self.critic = Critic(state_dims=state_dims, action_dims=action_dims)
        self.actor = Actor(state_dims=state_dims, action_dims=action_dims, critic=self.critic)

        self.action_lims = np.array([action_low, action_high])

        self.buffer = ExperienceReplay(maxsize=10000)

        self.target_update_rate = target_update_rate
        self.batch_size = batch_size
        self.step_counter = 0

    def act(self, state):
        return self.actor.pi(state)[0]

    def observe(self, state, action, reward, state_, done, episode=-1, step=-1):
        self.buffer.push(state, action, reward, state_, 0 if done else 1)

        if self.step_counter>self.batch_size:
            states, actions, rewards, states_, dones = self.buffer.get_random(self.batch_size)

            actions_ = self.actor.pi(states_)

            self.critic.update(states, actions, rewards, states_, actions_, action_lims=self.action_lims)

            if self.step_counter % self.target_update_rate == self.target_update_rate-1:
                self.actor.update(states)
                self.critic.update_targets()
                self.actor.update_target()

        self.step_counter += 1



