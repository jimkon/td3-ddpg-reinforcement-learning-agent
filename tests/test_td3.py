import unittest
from td3 import *


class TestCritic(unittest.TestCase):

    def setUp(self):
        self.critic = Critic(3, 1)

    def test_Q_shape(self):
        self.assertTrue(self.critic.Q(np.array([0, 1, 2]), np.array([0])).shape == (1,),
                        self.critic.Q(np.array([0, 1, 2]), np.array([0])).shape)
        self.assertTrue(self.critic.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape == (1,),
                        self.critic.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape)
        self.assertTrue(self.critic.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape == (3,),
                        self.critic.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape)

    def test_update(self):
        try:
            self.critic.update(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                               np.array([[0], [1], [2]]),
                               np.array([0, 1, 2]),
                               np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                               np.array([[0], [1], [2]]))
        except Exception as e:
            self.assertTrue(False, "Update threw Exception {}".format(e))

        try:
            self.critic.update(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                               np.array([[0], [1], [2]]),
                               np.array([0, 1, 2]),
                               np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                               np.array([[0], [1], [2]]), update_targets=True)

        except Exception as e:
            self.assertTrue(False, "Update threw Exception {}".format(e))


class TestActor(unittest.TestCase):

    def setUp(self):
        critic = Critic(3, 2)
        self.actor = Actor(3, 2, critic)

    def test_pi_shape(self):
        self.assertTrue(self.actor.pi(np.array([0, 1, 2])).shape == (1, 2),
                        self.actor.pi(np.array([0, 1, 2])).shape)
        self.assertTrue(self.actor.pi(np.array([[0, 1, 2]])).shape == (1, 2),
                        self.actor.pi(np.array([[0, 1, 2]])).shape)
        self.assertTrue(self.actor.pi(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])).shape == (3, 2),
                        self.actor.pi(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])).shape)

    def test_update(self):
        try:
            self.actor.update(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
        except Exception as e:
            self.assertTrue(False, "Update threw Exception {}".format(e))


if __name__ == '__main__':
    unittest.main()
