import unittest
from td3 import *

class TestCritic(unittest.TestCase):

    def setUp(self):
        self.critic = Critic(3, 1)

    def test_Q_shape(self):
        self.assertTrue(self.critic.Q(np.array([0, 1, 2]), np.array([0])).shape == (1,),
                        self.critic.Q(np.array([0, 1, 2]), np.array([0])).shape == (1,))
        self.assertTrue(self.critic.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape == (1,),
                        self.critic.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape == (1,))
        self.assertTrue(self.critic.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape == (3,),
                        self.critic.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape == (3,))

    def test_update(self):
        try:
            self.critic.update(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                               np.array([[0], [1], [2]]),
                               np.array([0, 1, 2]),
                               np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
                               np.array([[0], [1], [2]]))

        except Exception as e:
            self.assertTrue(False, "Update threw Exception {}".format(e))


if __name__ == '__main__':
    unittest.main()
