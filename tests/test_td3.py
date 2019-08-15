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
            self.critic.update_targets()

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


class TestExperienceReplay(unittest.TestCase):

    def test_push(self):
        f = np.arange
        buffer = ExperienceReplay(10)
        buffer.push(f(3), f(2), f(1), f(3))
        s, a, r, s_ = buffer.get_random(1)
        self.assertTrue(np.array_equal(s[0], f(3)))
        self.assertTrue(np.array_equal(a[0], f(2)))
        self.assertTrue(np.array_equal(r[0], f(1)))
        self.assertTrue(np.array_equal(s_[0], f(3)))

        self.assertEqual(buffer.length(), 1)
        for i in range(2, 20):
            buffer.push(f(3), f(2), f(1), f(3))
            self.assertEqual(buffer.length(), min(i, 10))

    def test_get_random(self):
        f = np.arange
        buffer = ExperienceReplay(10)
        for i in range(2, 20):
            buffer.push(f(3), f(2), f(1), f(3))

        s, a, r, s_ = buffer.get_random(3)
        self.assertTrue(len(s) == 3)
        self.assertTrue(len(a) == 3)
        self.assertTrue(len(r) == 3)
        self.assertTrue(len(s_) == 3)



if __name__ == '__main__':
    unittest.main()
