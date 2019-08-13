import unittest
from nets import *


class TestCriticModel(unittest.TestCase):

    def setUp(self):
        self.model = CriticModel(state_dims=3, action_dims=1, drop_out=.0)

    def test_Q_shape(self):
        self.assertTrue(self.model.Q(np.array([0, 1, 2]), np.array([0])).shape == (1,),
                        self.model.Q(np.array([0, 1, 2]), np.array([0])).shape == (1,))
        self.assertTrue(self.model.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape == (1,),
                        self.model.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape == (1,))
        self.assertTrue(self.model.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape == (3,),
                        self.model.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape == (3,))

    # def test_advantages(self):
    #     def adv(g, r, v):
    #         return self.model.sess.run(self.model.advantages,
    #                                    feed_dict={self.model.gammas: g,
    #                                               self.model.rewards: r,
    #                                               self.model.vs: v
    #                                    })
    #     temp = adv([1, .9, .8, .7, .6, .5, .4, .3, .2, .1],
    #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
    #     self.assertTrue(temp.shape == (10,), temp.shape)
    #     self.assertTrue(((temp - np.array([.5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4])) < 1e-10).all())

if __name__ == '__main__':
    unittest.main()
