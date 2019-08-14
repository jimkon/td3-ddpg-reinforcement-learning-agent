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


if __name__ == '__main__':
    unittest.main()
