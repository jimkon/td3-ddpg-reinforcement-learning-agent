import unittest
from nets import *


class TestFullyConnectedDNN(unittest.TestCase):

    def setUp(self):
        self.model = FullyConnectedDNN(input_dims=3, output_dims=2, drop_out=.0)

    def test_predict_shape(self):
        self.assertTrue(self.model.predict(np.array([0, 1, 2])).shape == (1, 2),
                        self.model.predict(np.array([0, 1, 2])).shape)
        self.assertTrue(self.model.predict(np.array([[0, 1, 2]])).shape == (1, 2),
                        self.model.predict(np.array([[0, 1, 2]])).shape)
        self.assertTrue(self.model.predict(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])).shape == (3, 2),
                        self.model.predict(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])))

    def test_copy(self):
        copy = self.model.copy()
        self.assertTrue(np.array_equal(self.model.predict(np.array([0, 1, 2])), copy.predict(np.array([0, 1, 2]))),
                        '{} != {}'.format(self.model.predict(np.array([0, 1, 2])), copy.predict(np.array([0, 1, 2]))))

        self.assertTrue(self.model.copy(drop_out=.3).init_args['drop_out'] == .3,
                        self.model.copy(drop_out=.3).init_args['drop_out'])


class TestCriticModel(unittest.TestCase):

    def setUp(self):
        self.model = CriticModel(state_dims=3, action_dims=1, drop_out=.0)

    def test_Q_shape(self):
        self.assertTrue(self.model.Q(np.array([0, 1, 2]), np.array([0])).shape == (1,),
                        self.model.Q(np.array([0, 1, 2]), np.array([0])).shape)
        self.assertTrue(self.model.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape == (1,),
                        self.model.Q(np.array([[0, 1, 2]]), np.array([[0]])).shape)
        self.assertTrue(self.model.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape == (3,),
                        self.model.Q(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), np.array([[0], [1], [2]])).shape)


class TestActorModel(unittest.TestCase):

    def setUp(self):
        self.model = ActorModel(state_dims=3, action_dims=2, drop_out=.0)

    def test_policy_shape(self):
        self.assertTrue(self.model.policy(np.array([0, 1, 2])).shape == (1, 2),
                        self.model.policy(np.array([0, 1, 2])).shape)
        self.assertTrue(self.model.policy(np.array([[0, 1, 2]])).shape == (1, 2),
                        self.model.policy(np.array([[0, 1, 2]])).shape)
        self.assertTrue(self.model.policy(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])).shape == (3, 2),
                        self.model.policy(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])).shape)


if __name__ == '__main__':
    unittest.main()
