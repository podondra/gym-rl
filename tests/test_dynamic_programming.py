import gym
import gym_gridworlds   # noqa
import numpy
from examples import dynamic_programming


def test_policy_evaluation():
    env = gym.make('Gridworld-v0')
    # define transition for stochastic policy
    stochastic = numpy.ones((env.observation_space.n, env.action_space.n))
    stochastic /= env.action_space.n
    # evaluate the stochastic policy
    V = dynamic_programming.policy_evaluation(stochastic, env)
    optimal_V = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22,
                 -20, -14]
    assert numpy.allclose(V, optimal_V)
