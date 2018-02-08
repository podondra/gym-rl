import pytest
import numpy
from examples import dynamic_programming


@pytest.fixture
def gridworld():
    import gym
    import gym_gridworlds   # noqa
    return gym.make('Gridworld-v0')


@pytest.fixture
def true_state_values():
    return numpy.array(
            [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22,
                -20, -14]
            )


def test_policy_evaluation(gridworld, true_state_values):
    # define transition for stochastic policy
    stochastic = numpy.ones(
            (gridworld.observation_space.n, gridworld.action_space.n)
            )
    stochastic /= gridworld.action_space.n
    # evaluate the stochastic policy
    V = dynamic_programming.policy_evaluation(stochastic, gridworld)
    assert numpy.allclose(V, true_state_values)


def test_policy_improvement(gridworld, true_state_values):
    policy = dynamic_programming.policy_improvement(
            gridworld, true_state_values
            )
    # make sure that probabilities sum up to 1
    assert numpy.all(policy.sum(axis=1) == 1)
