import pytest
from collections import defaultdict
from examples import temporal_difference


@pytest.fixture
def windygridworld():
    import gym
    import gym_gridworlds   # noqa
    return gym.make('WindyGridworld-v0')


def test_sarsa(windygridworld):
    Q = temporal_difference.sarsa(windygridworld, 1000)
    # check that whole state-action space was explored
    assert isinstance(Q, defaultdict)


def test_q_learning(windygridworld):
    # solve windy gridworld
    Q = temporal_difference.q_learning(windygridworld, 200)
    # get optimal policy
    policy = temporal_difference.get_policy(windygridworld, Q)
    # apply it to environment
    S = windygridworld.reset()
    G = 0
    for t in range(16):
        A = policy[S]
        S, R, done, _ = windygridworld.step(A)
        G += R
        if done:
            break
    # check that the trajectory was optimal
    assert G == -15
    assert t == 14
