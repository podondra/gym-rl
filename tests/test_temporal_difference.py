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
