import gym
from gym import spaces
import numpy


class GridWorldEnv(gym.Env):
    reward_range = (-1, 0)
    action_space = spaces.Discrete(4)
    observation_space = spaces.Discrete(15)

    def __init__(self):
        gridworld = numpy.arange(16).reshape((4, 4))
        gridworld[-1, -1] = 0

        self.P = {0: {}}
        for a in range(4):
            self.P[0][a] = [(1.0, 0, 0, True)]

        for s in gridworld.flat[1:-1]:
            self.P[s] = {}
            row, col = numpy.argwhere(gridworld == s)[0]
            for a, d in zip(range(4), [(-1, 0), (0, 1), (1, 0), (0, -1)]):
                next_row = max(0, min(row + d[0], 3))
                next_col = max(0, min(col + d[1], 3))
                next_state = gridworld[next_row, next_col]
                done = True if s == 0 else False
                self.P[s][a] = [(1.0, next_state, -1, done)]
