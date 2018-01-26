from gym.envs.registration import register

register(
        id='GridWorld-v0',
        entry_point='gym_basic.envs:GridWorldEnv',
        )
register(
        id='WindyGridWorld-v0',
        entry_point='gym_basic.envs:WindyGridWorldEnv',
        )
