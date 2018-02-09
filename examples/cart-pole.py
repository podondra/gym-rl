# https://gym.openai.com/docs/
import gym
from gym import envs, spaces


print(envs.registry.all())

env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

space = spaces.Discrete(8)
x = space.sample()
assert space.contains(x)
assert space.n == 8

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()  # take a random action
        observation, reward, done, info = env.step(action)
        if done:
            print('episode finished after {:>2} time steps'.format(t))
            break
env.close()
