# inspired by
# http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html
import numpy


class LinearPolicy:
    def __init__(self, theta, env):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.W = theta[:obs_dim * act_dim].reshape(obs_dim, act_dim)
        self.b = theta[obs_dim * act_dim:]

    def act(self, observation):
        y = numpy.dot(observation, self.W) + self.b
        return y.argmax()


def run_episode(policy, env, n_timesteps, render=False):
    total_reward = 0
    S = env.reset()
    for t in range(n_timesteps):
        a = policy.act(S)
        S, R, done, _ = env.step(a)
        total_reward += R
        if render:
            env.render()
        if done:
            break
    return total_reward


def noisy_evaluation(theta, env, n_timesteps):
    policy = LinearPolicy(theta, env)
    return run_episode(policy, env, n_timesteps)


def cross_entropy_method(
        env, n_iteration, n_timesteps, batch_size=25, elite=0.2, render=True
        ):
    theta_dim = (env.observation_space.shape[0] + 1) * env.action_space.n
    theta_mean = numpy.zeros(theta_dim)
    theta_std = numpy.ones(theta_dim)
    n_elite = int(batch_size * elite)

    for iteration in range(n_iteration):
        # sample parameter vectors
        thetas = numpy.random.normal(
                loc=theta_mean,
                scale=theta_std,
                size=(batch_size, theta_dim)
                )
        rewards = numpy.zeros(batch_size)
        for i, theta in enumerate(thetas):
            rewards[i] = noisy_evaluation(theta, env, n_timesteps)
        # get elite parameters
        elite_idxs = numpy.argsort(rewards)[-n_elite:]
        elite_thetas = thetas[elite_idxs]
        theta_mean = elite_thetas.mean(axis=0)
        theta_std = elite_thetas.std(axis=0)
        print('iteration:{:9d} mean reward: {:f} max reward: {:f}'.format(
            iteration, numpy.mean(rewards), numpy.max(rewards)
            ))
        policy = LinearPolicy(theta_mean, env)
        run_episode(policy, env, n_timesteps, render)
