import numpy
from collections import defaultdict


def epsilon_greedy_policy(env, S, Q, epsilon):
    """Act epsilon greedily with respect to action-value funtion Q in state S.

    env is an OpenAI gym environment.
    S is state where the agent is.
    Q is action-value function.
    epsilon is probability of taking random action.
    """
    if numpy.random.rand() < epsilon:
        return env.action_space.sample()
    # else choose action which maximize Q in S
    return numpy.argmax([Q[S, A] for A in range(env.action_space.n)])


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Sarsa algorithm.

    Return action-value function for optimal epsilon-greedy policy for an
    environment.

    env is an OpenAI gym environment.
    n_episodes is number of episodes to run.
    gamma is an discount factor.
    alpha is an learning rate.
    epsilon is probability of selecting random action.
    """
    # rather than store Q as table use dictionary (default value is 0)
    Q = defaultdict(float)
    for _ in range(n_episodes):
        # reset environment
        S = env.reset()
        # choose epsilon-greedy action with respect to Q
        A = epsilon_greedy_policy(env, S, Q, epsilon)
        while True:
            S_prime, R, done, _ = env.step(A)
            A_prime = epsilon_greedy_policy(env, S_prime, Q, epsilon)
            # temporal-difference update
            Q[S, A] += alpha * (R + gamma * Q[S_prime, A_prime] - Q[S, A])
            S, A = S_prime, A_prime
            if done:
                break
    return Q


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Q-learning algorithm.

    Return optimal action value function Q. It is off-policy
    temporal-difference method.

    env is an OpenAI gym environment.
    n_episodes is number of episodes to run.
    gamma is an discount factor.
    alpha is an learning rate.
    epsilon is probability of selecting random action.
    """
    # rather than store Q as table use dictionary (default value is 0)
    Q = defaultdict(float)
    for _ in range(n_episodes):
        S = env.reset()
        while True:
            # select epsilon-greedy action
            A = epsilon_greedy_policy(env, S, Q, epsilon)
            S_prime, R, done, _ = env.step(A)
            # Q update
            max_Q = numpy.max(
                    [Q[S_prime, A] for A in range(env.action_space.n)]
                    )
            Q[S, A] += alpha * (R + gamma * max_Q - Q[S, A])
            S = S_prime
            if done:
                break
    return Q


def get_policy(env, Q):
    """Return greedy policy with respect to action-value function Q.

    env is an OpenAI environment.
    Q is action-value function.
    """
    policy = numpy.zeros((env.height, env.width), numpy.int)
    for i in range(env.height):
        for j in range(env.width):
            S = i, j
            policy[S] = numpy.argmax(
                    [Q[S, A] for A in range(env.action_space.n)]
                    )
    return policy
