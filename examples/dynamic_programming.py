import numpy


def policy_evaluation(policy, env, gamma=1.0, epsilon=1e-5):
    """Policy evaluation algorithm.

    Compute state-value function of given policy.

    policy is the policy to by evaluated in form of transition matrix.
    env is a OpenAI gym environment transition dynamics as attribute P.
    gamma is a discount rate.
    epsilon is a threshold determining accuracy of estimation.
    """
    # initialize state-value function arbitrarily
    V = numpy.zeros(env.observation_space.n)
    # while unsufficient accuracy defined by treshold
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            # apply update given by Bellman equation
            v = V[s]
            acc = 0
            for a in range(env.action_space.n):
                acc += policy[s, a] * (env.P[a, s] @ (env.R[a, s] + gamma * V))
            V[s] = acc
            # store biggest inaccuracy
            delta = max(delta, numpy.abs(v - V[s]))
        if delta < epsilon:
            return V


def policy_improvement(env, V, gamma=1.0):
    """Policy improvement algorithm.

    Return transition matrix of policy given by state-value function V.

    env is a OpenAI gym environment transition dynamics as attribute P.
    V is numpy array of state-values.
    gamma is a discount rate.
    """
    # construct empty transition matrix
    policy = numpy.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        # for each state construct state-action value
        action_values = numpy.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            action_values[a] = env.P[a, s] @ (env.R[a, s] + gamma * V)
        # choose the best action
        a = numpy.argmax(action_values)
        # set probability 1 for that action
        policy[s, a] = 1.0
    return policy
