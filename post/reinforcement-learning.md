Title: Introduction to Reinforcement Learning
Date: 2018-01-04 08:03
Category: reinforcement learning
Tags: reinforcement learning
Status: draft

In this post I would like to introduce reinforcement learning
because it is currently producing exciting results
(see [OpenAI][2] and [DeepMind][3]).
and in my opinion leads to general artificial intelligence.
For further reading please refer to book
[Reinforcement Learning: An Introduction][4] by Sutton and Barto.

[1]: http://incompleteideas.net/book/the-book.html
     (Sutton and Barto - Reinforcement Learning: An Introduction)

Reinforcement learning is a third machine learning paradigm alongside
supervised and unsupervised learning and perhaps others.
From all forms of machine learning it is the closest to the kind of learning
that humans and other animals do.
A computational approach to learning from interaction.
Reinforcement learning is learning what to do
(how to map situations to actions)
so as to maximize a numerical reward signal.
There are two main characteristics:

- trial-and-error search,
- delayed reward.

A learning agent interacting over time with its environment to achieve a goal.
It must be able to sense the state of its environment and able to take actions
that affect the state.

One of the challenges that arise in reinforcement learning is to balance
exploration and exploitation.
The agent has to exploit an action to obtain reward but it also has to explore
to find new better action for future.

## Policy, Reward Signal, Value Function and Model of Environment

A policy, a reward signal, a value function and optionally a model of an
environment are four main subelements of a reinforcement learning system.

A *policy* is a mapping from perceived stated of the environment to actions
when in those stated.
Therefore, it defines the learning agent's behaviour at a given time.

A *reward signal* defines the goal in a reinforcement learning problem.
It is a single number send to the learning agent by the environment at every
time step.
The agent's objective is to maximize the total reward over the long run.
Thus, the reward signal is the primary source for altering the policy.

On contrary to reward signal, which is good in an immediate sense,
*value function* specifies what is good in the long run.
What an agent can expect to accumulate from a state over the future.
While a reward signal might be low in some state it might have high value
as it is usually followed by states that yield high rewards.

The last element is a *model of environment* which mimics the behaviour
of the environment or allows inferences about how the environment will behave.
They are used for estimating future situations before they are actually
experienced (planning).

Most of the reinforcement learning methods are concerned with estimating value
functions (except evolutionary methods).
