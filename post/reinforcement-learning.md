Title: Introduction to Reinforcement Learning
Date: 2018-01-04 08:03
Category: reinforcement learning
Tags: reinforcement learning
Status: draft

Refer to book [Reinforcement Learning: An Introduction][4] by Sutton and Barto.

[4]: http://incompleteideas.net/book/the-book.html
     (Sutton and Barto - Reinforcement Learning: An Introduction)

Reinforcement learning is a third machine learning paradigm alongside
supervised and unsupervised learning and perhaps others.
From all forms of machine learning it is the closest to the kind of learning
that humans and other animals do.
A coputational approach to learning from interaction.
Reinforcement learning is learning what to do (how to map situations to
actions) so as to maximize a numerical reward signal.
There are two main characteristics:

- trial-and-error search,
- delayed reward.

A learning agent interacting over time with its environment to achieve a goal.
It must be able to sense the state of its environment and able to take actions
that affect the state.

One of the challenges that arise in reinforcement learning is to balance
exploration and exploation.
The agent has to exploit an action to obtain reward but it also has to explore
to find new better action for future.

## Policy, Reward Signal, Value Function and Model of Environment

A policy, a reward signal, a value function and optionally a model of the
environment are four main subelements of a reinforcement learning system.

A policy is a mapping from perceived stated of the environment to actions
when in those stated. Therefore, it defines the learning agent's behaviour
at a given time.

A reward signal defines the goal in a reinforcement learning problem.
It is a single number send to the learning agent by the environment at every
time step.
The agent's objective is to maximalize the total reward over the long run.
Thus, the reward signal is the primary source for altering the policy.

On contrary to reward signal, which is good in an immediate sense,
value function specifies what is good in the long run.
What an agent can expect to accumulate from a state over the future.
While a reward signal might be low in some state it might have high value
as it is usually followed by states that yield high rewards.

The last element is a model of the environment which mimics the behaviour
of the environment or allows inferences about how the environment will behave.
They are used for estimating future situations before they are actually
experienced (planning).

Most of the reinforcement learning methods are concerned with estimating value
functions (except evolutionary methods).

## Q-learning

Q-learning is model-free reinforcement learning method which finds optimal
action-selection policy for a given finite Markov decision process (MDP)
by learning an action-value function $Q$:

$Q^{*}(s, a) = max_{\pi} \mathbb{E}[r_t + \gamma r_{t + 1} + \gamma ^ 2
r_{t + 2} + \dots | s_t = s, a_t = a, \pi]$

which is the maximum sum of rewards $r_t$ discounted by $\gamma$ at each time
step $t$, achievable by a behaviour policy $\pi = P(a|s)$,
after making an observation $s$ and taking action $a$.

It is proven that Q-learning method will find a optimal policy
for any finite MDP.

[5]: http://www.cs.rhul.ac.uk/~chrisw/thesis.html
     (Learning from Delayed Rewards)
[6]: https://link.springer.com/article/10.1007/BF00992698
     (Technical Note Q-learning)
