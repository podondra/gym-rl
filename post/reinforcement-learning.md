Title: Introduction to Reinforcement Learning
Date: 2018-01-04 08:03
Category: reinforcement learning
Tags: reinforcement learning

In this post I would like to introduce reinforcement learning
because it is currently producing exciting results
(see [OpenAI] and [DeepMind]).
and in my opinion leads to general artificial intelligence.
For further reading please refer to book
[Reinforcement Learning: An Introduction][sutton2018] by Sutton and Barto.

[openai]: https://openai.com/ (OpenAI)
[deepmind]: https://deepmind.com/ (DeepMind)
[sutton2018]: http://incompleteideas.net/book/the-book-2nd.html
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

## Multi-armed Bandits

Reinforcement learning uses training information
which *evaluates* the actions taken
rather than *instructs* by giving correct actions.
Therefore the search for good behaviour is needed.
Purely evaluative feedback indicates how good the action taken was
so depends entirely on the action.
Purely instructive feedback indicates the correct action independently of
action actually taken.

### A $k$-armed Bandit Problem

The $k$-armed bandit problem is named by analogy to
[slot machine][slot-machine].
An agent is repeatedly faced with a choice among $k$ different actions.
After each choice it receive a numerical reward from a stationary probability
distribution.
The objective is to maximize the expected total reward over some *time steps*.

[slot-machine]: https://en.wikipedia.org/wiki/Slot_machine (Slot Machine)

In the $k$-armed bandit problem each of the $k$ actions has an expected reward
given that the action is selected (action's value).
The action selected at time step $t$ is denoted as $A_t$
and its reward as $R_t$.
The value of an arbitrary action $a$ is the expected reward
given that $a$ is selected:

$$q_*(a) \equiv \mathrm{E}(R_t | A_t = a).$$

If the value of each function was known it would be trivial to solve
the $k$-armed bandit problem.
But their are not certainly known although there might be estimates.
The estimated value of action $a$ at time step $t$ is $Q_t(a)$
and it should be as close as possible to $q_*(a)$.

### Exploration and Exploitation

Actions whose estimated value is greatest are called *greedy* actions.
Selecting one of these actions is *exploiting* current knowledge of the
values of the actions.
If a non-greedy action is selected than an agent is *exploring*
cause it enables to improve estimate of the non-greedy action.
Exploitation maximize the expected reward one step
and exploration may produce greater total reward.
It is impossible to explore and exploit with any single action selection
so this is referred as *conflict* between exploration and exploitation.
The need to balance exploration and exploitation is big challenge in
reinforcement learning.

### Action-value Methods

The true value of an action it the mean reward
so natural way to estimate is by averaging received rewards:

$$Q_t(a) \equiv \frac{\sum^{t - 1}_{i = 1} R_i \cdot \mathbb{1}_{A_i = a}}
{\sum^{t - 1}_{i = 1} \mathbb{1}_{A_i = a}},$$

where $\mathbb{1}_{\text{condition}}$ is random variable indicator
that is $1$ if condition is true else $0$.
If the denominator is $0$ then $Q_t(a)$ is defined as a default value
(for example $0$).
Moreover if the denominator goes to infinity
then $Q_t(a)$ converges to $q_*(a)$.
This way of estimating action values is called *sample-average*.

The simplest action selection rule based on sample-average is to select the
action with highest estimated value (greedy action):

$$A_t \equiv \operatorname*{arg\,max}_a Q_t(a).$$

Greedy action selection always exploit current knowledge to maximize immediate
reward and spends no time exploring.
Simple modification is with probability $\varepsilon$ select instead randomly
any action with equal probability independently of the action-value estimates.
These which use near-greedy action selection rule are called
*$\varepsilon$-greedy* methods.
Advantage of these methods is in limit every action will be sampled
an infinite number of times thus all $Q_t(a)$ converge to $q_*(a)$.
