Title: Introduction to Reinforcement Learning
Date: 2018-01-04 08:03
Category: reinforcement learning
Tags: reinforcement learning

In this post I would like to introduce reinforcement learning.
Reinforcement learning is currently producing exciting results (see [OpenAI]
and [DeepMind]) and in my opinion would lead to general artificial
intelligence.

[openai]: https://openai.com/ (OpenAI)
[deepmind]: https://deepmind.com/ (DeepMind)
[sutton2018]: http://incompleteideas.net/book/the-book-2nd.html (Sutton and Barto, Reinforcement Learning: An Introduction)

I also prepared some algorithm implemented in Python in my
[gym-rl GitHub repository][gym-rl] and I encourage you to also implement them
while reading this post, [the book][sutton2018] and
[UCL reinforcement learning course][ucl-course] by David Silver.

[gym-rl]: https://github.com/podondra/gym-rl
[ucl-course]: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

## Introduction

Reinforcement learning is a branch of machine learning.
It is concerned with taking sequences of actions.
From all forms of machine learning it is the closest to the humans' learning.
Reinforcement learning is learning what to do so as to maximize cumulative
reward.
Usually described in terms of agent interacting with previously unknown
environment.

Here are some reinforcement learning characteristics:

- trial-and-error search,
- no supervisor, only reward signal,
- delayed reward,
- time matter, sequential non-i.i.d. data, and
- agent's actions affect the received data.

A learning agent interacts over time with an environment to achieve a goal.
The agent observes the environment's state (robot's camera images and its joint
angles).
Takes actions (joint torques) that affect the state.
Gets rewards (for staying balanced, navigate to target location and so on).

Reinforcement learning has many faces.
It is in intersection of:

- computer science (machine learning),
- neuroscience (reward system),
- psychology,
- economics,
- mathematics and
- engineering (optimal control).

### Basic Concepts

A policy, a reward, a value function and a model of an environment are four
main subelements of a reinforcement learning.

A *policy* is a mapping from perceived stated to actions.
Agent's behavior function.
Maps state to action.
Can be defined as deterministic policy \\(a = \pi(s)\\)
or stochastic policy \\(\mathbb{P}(A\_t = a | S\_t = s) = \pi(a | s)\\).


A *reward* \\(R\_t\\) defines the goal in a reinforcement learning problem.
It is a scalar feedback signal send to the agent by the environment at every
time step \\(t\\).
The agent's objective is to maximize the total reward over the long run.
Reinforcement learning is based on the *reward hypothesis*:

> All goals can be described by the maximization of expected cumulative reward.

On contrary to reward *value function* specifies what is good in the long run.
It is used to evaluate goodness of states.
What an agent can expect to accumulate from a state over the future.

The last element is a *model of environment*.
It mimics the behavior of the environment or allows inferences about its
behavior.
They are used for estimating future situations before they are actually
experienced.

### Approaches to Reinforcement Learning

There are different approaches to reinforcement learning.
The two different strategies are to either optimize policy or work with value
functions.

Policy optimization includes evolutionary methods and policy gradients.
Value function methods includes dynamic programming, Monte Carlo methods,
temporal-difference learning and function approximation.
It their intersection are actor-critic methods.

## Markov Decision Processes

Markov decision processes formally describe an environment for reinforcement
learning.
They are mathematically idealized from of reinforcement learning problems.
Key elements are returns, value functions and Bellman equations.

In MDP the learner is an *agent* interacting with an *environment*.
Agent and environment interact at discrete time steps \\(t = 0, 1, 2, \dots\\).
The agent selects and action \\(A\_t \in \mathcal{A(s)}\\) and the environment
responses with reward \\(R\_{t + 1} \in \mathbb{R}\\) and next state
\\(S\_{t + 1} \in \mathcal{S}\\) and so forth.
This loop in MDP can be unrolled into a *trajectory*:

\\[S\_0, A\_0, R\_1, S\_1, A\_1, R\_2, S\_2, A\_2, R\_3, \dots\\] 

That gives a probability of occurrence for \\(s' \in \mathcal{S}\\) and
\\(r \in \mathcal{R}\\) at time \\(t\\) given preceding state and action:

\\[p(s', r | s, a) \equiv \mathbb{P}(S\_t = s', R\_t = r | S\_{t - 1} = s,
A\_{t - 1} = a),\\]

for all \\(s', s \in \mathcal{S}\\), \\(r \in \mathcal{R}\\)
and \\(a \in \mathcal{A}(s)\\).
The function
\\(p: \mathcal{S} \times \mathbb{R} \times \mathcal{S} \times \mathcal{A} \to [0, 1]\\)
is ordinary deterministic function of four arguments.
The probabilities give by the function \\(p\\) completely characterize the
dynamics of a finite MDP.

The agent's goal is to maximize *expected return*, the total amount of reward
\\(R\_t\\):

\\[G\_t \equiv \sum\_{k = t + 1}^T \gamma^{k - t - 1} R\_k,\\]

where \\(\gamma \in (0, 1]\\) is an discount rate and \\(T\\) is time horizon
which might be infinity but then \\(\gamma \neq 1\\) else the sum will not be
finite.
This is necessary due to distinction between continuing and episodic tasks.

Consider a game of Go.
An agent might get a reward of -1 for losing the game and 1 for winning the
game.
The agent always learns to maximize its reward so it is important to provide
reward in such way that the agent when maximizing its reward will also achieve
goal given.
The reward signal is not the place to impart *how* the agent such achieve
something but rather *what* to achieve.
For example there such be no rewards during the game of Go for some
intermediate accomplishments as the agent might learn to achieve these
subgoals without learning to win the game.

### Policies and Value Functions

Most of basic reinforcement learning algorithms involve a notion of value
functions.
They define the notion of how a state or a state and an action pair is good
in terms of future rewards that can be expected.
The value of a state is denoted \\(v\_{\pi}(s)\\), called *state-value
function for policy \\(\pi\\)* and for MPDs is formally:

\\[v\_{\pi}(s) \equiv \mathbb{E}\_{\pi}(G\_t | S\_t = s).\\]

Similarly is defined the value of taking action in state which is called
*action-value function for policy \\(\pi\\)*:

\\[q\_{\pi}(s, a) \equiv \mathbb{E}\_{\pi}(G\_t | S\_t = s, A\_t = a).\\]

A fundamental property of value functions is that they satisfy recursive
relationships called *Bellman equations*.
For example the Bellman equation of state-value function:

\\[v\_{\pi}(s) \equiv \mathbb{E}\_{\pi}(R\_{t + 1} + \gamma
v\_{\pi}(S\_{t + 1}) | S\_t = s).\\]

Solving a reinforcement learning problem is roughly finding a policy that
get a lot of long run reward.
An *optimal policy* is a policy which has the *optimal state-value function*
\\(v\_\*\\):

\\[v\_\*(s) \equiv \max\_{\pi} v\_{\pi}(s).\\]

The optimal policy also has the *optimal action-value function*:

\\[q\_\*(s, a) \equiv \max\_{\pi} q\_{\pi}(s, a).\\]

Having one of these functions makes determining optimal policy easy
as always action that maximizes the future reward is selected.
This policy is called *greedy* with respect to the optimal value function.

## Dynamic Programming

Dynamic programming refers to collection of algorithms used for computing
optimal policies
when a perfect model of the environment is given as a MDP.
These algorithms are limited by their assumptions that the MDP is fully known
and because they are very computationally demanding.
But their are very important as other algorithm might be viewed
as attempting the same effect as dynamic programming
but more effectively and without full knowledge of the environment.

The main idea of dynamic programming is to use value functions to structure
the search for good policies via Bellman optimality equations:

\\[v\_\*(s) = \max\_a \sum\_{s', r} p(s', r | s, a) \big[r + \gamma
v\_\*(s')\big]\\]

or

\\[q\_\*(s, a) = \sum\_{s', r} p(s', r | s, a) \big[r + \gamma \max\_{a'}
q\_\*(s', a')\big],\\]

for all \\(s, s' \in \mathcal{S}\\) and \\(a \in \mathcal{A}(s)\\).
Dynamic programming makes these equations into iterative update rules
for improving approximations of the desired value functions.

### Policy Evaluation

Consider how to compute state-value function \\(v\_{\pi}\\) for a given
policy \\(\pi\\).
This is referred as *policy evaluation* or *prediction problem*.

Having a sequence of approximate value function \\(v\_0, v\_1, v\_2, \dots\\)
each mapping \\(\mathcal{S^+} \to \mathbb{R}\\).
The first approximation \\(v\_0\\) is chosen arbitrarily.
Only the terminal state must have value \\(0\\).
Then each next approximation is obtained by using the Bellman equation
for \\(v\_{\pi}\\) as an update rule:

\\[v\_{k + 1}(s) = \max\_a \sum\_{s', r} p(s', r | s, a) \big[r + \gamma
v\_{k}(s')\big],\\]

for all \\(s \in \mathcal{S}\\).
In general the sequence \\(\\{v\_k\\}\_{k = 0}^{\infty}\\) can be shown to
converge to \\(v\_{\pi}\\).
This algorithm is called *iterative policy evaluation*.

When implementing the algorithm a usual way would be to use two arrays,
one for old values \\(v\_k(s)\\), one for new values \\(v\_{k + 1}(s)\\)
and the new values would be computed base on the old values.
But it is easier to implement it as in-place procedure with only one array
which also converges to \\(v\_{\pi}\\) and usually converges faster because
it has more recent data available sooner.

### Policy Improvement

TODO policy improvement.

### Policy Iteration

TODO policy iteration.

### Value Iteration

TODO value iteration.

## Monte Carlo Methods

TODO Monte Carlo methods.

### First-visit Monte Carlo Prediction

TODO first-visit Monte Carlo prediction.

### On-policy First-visit Monte Carlo Control

TODO on-policy first-visit Monte Carlo control.

### Off-policy Prediction via Importance Sampling

TODO off-policy prediction via importance sampling.

## Temporal-Difference Learning

TODO temporal-difference learning.

### Sarsa

TODO Sarsa.

### Q-learning

TODO Q-learning.

## Exploration and Exploitation

Reinforcement learning uses training information
which *evaluates* the actions taken
rather than *instructs* by giving correct actions.
Therefore the search for good behavior is needed.
Purely evaluative feedback indicates how good the action taken was
so depends entirely on the action.
Purely instructive feedback indicates the correct action independently of
action actually taken.

### A \\(k\\)-armed Bandit Problem

The \\(k\\)-armed bandit problem is named by analogy to
[slot machine][slot-machine].
An agent is repeatedly faced with a choice among \\(k\\) different actions
in *non-associative*, *stationary* setting (action taken only in one situation).
After each choice it receive a numerical reward from a stationary probability
distribution.
The objective is to maximize the expected total reward over some *time steps*.

[slot-machine]: https://en.wikipedia.org/wiki/Slot_machine (Slot Machine)

In the \\(k\\)-armed bandit problem each of the \\(k\\) actions has an expected reward
given that the action is selected (action's value).
The action selected at time step \\(t\\) is denoted as \\(A\_t\\)
and its reward as \\(R\_t\\).
The value of an arbitrary action \\(a\\) is the expected reward
given that \\(a\\) is selected:

\\[q\_\*(a) \equiv \mathbb{E}(R\_t | A\_t = a).\\]

If the value of each function was known it would be trivial to solve
the \\(k\\)-armed bandit problem.
But their are not certainly known although there might be estimates.
The estimated value of action \\(a\\) at time step \\(t\\) is \\(Q\_t(a)\\)
and it should be as close as possible to \\(q\_\*(a)\\).

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

\\[Q\_t(a) \equiv \frac{\sum^{t - 1}\_{i = 1} R\_i \cdot 1\_{A\_i = a}}
{\sum^{t - 1}\_{i = 1} 1\_{A\_i = a}},\\]

where \\(\mathbb{1}\_{\text{condition}}\\) is random variable indicator
that is \\(1\\) if condition is true else \\(0\\).
If the denominator is \\(0\\) then \\(Q\_t(a)\\) is defined as a default value
(for example \\(0\\)).
Moreover if the denominator goes to infinity
then \\(Q\_t(a)\\) converges to \\(q\_\*(a)\\).
This way of estimating action values is called *sample-average*.

The simplest action selection rule based on sample-average is to select the
action with highest estimated value (greedy action):

\\[A\_t \equiv \operatorname{argmax}\_a Q\_t(a).\\]


Greedy action selection always exploit current knowledge to maximize immediate
reward and spends no time exploring.
Simple modification is with probability \\(\varepsilon\\) select instead
randomly any action with equal probability independently of the action-value
estimates.
These which use near-greedy action selection rule are called
\\(\varepsilon\\)*-greedy* methods.
Advantage of these methods is in limit every action will be sampled
an infinite number of times thus all \\(Q\_t(a)\\) converge to \\(q\_\*(a)\\).

### Incremental Implementation

Sample average action value methods can be computed with constant memory
and constant per-time-step computation.
Let \\(R\_i\\) be the reward received after the \\(i\\)th selection *of this action*
and \\(Q\_n\\) denote the estimate of its action value
after it has been selected \\(n - 1\\) times
which can be written as:

\\[Q\_n \equiv \frac{R\_1 + R\_2 + \cdots + R\_{n - 1}}{n - 1}\\]

Obvious implementation would be to store all rewards
and compute the estimate when needed.
Then, the memory and computational requirements would grow linearly
as more rewards are seen.
That is not necessary because and incremental formula can be devised.
Given \\(Q\_n\\) and the \\(n\\)th reward \\(R\_n\\)
the new average is computed by:

\\[
\begin{aligned}
Q\_{n + 1} &= \frac{1}{n} \sum^n\_{i = 1} R\_i \\\\
          &= \frac{1}{n} (R\_n + \sum^{n - 1}\_{i = 1} R\_i) \\\\
          &= \frac{1}{n} (R\_n + (n - 1) \frac{1}{n - 1} \sum^{n - 1}\_{i = 1}
             R\_i) \\\\
          &= \frac{1}{n} (R\_n + (n - 1) Q\_n) \\\\
          &= \frac{1}{n} (R\_n + n Q\_n - Q\_n) \\\\
          &= Q\_n + \frac{1}{n} (R\_n - Q\_n).
\end{aligned}
\\]

Code for complete bandit algorithm with incrementally computed sample averages
and \\(\varepsilon\\)-greedy action selection:

```python3
# estimates of action values
Q = numpy.zeros(k)
# numbers of action's selections
N = numpy.zeros(k)

while True:
    # choose an action
    if numpy.random.rand() < 1 - epsilon:
	A = numpy.argmax(Q)
    else:
	A = numpy.random.randint(0, k)
    # reward received
    R = bandit(A)
    # update the estimated action value
    N[A] += 1
    Q[A] += (R - Q[A]) / N[A]
```

where `k` is number of actions
and `bandit(a)` is function which takes an action
and returns a corresponding reward.

The update rule above occurs frequently and its general form:

\\[\textit{new estimate} \gets \textit{old estimate} + \textit{step size}
\cdot (\textit{target} - \textit{old estimate}).\\]

The expression \\((\textit{target} - \textit{old estimate})\\) is the estimate's
*error* which is reduced by taking a step toward the \\(\textit{target}\\).
Note that the step-size parameter changes over time steps
(for \\(n\\)th reward for action \\(a\\) the step-size is \\(\frac{1}{n}\\)).
Further the step-size parameter is denoted as \\(\alpha\\)
or \\(\alpha\_t(a)\\).

### Tracking a Non-stationary Problem

Methods discussed above are not appropriate for *non-stationary* problems
in which the reward probabilities might change over time.
In such cases it makes sense to give more weight to recent rewards.
For example with constant step-size parameter:

\\[Q\_{n + 1} \equiv Q\_n + \alpha (R\_n - Q\_n),\\]

where \\(\alpha \in (0, 1]\\) is the constant step-size parameter.
\\(Q\_{n + 1}\\) is then a weighted average of past rewards
and the initial estimate \\(Q\_1\\):

\\[
Q\_{n + 1} = Q\_n + \alpha (R\_n - Q\_n)
           = (1 - \alpha)^n Q\_1
             + \sum\_{i = 1}^n \alpha (1 - \alpha)^{n - i} R\_i.
\\]

Note that
\\((1 - \alpha)^n + \sum\_{i = 1}^n \alpha (1 - \alpha)^{n - i} = 1\\)
and that the weight \\(\alpha (1 - \alpha)^{n - i}\\) of reward \\(R\_i\\)
depends on how many time steps ago it was observed (\\(n - i\\)).
The weight decays exponentially with respect to exponent \\(1 - \alpha\\)
therefore it is called *exponential recency-weighted average*.

### Optimistic Initial Values

All methods mentioned above are biased by their initial estimate \\(Q\_1(a)\\).
In the case of the sample-average methods the bias disappear after each
action is selected
but for method with constant \\(\alpha\\) is permanent though decreasing.
It is easy way to supply prior knowledge to the algorithm.

Initial values might be simple way to encourage exploration.
If they are set very optimistic whichever action is selected
its reward is less than the starting estimate
thus the agent will switch to other actions.
This simple technique is called *optimistic initial values*
and is only useful for stationary problems.

### Upper-confidence-bound Action Selection

\\(\varepsilon\\)-greedy action selection forces exploration through
non-greedy actions but without preference for those that are nearly greedy
or particularly uncertain.
*Upper confidence bound* (UCB) selects according to their potential for
being optimal taking into account how close their estimate are to being
maximal and estimates uncertainties by equation:

\\[
A\_t \equiv \operatorname{argmax}\_a \left[Q\_t(a) + c
\sqrt{\frac{\ln t}{N\_t(a)}}\right],
\\]

where \\(N\_t(a)\\) denotes the number of times that action \\(a\\) has been
selected prior to time step \\(t\\)
and \\(c \gt 0\\) controls the degree of exploration.
When \\(N\_t(a) = 0\\) then \\(a\\) is considered to be maximizing action.

The idea is that the square-root term measures the uncertainty in the
estimate of an action's value.
It tries to be upper bound for possible true value
and \\(c\\) determines the confidence level.

### Contextual Bandits

All methods above considered non-associative tasks in which the agent do not
associate different actions with different states.
The agent only tries to find best action
or track the best action as it changes over time.
However in a general reinforcement learning task agent's goal
it to learn a policy (mapping from situations to its best actions).

*Contextual bandits* tasks are intermediate between \\(k\\)-armed bandit
and the full reinforcement learning problems.
It is an *associative search* task as it involves both search for best actions
and *association* of theses actions with situation they are best for.
They involve learning a policy but each action affects only the immediate
reward.
If action affect the next state as well as the reward then it is a full
reinforcement learning problem.

## Approximate Solution Methods

In this post only tabular methods are discussed.
Methods were the value function can be represented as matrices.
But there is large area of approximate solution methods that scale up to large
or continuous state spaces.
These methods mainly include value-function approximation and policy gradient
methods.

Furthermore there is so called deep reinforcement learning which is branch of
field using non-linear function approximators.
Usually updating parameters with stochastic gradient descent.
Deep reinforcement learning has achieved remarkable successes as
[playing Atari 2600 games][atari], [mastering Go][go] or
[training an agent on many tasks][impala].

[atari]: https://deepmind.com/research/dqn/
[go]: https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge/
[impala]: https://deepmind.com/blog/impala-scalable-distributed-deeprl-dmlab-30/

Material to get into these areas are in the references.

## References

- Richard S. Sutton and Andrew G. Barto, [Reinforcement Learning: An Introduction][sutton2018]
- David Silver, [UCL Course on Reinforcement Learning][ucl-course]
- John Schulman, [Deep Reinforcement Learning][schulman-course]
- Denny Britz, [Learning Reinforcement Learning][britz-intro]

[schulman-course]: https://www.youtube.com/watch?v=aUrX-rP_ss4
[britz-intro]: http://www.wildml.com/2016/10/learning-reinforcement-learning/
