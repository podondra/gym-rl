Title: Introduction to Reinforcement Learning
Date: 2018-01-04 08:03
Category: reinforcement learning
Tags: reinforcement learning

In this post I would like to introduce reinforcement learning
by shortening some chapters from book
[Reinforcement Learning: An Introduction][sutton2018] by Sutton and Barto.
Reinforcement learning is currently producing exciting results
(see [OpenAI] and [DeepMind])
and in my opinion leads to general artificial intelligence.

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
Therefore, it defines the learning agent's behavior at a given time.

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

The last element is a *model of environment* which mimics the behavior
of the environment or allows inferences about how the environment will behave.
They are used for estimating future situations before they are actually
experienced (planning).

Most of the reinforcement learning methods are concerned with estimating value
functions (except evolutionary methods).

## Multi-armed Bandits

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
and it should be as close as possible to \\(q_*(a)\\).

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

\\[Q_t(a) \equiv \frac{\sum^{t - 1}\_{i = 1} R\_i \cdot 1\_{A\_i = a}}
{\sum^{t - 1}_{i = 1} 1\_{A\_i = a}},\\]

where \\(\mathbb{1}\_{\text{condition}}\\) is random variable indicator
that is \\(1\\) if condition is true else \\(0\\).
If the denominator is \\(0\\) then \\(Q_t(a)\\) is defined as a default value
(for example \\(0\\)).
Moreover if the denominator goes to infinity
then \\(Q_t(a)\\) converges to \\(q\_\*(a)\\).
This way of estimating action values is called *sample-average*.

The simplest action selection rule based on sample-average is to select the
action with highest estimated value (greedy action):

\\[A_t \equiv \operatorname{argmax}_a Q_t(a).\\]


Greedy action selection always exploit current knowledge to maximize immediate
reward and spends no time exploring.
Simple modification is with probability \\(\varepsilon\\) select instead
randomly any action with equal probability independently of the action-value
estimates.
These which use near-greedy action selection rule are called
\\(\varepsilon\\)*-greedy* methods.
Advantage of these methods is in limit every action will be sampled
an infinite number of times thus all \\(Q_t(a)\\) converge to \\(q\_\*(a)\\).

### Incremental Implementation

Sample average action value methods can be computed with constant memory
and constant per-time-step computation.
Let \\(R_i\\) be the reward received after the \\(i\\)th selection *of this action*
and \\(Q_n\\) denote the estimate of its action value
after it has been selected \\(n - 1\\) times
which can be written as:

\\[Q\_n \equiv \frac{R\_1 + R\_2 + \cdots + R\_{n - 1}}{n - 1}\\]

Obvious implementation would be to store all rewards
and compute the estimate when needed.
Then, the memory and computational requirements would grow linearly
as more rewards are seen.
That is not necessary because and incremental formula can be devised.
Given \\(Q_n\\) and the \\(n\\)th reward \\(R_n\\)
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

## Finite Markov Decision Processes

Now the finite Markov decision processes (finite MDPs) are introduced.
It involves choosing different actions in different situations.
MDPs are a classical mathematical formalization of sequential decision making
where action influence all future rewards so MDPs involve delayed reward
and the need to trade off between immediate and delayed reward.
In MDPs the value \\(q\_\*(s, a)\\) of each action \\(a\\) in state \\(s\\)
or the value \\(v\_\*(s)\\) of each state given optimal action selections.

### The Agent-environment Interface

MDP is framework for a problem of learning from interaction to achieve a goal.
The learner is an *agent* interacting with an *environment*
which is everything outside it.
The agent select actions and the environment responses to these actions
and present new situations to the agent.
One of the responses from environment are the rewards
which the agent tries to maximize over time through actions.

Agent and environment interact at discrete time steps \\(t = 0, 1, 2, \dots\\).
At each time step \\(t\\) the agent receives representation of the environment
*state* \\(S\_t \in \mathcal{S}\\) and based on it selects an action \\(A\_t \in
\mathcal{A}(s)\\).
The next step the agent gets a numerical *reward* \\(R\_{t + 1} \in \mathcal{R}
\subset \mathbb{R}\\) and find itself in a new state \\(S\_{t + 1}\\).
This loop in MDP can be unrolled into sequence or *trajectory*:

\\[S\_0, A\_0, R\_1, S\_1, A\_1, R\_2, S\_2, A\_2, R\_3, \dots\\] 

In a *finite* MDP sets of states \\(\mathcal{S}\\),
actions \\(\mathcal{A}\\) and rewards \\(\mathcal{R}\\) are finite.
Therefore random variables \\(R\_t\\) and \\(S\_t\\) have well defined
discrete probability distributions dependent only on previous state and action.
That gives for \\(s' \in \mathcal{S}\\) and \\(r \in \mathcal{R}\\)
a probability of occurrence at time \\(t\\) given preceding state and action:

\\[p(s', r | s, a) \equiv \mathbb{P}(S\_t = s', R\_t = r | S\_{t - 1} = s,
A\_{t - 1} = a),\\]

for all \\(s', s \in \mathcal{S}\\), \\(r \in \mathcal{R}\\)
and \\(a \in \mathcal{A}(s)\\).
The function \\(p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times
\mathcal{A} \to [0, 1]\\) is ordinary deterministic function of four
arguments.
The probabilities give by the function \\(p\\) completely characterize the
dynamics of a finite MDP
and other useful quantities can be derived from it
as *state-transition probabilities*:

\\[p(s' | s, a) \equiv \mathbb{P}(S\_t = s' | S\_{t - 1} = s,
A\_{t - 1} = a) = \sum\_{r \in \mathcal{R}} p(s', r | s, a),\\]

expected reward for each state and action as function
\\(r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}\\):

\\[r(s, a) \equiv \mathbb{E}(R\_t | S\_{t - 1} = s, A\_{t - 1} = a)
= \sum\_{r \in \mathcal{R}} r \sum\_{s' \in \mathcal{S}} p(s', r | s, a)\\]

or expected reward for state, action and next-state triples
\\(r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}\\):

\\[r(s, a, s') \equiv \mathbb{E}(R\_t | S\_{t - 1} = s, A\_{t - 1} = a,
S\_t = s') = \sum\_{r \in \mathcal{R}} r
\frac{p(s', r | s, a)}{p(s' | s, a)}.\\]

The MDP framework is abstract and very flexible and can describe many different
problems in many different ways.
It is an abstraction of the problem of goal-directed learning from interaction.
MDP reduces any such problem to three signals passing between an agent
and its environment:

- signal to make choices, actions,
- signal to represent the basis for choices, states and
- signal to define agent's goal, rewards.

### Goals and Rewards

The agent's goal is to maximize the total amount of reward
\\(R\_t \in \mathbb{R}\\) it receives.
This means maximizing cumulative reward in the long run
not only the immediate reward.
This idea is given as *reward hypothesis*
that all of what we mean by goals and purposes can be well thought of as
the maximization of the expected value of the cumulative sum of a received
scalar signal.

Consider a game of Go.
An agent might get a reward of \\(-1\\) for losing the game
and \\(1\\) for winning the game.
The agent always learns to maximize its reward
so it is important to provide reward in such way that the agent
when maximizing its reward will also achieve goal given.
The reward signal is not the place to impart
*how* the agent such achieve something but rather *what* to achieve.
For example there such be no rewards during the game of Go for some
intermediate accomplishments as the agent might learn to achieve theses subgoals
without learning to win the game.
