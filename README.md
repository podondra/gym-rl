# Reinforcement Learning in gym

This work aims to use reinforcement learning to solve some [gym]
environments.

[gym]: https://github.com/openai/gym (gym GitHub repository)

It also uses OpenAI [gym-gridworlds] environments
that implemented myself.

[gym-gridworlds]: https://github.com/podondra/gym-gridworlds

## Setup

	$ git clone https://github.com/podondra/gym-rl
	$ git submodule init
	$ git submodule update

## Checklist

- [ ] write an introductory blog post about reinforcement learning
      with Python examples and publish it on https://podondra.cz
- [x] make a survey of reinforcement learning algorithms
      which are applicable to gym's problems
- [ ] implement some algorithms from survey and apply them to gym environments
    - [ ] some [Atari 2600][atari] environments
    - [ ] *HumanoidFlagrun* from [Roboschool]
    - [ ] *Kick and Defend* [competitive self-play environment][self-play]
- [x] use Python 3.4 or higher
- [ ] use [pytest] to make sure algorithm are correctly implemented
- [x] for implementation use [NumPy], (and [pandas])

[roboschool]: https://blog.openai.com/roboschool/ (Roboschool OpenAI Blog)
[self-play]: https://github.com/openai/multiagent-competition
             (Competitive Multi-Agent Environments)
[pytest]: https://docs.pytest.org/en/latest/ (pytest Documentation)
[numpy]: http://www.numpy.org/ (NumPy Documentation)
[pandas]: https://pandas.pydata.org/ (Python Data Analysis Library)
[atari]: https://en.wikipedia.org/wiki/Atari_2600 (Atari 2600 Wikipedia)

## Blog Post

The blog post will be available on [podondra site][site].

[site]: https://podondra.cz

## Algorithms for OpenAI gym

This [survey] list algorithms which can be applied to OpenAI gym environment
such as Atari 2600 game, HumanoidFlagrun or Kick and Defend.

[survey]: surveys/gym-algorithms-survey.md

## References

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [David Silver: Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [John Schulman: Deep Reinforcement Learning](https://www.youtube.com/watch?v=aUrX-rP_ss4)
- [Dynamic Programming and Optimal Control](http://web.mit.edu/dimitrib/www/dpchapter.html)
- [Learning Reinforcement Learning](http://www.wildml.com/2016/10/learning-reinforcement-learning/)
- [Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748)
- [Deep Reinforcement Learning: A Tutorial](https://web.archive.org/web/20160830014637/https://gym.openai.com/docs/rl)
- [OpenAI Baselines](https://github.com/openai/baselines)
- [Roboschool](https://github.com/openai/roboschool)
- [OpenAI Gym Paper](https://arxiv.org/abs/1606.01540)
- [OpenAI Gym Tutorial](https://gym.openai.com/docs/)
- [DQN Paper in Nature](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- [Learning from Delayed Rewards](http://www.cs.rhul.ac.uk/~chrisw/thesis.html)
