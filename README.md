# Reinforcement Learning in Gym

This work aims to use reinforcement learning to solve some [gym]
environments.

[gym]: https://github.com/openai/gym (gym GitHub repository)

## Checklist

- [ ] write an introductory blog post about reinforcement learning
      with Python examples and publish it on https://podondra.cz
- [ ] make a survey of reinforcement learning algorithms
      which are applicable to gym's problems
- [ ] implement some algorithms from survey and apply them to gym environments
    - [ ] some [Atari 2600][atari] environments
    - [ ] *HumanoidFlagrun* from [Roboschool]
    - [ ] *Kick and Defend* [competitive self-play environment][self-play]
- [ ] use Python 3.4 or higher
- [ ] use [pytest] to make sure algorithm are correctly implemented
- [ ] for implementation use [NumPy] and [pandas]

[roboschool]: https://blog.openai.com/roboschool/ (Roboschool OpenAI Blog)
[self-play]: https://github.com/openai/multiagent-competition
             (Competitive Multi-Agent Environments)
[pytest]: https://docs.pytest.org/en/latest/ (pytest Documentation)
[numpy]: http://www.numpy.org/ (NumPy Documentation)
[pandas]: https://pandas.pydata.org/ (Python Data Analysis Library)
[atari]: https://en.wikipedia.org/wiki/Atari_2600 (Atari 2600 Wikipedia)

## Algorithms for Gym's Problems

[OpenAI Gym][gym] is an open-source toolkit for developing and comparing
reinforcement learning algorithms.

Try Cross-entropy method to problem
proposed in [Deep Reinforcement Learning Tutorial][deep-rl],
[Learning Tetris Using the Noisy Cross-Entropy Method][tetris]
and [The Cross-Entropy Method Optimizes for Quantiles][xentropy].

[deep-rl]: https://web.archive.org/web/20160830014637/https://gym.openai.com/docs/rl
[tetris]: http://ie.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
[xentropy]: http://proceedings.mlr.press/v28/goschin13.pdf

## Blog Post

The blog post in available on [podondra site][8].

[8]: https://podondra.cz/introduction-to-reinforcement-learning.html
     (Introduction to Reinforcement Learning Blog Post)

## References

- Reinforcement Learning: An Introduction,
  http://incompleteideas.net/book/bookdraft2017nov5.pdf
- Learning from Delayed Rewards,
  http://www.cs.rhul.ac.uk/~chrisw/thesis.html
- Dynamic Programming and Optimal Control,
  http://web.mit.edu/dimitrib/www/dpchapter.html
- Learning Reinforcement Learning,
  http://www.wildml.com/2016/10/learning-reinforcement-learning/
- Emergent Complexity via Multi-Agent Competition,
  https://arxiv.org/abs/1710.03748
- Deep Reinforcement Learning: A Tutorial,
  https://web.archive.org/web/20160830014637/https://gym.openai.com/docs/rl
- OpenAI Baselines,
  https://github.com/openai/baselines
- Roboschool,
  https://github.com/openai/roboschool
- OpenAI Gym Paper,
  https://arxiv.org/abs/1606.01540
- OpenAI Gym Tutorial,
  https://gym.openai.com/docs/
- DQN Paper in Nature,
  https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
