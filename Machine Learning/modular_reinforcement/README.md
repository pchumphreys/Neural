# Modular reinforcement framework

This is my code for reinforcement learning. The main notebook is [Reinforcement Learning.ipynb](Reinforcement%20Learning.ipynb)

The code follows a standard structure of defining an [agent](base_agent.py) and a [runner](runner.py) that runs the experiment.

There are a bunch of helper functions for e.g. grid searching, doing ablation studies in [reinforcement_expm.py](reinforcement_expm.py)
## Policy gradient methods

I have currently implemented two policy gradient methods: [A2C](a2c_agent.py) and a version of [PPO](ppo_agent.py) (no guarantees)

## Q learning methods

The code for the [DQN agent](dqn_agent.py) is pretty flexible. In particular, it can implement a discrete version of the soft-actor-critic proposed in https://arxiv.org/pdf/1801.01290.pdf


