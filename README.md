# learning-to-learn-2
Reproduction of the experiment 3.1.2 on meta-learning described in the paper "Learning to Reinforcement Learn"

This repository works in a similar way as : https://github.com/ThomasLecat/learning-to-learn-1
The main difference comes from the adaptation of the code to the eleven armed bandit environment used in this experiment.

# Environment

The environment required in this experiment can be found here :
https://github.com/ThomasLecat/gym-bandit-environments.git
and is named "BanditElevenArmedWithIndex-v0"

# Experiment

In this setup, each bandit MDP has 11 arms which always give a reward. Among the ﬁrst ten arms, nine ”non-target” arms give a reward of 1.1, and one ”target-arm” gives a reward of 5. The eleventh arm is an ”informative arm” whose reward is worth a tenth of the index of the target-arm. For example, if the reward of the informative arm is 0.1, then the target arm is the ﬁrst arm ; if it is 0.2, the the target arm is the second arm, etc. Each trial is 5 steps long, so the optimal strategy is to pull the informative arm ﬁrst, infer from the reward obtained the index of the target arm, then pull this arm for the last four steps.

The purpose of the experiment is to study whether the agent is capable of learning the link between the eleventh arm reward value and the index of the optimal arm.

Bandits environments are stateless but the training is organised in fake episodes during which the internal state of the LSTM is kept. The length of the fake episode is 5 trials in this experiment but can be set to a different value with the parameter -n (see "How to" section)

Please refer to the original paper for a detailed description of the experiment.

# Specifics of this code

 Similarly to the repository learning-to-learn-1, the input of the agent's network when using bandit environments is the last action, last reward and timestep in the episode stacked together.

 The last action and last reward both are translated into one-hot vectors before being fed to the network. The code in this repository is specifically designed to take care of this translation, which makes it unusable with other bandit environments than BanditElevenArmedWithIndex-v0.

 # How to

 The two callable scripts are train.py and test.py.
 Both take the same arguments as input. Among them, we can find:

 * -w : number of workers working in parallel
 * -m : to resample environments at the beginning of each episode
 * -n : number of trials in each  episode (default is 5)
 * -lr : learning rate (default is 1e-4)

 Meta-learning is performed as soon as the -m argument is present. In that case, the environment is recreated at the beginning of each episode. As the index of the target arm is sampled randomly, the configuration of the MDP changes from one episode to another. This results in training (and / or testing) the agent on a set of MDPs instead of a single one.


 # Hyperparameters

 The hyperparameters are somewhat spread across the code... Here's the location of some of them :
 * number of training steps : num_global_step in worker.py run function.
 * number of testing steps : num_test_step next to num_global_step in worker_test.py
 * discount factor : file A3C.py, class A3C, method process : change gamma value in the line : "batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)"
 * number of steps in each rollout (t_max in the original A3C paper): file A3C.py, class A3c, method __init__, change the value in line : "num_local_step = 5"
 * learning rate : change by adding the argument -lr <value> when calling python train.py (see section above)
 * number of trials in a fake episode for bandit environments : change by adding the argument -n <value> when calling python trian.py (see section above)

 # Dependencies

 * Python 2.7 or 3.5
 * [Golang](https://golang.org/doc/install)
 * [six](https://pypi.python.org/pypi/six) (for py2/3 compatibility)
 * [TensorFlow](https://www.tensorflow.org/) 0.12
 * [tmux](https://tmux.github.io/) (the start script opens up a tmux session with multiple windows)
 * [htop](https://hisham.hm/htop/) (shown in one of the tmux windows)
 * [gym](https://pypi.python.org/pypi/gym)
 * gym[atari]
 * libjpeg-turbo (`brew install libjpeg-turbo`)
 * [universe](https://pypi.python.org/pypi/universe)
 * [opencv-python](https://pypi.python.org/pypi/opencv-python)
 * [numpy](https://pypi.python.org/pypi/numpy)
 * [scipy](https://pypi.python.org/pypi/scipy)

 # Installation

 ```
 conda create --name learning-to-learn-2 python=3.5
 source activate learning-to-learn-2

 brew install tmux htop cmake golang libjpeg-turbo      # On Linux use sudo apt-get install -y tmux htop cmake golang libjpeg-dev

 pip install "gym[atari]"
 pip install universe
 pip install six
 pip install tensorflow
 conda install -y -c https://conda.binstar.org/menpo opencv3
 conda install -y numpy
 conda install -y scipy
 ```


 Add the following to your `.bashrc` so that you'll have the correct environment when the `train.py` script spawns new bash shells
 ```source activate learning-to-learn-2```

 # Example

 `python train.py --num-workers 2 --env-id BanditElevenArmedWithIndex-v0 --log-dir /tmp/banditEleven`

 The code will launch the following processes:
 * worker-0 - a process that runs policy gradient
 * worker-1 - a process identical to process-1, that uses different random noise from the environment
 * ps - the parameter server, which synchronizes the parameters among the different workers
 * tb - a tensorboard process for convenient display of the statistics of learning
