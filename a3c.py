from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, LSTMPolicyBandit
from envs import create_env
import six.moves.queue as queue
import scipy.signal
import threading
import gym
import distutils.version
import pdb # debugger
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def discount(x, gamma): # discount([1,1,1,1], 0.5) gives [1.875,1.75, 1.5, 1]
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states) # np.asarray : convert to an array
    batch_a = np.asarray(rollout.actions)
    #pdb.set_trace()
    #print("rollout.last_action.shape : ", np.asarray(rollout.last_action).shape, "\nrollout.actions.shape:", np.asarray(rollout.actions).shape)
    aPred_t = np.append(rollout.last_action, batch_a[:-1], axis = 0)
    rewards = np.asarray(rollout.rewards)
    onehot_rewards = np.asarray(rollout.onehot_rewards)
    rPred_t = np.reshape(np.append(rollout.last_reward, rewards[:-1]), (-1,1))
    onehot_rPred_t = np.reshape(np.append(rollout.onehot_last_reward, onehot_rewards[:-1]), (-1,13))
    batch_l = np.asarray(rollout.length)
    vPred_t = np.asarray(rollout.values + [rollout.r])
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_return = discount(rewards_plus_v, gamma)[:-1] # compute the n-step estimated return obtained after visiting each state (with n: number of steps succeeding the current one in the rollout) ; gamma to remove the v = rollout.r at the end
    delta_t = rewards + gamma * vPred_t[1:] - vPred_t[:-1] # array of the TD residual at each time step : r_{t+1} + \gamma*v(s_{t+1}) - v(s_t)
    # (original comment) this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_) # array of the estimations of the advantage at each time step

    features = rollout.features[0]
    return Batch(batch_si, batch_a, aPred_t, rPred_t, onehot_rPred_t, batch_adv, batch_return, rollout.terminal, features, batch_l)

Batch = namedtuple("Batch", ["si", "a", "aPred_t", "rPred_t", "onehot_rPred_t", "adv", "return_", "terminal", "features", "l"])

class PartialRollout(object):
    """ (original comment)
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
    def __init__(self):
        self.last_reward = [0] # last reward obtained before this partial rollout (remains 0 if beginning of the episode)
        self.last_action = [0] # the last action taken before this partial rollout begins (remains [0...0] if beginning of the episode)
        self.states = []
        self.actions = []
        self.rewards = []
        self.onehot_rewards = []
        self.values = []
        self.r = 0.0 # value function of the last state of the rollout (remains 0 if terminal)
        self.terminal = False
        self.features = []
        self.length =[]

    def add(self, state, action, reward, onehot_reward, value, terminal, features, length):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.onehot_rewards += [onehot_reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.length += [[length]]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.onehot_rewards.extend(other.onehot_rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        self.length.extend(other.length)

    def initialReAc(self, last_reward, onehot_last_reward, last_action):
        self.last_reward = [last_reward]
        self.onehot_last_reward = [onehot_last_reward]
        self.last_action = [last_action]

    def __repr__(self):
        return "states: {} states with shape {} \nactions: {} actions \nrewards: {} \nvalues: {}\nr: {} \nlength: {}\nterminal: {}\n".format(len(self.states), np.asarray(self.states[0]).shape, len(self.actions), self.rewards, self.values, self.r, self.length, self.terminal)

class RunnerThread(threading.Thread):
    """ (original comment)
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
    def __init__(self, env, policy, num_local_steps, visualise, meta, task, remotes, num_trials):
        threading.Thread.__init__(self) # mandatory stuff for threads, (see openclassroom for tutorial)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True # By setting the threads as daemon threads, when your program quits, any daemon threads are killed automatically. (so you can let them run and forget about them).
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.meta = meta
        self.task=task
        self.remotes = remotes
        self.num_trials = num_trials

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
	# yield a generator, but nothing is executed at this line :
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise, self.meta, self.task, self.remotes, self.num_trials)
        while True:
	    # the execution of the function "env_runner" happens here, which is why the function is executed indefinitely, not only once :
            self.queue.put(next(rollout_provider), timeout=600.0) # *(2)
	    # "num_local_steps" of experience are returned and put into the queue (unless the episode ends before "num_local_steps" steps)

def onehot(isBanditEnvironment, env, reward):
	if isBanditEnvironment:
		if 'Eleven' in env.env.spec.id:
			onehot_reward = np.zeros(13)
			if reward == 5:
				onehot_reward[-1] = 1
			elif reward == 1.1:
				onehot_reward[-2] = 1
			else: # informative arm
				onehot_reward[int(10*reward)] = 1
			return onehot_reward
		elif 'Two' in env.env.spec.id:
			onehot_reward = np.zeros(2)
			if reward == 0:
				onehot_reward[0] = 1
			else:
				onehot_reward[1] = 1
			return onehot_reward
	else: # if not bandit environment, return the non one-hot reward.
		return reward


def env_runner(env, policy, num_local_steps, summary_writer, render, meta, task, remotes, num_trials): # render = visualize
    """ (original comment)
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    isBanditEnvironment = "Bandit" in env.env.spec.id
    last_state = env.reset()
    last_features = policy.get_initial_features() # list of 2 lists of zeros : [c_init, h_init] = [[0...0],[0...0]]
    last_reward = 0
    onehot_last_reward = onehot(isBanditEnvironment, env, last_reward)
    last_action = np.zeros(env.action_space.n) # last_action and action are one-hot vectors
    length = 0  #  length of the episode (so far)
    rewards = 0 # cummulatic reward (with no discount) for the episode.

    while True: # the "while True" is necessary because the generator will be called several times by the "_run method" of the RunnerThread "trainer.runner" (see above)
        terminal_end = False
        rollout = PartialRollout() # creates an empty rollout
        rollout.initialReAc(last_reward, onehot_last_reward, last_action)

        # beginning of the rollout
        for _ in range(num_local_steps):
            # Feedforward on the network : gives the network the observation ("last_state") and the internal state ("last_features") and gets the action prob vector, value func and new internal state
            fetched = policy.act(last_state, onehot_last_reward, last_action, length, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # Take action on the environnement ; argmax to convert from one-hot to index
            state, reward, terminal, info = env.step(action.argmax())
            onehot_reward = onehot(isBanditEnvironment, env, reward)
            print(action.argmax())
            # show the game running in a window
            if render:
                env.render()

            # collect the experience in the partial rollout
            rollout.add(last_state, action, reward, onehot_reward, value_, terminal, last_features, length)

            # prepare next step
            length += 1
            last_state = state
            last_features = features
            last_reward = reward
            onehot_last_reward = onehot_reward
            last_action = action
            rewards += reward

	       # for tensorbard :
            if info:
                summary = tf.Summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

            # Check if we have finished an episode or reached the time limit for that game.
            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps') # value given by gym, maximum length of an episode for that game (time limit) (is "None" for Bandits environments)

            if isBanditEnvironment:
                terminal = False # otherwise with Bandit environment, the episode finishes after each step
                timestep_limit = num_trials # to create a fake episode length for bandits environments

            # If we did, we reinitialise the environment, the network internal state, "rewards" and "lenght" and break the loop
            if terminal or (length >= timestep_limit and not timestep_limit==None):
                terminal_end = True
                if meta:
                    env = create_env(env.env.spec.id, str(task), remotes, num_trials) # samples a new environment with the same id as the old one.
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'): # ???
                    last_state = env.reset()
                last_features = policy.get_initial_features() # reinitialise the internal state of the LSTM
                print("Episode finished. Sum of rewards: {}. Length: {}".format(rewards, length))
                length = 0
                rewards = 0
                last_reward = 0
                onehot_last_reward = onehot(isBanditEnvironment, env, last_reward)
                last_action = np.zeros(env.action_space.n)
                break # breaks the for loop to finish the rollout early

            # If we didn't, go back to the beggining of the loop to take the next step unless we have reached num_local_steps steps

        # if the partial rollout is not terminal, we add the value of the last state (used to compute the estimation of the return, maybe for other purposes)
        if not terminal_end:
            rollout.r = policy.value(last_state, onehot_last_reward, last_action, length, *last_features) # the value of the last state is computed twice, since it will be computed at the beggining of then next partial rollout too.

        yield rollout # yield is a keyword that is used like return, except the function will return a generator *(1)


class A3C(object):
    def __init__(self, env, task, visualise, learning_rate, meta, remotes, num_trials, total_num_steps):
        """ (original comment)
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.env = env
        self.task = task
        self.remotes = remotes
        self.learning_rate = learning_rate
        self.num_trials = num_trials
        num_local_steps = 5 # t_max in the A3C paper: number of steps in the rollouts
        isBanditEnvironment = "Bandit" in env.env.spec.id # boolean variable, is True if the environment is a Bandit environment
        if isBanditEnvironment:
        	if 'Two' in env.env.spec.id:
        		reward_range = 2
        	elif 'Eleven' in env.env.spec.id:
        		reward_range = 13

        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                if isBanditEnvironment:
                    self.network = LSTMPolicyBandit(env.observation_space.shape, env.action_space.n, reward_range)
                else:
                    self.network = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False) # Cree un compteur global et l'initialise a zero, saud si on reprend un training existant 

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                if isBanditEnvironment:
                    self.local_network = pi = LSTMPolicyBandit(env.observation_space.shape, env.action_space.n, reward_range)
                else:
                    self.local_network = pi = LSTMPolicy(env.observation_space.shape, env.action_space.n)
                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac") # action, will contain the list of the action vectors at each step of the rollout ; placeholder called by A3C's process function in trainer.process(sess) in worker.py
            self.adv = tf.placeholder(tf.float32, [None], name="adv") # advantage, wil contain the list of the advantages at each step of the rollout ; placeholder called by A3C's process function in trainer.process(sess) in worker.py
            self.return_ = tf.placeholder(tf.float32, [None], name="return_") # return, wil contain the return obtained after visiting each of steps in the rollout ; placeholder called by A3C's process function in trainer.process(sess) in worker.py

            log_prob_tf = tf.nn.log_softmax(pi.logits) # the log probability of each action log(\pi(a|s))
            prob_tf = tf.nn.softmax(pi.logits) # the probability of each action \pi(a|s)

            # (original comment) the "policy gradients" loss:  its derivative is precisely the policy gradient. Notice that self.ac is a placeholder that is provided externally.
            #
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.return_)) # why not taking the sum of the squared values of self.adv directly ?
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
            beta_entropy = (float(1)/total_num_steps)*tf.cast(tf.constant(total_num_steps)-self.global_step, tf.float32)

            bs = tf.to_float(tf.shape(pi.x)[0]) # bs = batch size = number of steps in the rollout

            self.loss = pi_loss + (0.05 * vf_loss) - (beta_entropy * entropy) # why 0.5 when we already put 0.5 in the definition of vf_loss ?

            # (Original comment)
            # num_local_steps represents the number of timesteps we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate on the one hand;  but on the other hand, we get less frequent parameter updates, which slows down learning.
            # In this code, we found that making local steps be much smaller than 20 makes the algorithm more difficult to tune and to get to work.
            # (My comment):
            # The original A3C paper uses num_local_step = 5 on Atari games, but it uses an action repeat of 4 (not present here), so the network is updates every 20 frames, as in the original universe-starter-agent

            self.runner = RunnerThread(env, pi, num_local_steps, visualise, meta, task, remotes, num_trials) # Objet de la classe RunnerThread definie plus haut. 20 is the maximum number of steps in a partial rollout.

            # computes the gradient of the loss function:
            grads = tf.gradients(self.loss, pi.var_list)

            # tensorboard:
            if use_tf12_api:
                tf.summary.scalar("model/policy_loss", pi_loss / bs)
                tf.summary.scalar("model/value_loss", vf_loss / bs)
                tf.summary.scalar("model/entropy", entropy / bs)
                tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
                tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
                if not isBanditEnvironment:
                    tf.summary.image("model/state", pi.x)
                self.summary_op = tf.summary.merge_all()
            else:
                tf.scalar_summary("model/policy_loss", pi_loss / bs)
                tf.scalar_summary("model/value_loss", vf_loss / bs)
                tf.scalar_summary("model/entropy", entropy / bs)
                tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
                tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
                if not isBanditEnvironment:
                    tf.image_summary("model/state", pi.x)
                self.summary_op = tf.merge_all_summaries()

            # Create a list of (gradient, variable) pairs to feed into the Adam Optimizer (each variable will then be updated according to the paired gradient)
            grads, _ = tf.clip_by_global_norm(grads, 40.0) # ?
            grads_and_vars = list(zip(grads, self.network.var_list))

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)]) # remplace les valeurs de pi.var_list par ceux de self.network.var_list (execute dans la function "process")

            # updates the global counter: adds (and assign) tf.shape(pi.x)[0] to the value of the variable self.global_step (initialise a zero), and inc_step takes this updtated value:
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0]) # on incremente le compteur global du nombre de steps contenus dans le rollout (= batch size) ; appele par la fonction process
            self.inc_step = inc_step # so that we can call it directly from the inc_global_step method 

            # each worker has a different set of adam optimizer parameters
            opt = tf.train.AdamOptimizer(self.learning_rate) # the default learning rate is 1e-4. This value with the argument -lr <new_value>
            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step) # tf.group creates an op that groups multiple operations (here, two operations)
            self.summary_writer = None
            self.local_steps = 0

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer) # lance l'execution de la methode "run" sur le thread "runner" qui est du type RunnerThread
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        Original comment : self explanatory - take a rollout from the queue of the thread runner.
        My comment : If there is more than 1 rollout in the queue, build and return a batch with all the rollout from the queue (unless one of them is terminal).
        99 % of the time, there is only one rollout
        """
        q = self.runner.queue
        rollout = q.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(q.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner, and updates the parameters.
The update is then sent to the parameter server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue() # takes all the partial rollouts in the queue (unless one is terminal: it stops after the terminal one)
        # print("in process, rollout is :\n", rollout)
        batch = process_rollout(rollout, gamma=0.8, lambda_=1.0) # batch = Batch(batch_si, batch_a, batch_adv, batch_return, rollout.terminal, features) with batch_r : return (n-step)

        # tensorboard:
        should_compute_summary = self.task == 0 and self.local_steps % 11 == 0
        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]


        feed_dict = {
            self.local_network.x: batch.si,
            self.local_network.last_reward: batch.onehot_rPred_t,
            self.local_network.last_action: batch.aPred_t,
            self.local_network.local_time: batch.l,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.return_: batch.return_,
            self.local_network.state_in[0]: batch.features[0],
            self.local_network.state_in[1]: batch.features[1],
        }
        fetched = sess.run(fetches, feed_dict=feed_dict)
        # print("step_size in process: ", fetched[-1]) # add "+[self.local_network.step_size]" after fetches to see the size of the batches (= number of steps in the rollout)

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
            self.summary_writer.flush()

        self.local_steps += 1

    def inc_global_step(self, sess):
        """
        This function is made to keep incrementing the globa step counter during the test phase. (During the training phase, the counter is incremented by the process method)
        """
        rollout = self.pull_batch_from_queue()
        batch_si = np.asarray(rollout.states) # replaces the process_rollout phase
        feed_dict = {self.local_network.x: batch_si}
        sess.run(self.inc_step, feed_dict = feed_dict)

'''
*(1) yield is a keyword that is used like return, except the function will return a generator. To master yield, you must understand that when you call the function, the code you have written in the function body does not run. The function only returns the generator object. Then, your code will be run each time the "for" uses the generator.

*(2) The timeout variable exists because apparently, if one worker dies, the other workers won't die with it, unless the timeout is set to some large number. This is an empirical observation.

*(3) [::-1] pour une array ou une string renverse l'ordre : [1,2,3][::-1] = [3,2,1]. Que se passe-t-il ici ???

'''
