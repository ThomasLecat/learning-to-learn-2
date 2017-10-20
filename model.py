import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils.version
import pdb # debugger
use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # (original comment) there are "num input feature maps * filter height * filter width" inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # (original comment)
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # (original comment) initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)

class LSTMPolicyBandit(object):
    def __init__(self, ob_space, ac_space, r_range):
        self.last_reward = tf.placeholder(tf.float32, shape=[None, r_range], name="last_reward")
        self.last_action = tf.placeholder(tf.float32, shape=[None, ac_space], name="last_action") # the one-hot vector is of size ac_space = env.action_space.n
        self.local_time = tf.placeholder(tf.float32, shape=[None, 1], name="local_time")
        self.x = tf.placeholder(tf.float32, [None]+list(ob_space))

        input = tf.concat([self.last_reward, self.last_action, self.local_time], 1) 
        input = tf.expand_dims(input, [0]) # the first 1's is batch_size (fake)

    	size = 48
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1] 

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, input, initial_state=state_in, sequence_length=step_size,
            time_major=False)

        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size]) # x now is the output of the LSTM

        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.1))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1]) # reshape produces a 1-D vector here
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, r, a, length, c, h): 
        sess = tf.get_default_session()          
        return sess.run([self.sample, self.vf] + self.state_out,
                    {self.x: [ob], self.last_reward: [r], self.last_action: [a], self.local_time: [[length]], self.state_in[0]: c, self.state_in[1]: h})
        # Note : on met un + entre [self.sample, self.vf] et self.state_out car self.state_out est deja une liste en elle meme (donc on concatene)

    def value(self, ob, r, a, length, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf,
                        {self.x: [ob], self.last_reward: [r], self.last_action: [a], self.local_time: [[length]], self.state_in[0]: c, self.state_in[1]: h})[0]
    # function used to compute the value of the last state of a partial rollout when this one isn't terminal


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space):
        # in python, "+" concatenates the lists. For Pong, it will give: [None, 210, 160, 3]. We add None because the tf.nn.conv2d function requires a 4-D tensor [batch, in_height, in_width, in_channels]
        # batch = 1 when the agent runs on the environment, batch = "number fo steps in the rollout" when called by the function process. This also are the values of all the "?" in the the comments below
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        # shape of x is (?, 42, 42, 1) for pong (image output is rescaled to 42x42x1 in the create_atari function of envs.py)

        ''' Unused (for the moment), required to avoid errors '''
        self.last_reward = tf.placeholder(tf.float32, shape=[None, 1], name="last_reward")
        self.last_action = tf.placeholder(tf.float32, shape=[None, ac_space], name="last_action") # the one-hot vector is of size ac_space = env.action_space.n
        self.local_time = tf.placeholder(tf.float32, shape=[None, 1], name="local_time")

        # x goes through CNN:
        for i in range(4):
            x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # shape of x is (?, 3, 3, 32) after CNN, for pong

        # (original comment) introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0]) # x is a numpy array of shape [1,?,288] = [batch_size, max_time, data]
        # shape of flatten(x) is (?, 288) for pong
        # shape of x is (1, ?, 288), for pong (after evaluation, "?" is 1 here)

        size = 256 # number of units in the LSTM cell, = output size
        if use_tf100_api:
            lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1] # sequence_length parameter, is [1] when the agent runs on the env, is the number of steps in the rollout otherwise
        self.step_size = step_size

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        # shape of lstm_outputs is (1, ?, 256), for pong

        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size]) # shape of x is (?, 256), for pong
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01)) # shape of self.logits is (?, 6), for pong
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1]) # reshape produces a 1-D vector here. shape of vf is (?,)
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]] # self.state_out is of type "list"
        self.sample = categorical_sample(self.logits, ac_space)[0, :] # shape of self.sample is (6,)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name) # self.var_list is of type "list"

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, r, a, l, c, h): # r, a, and l (length) ununsed
        sess = tf.get_default_session()          
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h}) # *(1)
        

    def value(self, ob, r, a, l, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

'''
(1) The brackets [] autour de "ob" sont necessaire car elles font passer ob (current state) de 3 a 4 dimensions, ce qui est le nombre de dimensions attendu par le placeholder de self.x dans LSTMPolicy
'''
