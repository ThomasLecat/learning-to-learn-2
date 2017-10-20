#!/usr/bin/env python
import cv2
import go_vncdriver
import tensorflow as tf
import argparse
import logging
import sys, signal
import time
import os
from a3c import A3C
from envs import create_env
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# (original comment) Disables write_meta_graph argument, which freezes entire process and is mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix="meta", write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step, latest_filename,
                                    meta_graph_suffix, False)

def run(args, server):
    env = create_env(args.env_id, client_id=str(args.task), remotes=args.remotes, num_trials=args.num_trials)

    num_global_steps = 10000000 
    num_test_steps = 1000000
    trainer = A3C(env, args.task, args.visualise, args.learning_rate, args.meta, args.remotes, args.num_trials, num_global_steps)

    # log, checkpoints et tensorboard

    # (Original Comment) Variable names that start with "local" are not saved in checkpoints.
    if use_tf12_api:
        variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_op = tf.variables_initializer(variables_to_save)
        init_all_op = tf.global_variables_initializer()
    else:
        variables_to_save = [v for v in tf.all_variables() if not v.name.startswith("local")]
        init_op = tf.initialize_variables(variables_to_save)
        init_all_op = tf.initialize_all_variables()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    logger.info('Trainable vars:')
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    if use_tf12_api:
        summary_writer = tf.summary.FileWriter(logdir + "_%d" % args.task)
    else:
        summary_writer = tf.train.SummaryWriter(logdir + "_%d" % args.task)

    logger.info("Events directory: %s_%s", logdir, args.task)
    # The tf.train.Supervisor provides a set of services that helps implement a robust training process. *(4)
    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=tf.report_uninitialized_variables(variables_to_save),
                             global_step=trainer.global_step,
                             save_model_secs=30,
                             save_summaries_secs=30)

    '''
    # beginning of the training
    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync) # copy weights from the parameter server to the local model
        trainer.start(sess, summary_writer) # lance l'execution de la methode "_run" du TheadRunner "trainer.runner" (object A3C du fichier A3C), qui genere des partial rollouts et les mets dans la queue
        global_step = sess.run(trainer.global_step) # will check in the tmp folder if there is some previously interrupted training to be continued, otherwise start from sratch and initialize the global_step counter at 0
        logger.info("Starting training at step=%d", global_step)
        while not sv.should_stop() and (not num_global_steps or global_step < num_global_steps):
            trainer.process(sess) # (original comment) grabs a rollout in the queue and update the parameters of the server
            global_step = sess.run(trainer.global_step)

    # End of the training, asks for all the services to stop.
    sv.stop()
    logger.info('Training finished ; reached %s steps. worker stopped.', global_step)
    time.sleep(5)
    '''

    # Beginning of the test phase
    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        sess.run(trainer.sync)
        trainer.start(sess, summary_writer)
        initial_global_step = global_step = sess.run(trainer.global_step)
        logger.info("Starting tests at step=%d", global_step)
        while not sv.should_stop() and (not num_test_steps or (global_step - initial_global_step)  < num_test_steps):
            trainer.inc_global_step(sess)
            global_step = sess.run(trainer.global_step)
    logger.info('Tests finished ; reached %s steps. worker stopped.', global_step)
    sv.stop()

def cluster_spec(num_workers, num_ps): # "ps" = "parameters server" ; num_ps is always 1 when called by the main method
    """
More tensorflow setup for data parallelism
"""
    cluster = {} # empty dictionnary
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append('{}:{}'.format(host, port))
        port += 1
    cluster['ps'] = all_ps  # cluster = {'ps': ['127.0.0.1:12222']}

    all_workers = []
    for _ in range(num_workers):
        all_workers.append('{}:{}'.format(host, port))
        port += 1
    cluster['worker'] = all_workers

    return cluster
    # Exemple :  pour deux workers, on a "cluster = {'ps': ['127.0.0.1:12222'], 'worker': ['127.0.0.1:12223', '127.0.0.1:12224']}"

def main(_): # In Python shells, the underscore (_) means the result of the last evaluated expression in the shell, *(2)
    """
(Original Comment) Setting up Tensorflow for data parallel work
    """
# 1 - Parse the arguments sent by train.py
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/pong", help='Log directory path')
    parser.add_argument('--env-id', default="PongDeterministic-v3", help='Test Environment id')
    parser.add_argument('-lr', '--learning-rate', default = 1e-4, type=float, help='Learning rate for the Adam Optimizer') # (new)
    parser.add_argument('-m', '--meta', action='store_true', help='if present, the training is done on different environments to achieve meta-learning') # (new)
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                             'or the address of pre-existing VNC servers and '
                             'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')
    parser.add_argument('-n', '--num-trials', type=int, default=None, help='Number of trials per episode with bandit environments')


    # (Add visualisation argument)
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    args = parser.parse_args()

# 2 - Create the adresses for the servers
    spec = cluster_spec(args.num_workers, 1) # spec = {'ps': ['127.0.0.1:12222'], 'worker': ['127.0.0.1:12223', '127.0.0.1:12224']} (pour -w 2)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def() # Tells tensforflow about the adresses of the machines we want to run on, *(3)

# 3 - ???
    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

# 4 - Launch the jobs on each server
    if args.job_name == "worker":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                 config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)) # *(1)
        run(args, server) # (the run function is defined above)
    else:
        server = tf.train.Server(cluster, job_name="ps", task_index=args.task,
                                 config=tf.ConfigProto(device_filters=["/job:ps"]))
        while True:
            time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()

'''
*(1) A tf.train.Server object contains a set of local devices, a set of connections to other tasks in its tf.train.ClusterSpec, and a tf.Session that can use these to perform a distributed computation. 
    Each server is a member of a specific named job and has a task index within that job

*(2) La valeur de "_" change pour chaque job et pour chaque task a l'interieur des job. Par exemple, pour le worker w-1 :
    ['worker.py', '--log-dir', '/homes/tjl16/Software/universe-starter-agent/tmp/pong', '--env-id', 'PongDeterministic-v3', '--num-workers', '2', '--job-name', 'worker', '--task', '0', '--remotes', '1']

*(3) The "spec" is a dictionnary mapping the name of jobs to a list of one or more network adresses that cooresponds to tasks in these jobs

*(4) The tf.train.Supervisor provides a set of services that helps implement a robust training process, including :
    - Handles shutdowns and crashes cleanly.
    - Can be resumed after a shutdown or a crash.
    - Can be monitored through TensorBoard.


'''





