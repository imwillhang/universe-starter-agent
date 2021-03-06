from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import CnnPolicy
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0):
	"""
given a rollout, compute its returns and the advantage
"""
	batch_si = np.asarray(rollout.states)
	batch_a = np.asarray(rollout.actions)
	rewards = np.asarray(rollout.rewards)
	vpred_t = np.asarray(rollout.values + [rollout.r])

	rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
	batch_r = discount(rewards_plus_v, gamma)[:-1]
	delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
	# this formula for the advantage comes "Generalized Advantage Estimation":
	# https://arxiv.org/abs/1506.02438
	batch_adv = discount(delta_t, gamma * lambda_)

	return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal)

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal"])

class PartialRollout(object):
	"""
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.values = []
		self.r = 0.0
		self.terminal = False

	def add(self, state, action, reward, value, terminal):
		self.states += [state]
		self.actions += [action]
		self.rewards += [reward]
		self.values += [value]
		self.terminal = terminal

	def extend(self, other):
		assert not self.terminal
		self.states.extend(other.states)
		self.actions.extend(other.actions)
		self.rewards.extend(other.rewards)
		self.values.extend(other.values)
		self.r = other.r
		self.terminal = other.terminal

class RunnerThread():
	"""
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""
	def __init__(self, env, policy, num_local_steps, visualise):
		threading.Thread.__init__(self)
		self.rollout = None
		self.num_local_steps = num_local_steps
		self.env = env
		self.policy = policy
		self.daemon = True
		self.sess = None
		self.summary_writer = None
		self.visualise = visualise

	def init_runner(self, sess, summary_writer):
		self.sess = sess
		self.summary_writer = summary_writer
		self.rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise)

	def run(self):
		#with self.sess.as_default():
		
		# the timeout variable exists because apparently, if one worker dies, the other workers
		# won't die with it, unless the timeout is set to some large number.  This is an empirical
		# observation.
		self.rollout = next(self.rollout_provider)


def env_runner(env, policy, num_local_steps, summary_writer, render):
	"""
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
	last_state = env.reset()
	length = 0
	rewards = 0

	while True:
		terminal_end = False
		rollout = PartialRollout()

		for _ in range(num_local_steps):
			fetched = policy.step(last_state)
			action, value_, _ = fetched[0], fetched[1], fetched[2:]
			# argmax to convert from one-hot
			state, reward, terminal, info = env.step([action.argmax()])
			if render:
				env.render()

			last_state = np.squeeze(last_state).tolist()
			action = np.squeeze(action).tolist()
			reward = np.squeeze(reward).tolist()
			value_ = np.squeeze(value_).tolist()
			terminal = terminal.tolist()[-1]
			info = info[-1]

			# collect the experience
			rollout.add(last_state, action, reward, value_, terminal)
			length += 1
			rewards += reward

			last_state = state

			if 'episode' in info:
				info = info['episode']
				summary = tf.Summary()
				for k, v in info.items():
					summary.value.add(tag=k, simple_value=float(v))
				summary_writer.add_summary(summary, policy.global_step.eval())
				summary_writer.flush()

			timestep_limit = 5000
			if terminal or length >= timestep_limit:
				terminal_end = True
				if length >= timestep_limit:
					last_state = env.reset()
				print("Episode finished. Sum of rewards: %d. Length: %d" % (rewards, length))
				length = 0
				rewards = 0
				break

		if not terminal_end:
			rollout.r = policy.value(last_state).tolist()[-1]

		# once we have enough experience, yield it, and have the ThreadRunner place it on a queue
		yield rollout

class A3C(object):
	def __init__(self, env, task, visualise):
		"""
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""

		self.env = env
		self.task = task
		ob_space = self.env.observation_space
		ac_space = self.env.action_space
		worker_device = "/job:worker/task:{}/cpu:0".format(task)
		with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
			with tf.variable_scope("global"):
				self.network = CnnPolicy(ob_space, ac_space)
				self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
												   trainable=False)

		with tf.device(worker_device):
			with tf.variable_scope("local"):
				self.local_network = pi = CnnPolicy(ob_space, ac_space)
				pi.global_step = self.global_step

			self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
			self.adv = tf.placeholder(tf.float32, [None], name="adv")
			self.r = tf.placeholder(tf.float32, [None], name="r")

			log_prob_tf = tf.nn.log_softmax(pi.logits)
			prob_tf = tf.nn.softmax(pi.logits)

			# the "policy gradients" loss:  its derivative is precisely the policy gradient
			# notice that self.ac is a placeholder that is provided externally.
			# adv will contain the advantages, as calculated in process_rollout
			pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

			# loss of value function
			vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
			entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

			bs = tf.to_float(tf.shape(pi.x)[0])
			self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

			# 20 represents the number of "local steps":  the number of timesteps
			# we run the policy before we update the parameters.
			# The larger local steps is, the lower is the variance in our policy gradients estimate
			# on the one hand;  but on the other hand, we get less frequent parameter updates, which
			# slows down learning.  In this code, we found that making local steps be much
			# smaller than 20 makes the algorithm more difficult to tune and to get to work.
			self.runner = RunnerThread(env, pi, 20, visualise)


			grads = tf.gradients(self.loss, pi.var_list)

			if use_tf12_api:
				tf.summary.scalar("model/policy_loss", pi_loss / bs)
				tf.summary.scalar("model/value_loss", vf_loss / bs)
				tf.summary.scalar("model/entropy", entropy / bs)
				tf.summary.image("model/state", pi.x)
				tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
				tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
				self.summary_op = tf.summary.merge_all()

			else:
				tf.scalar_summary("model/policy_loss", pi_loss / bs)
				tf.scalar_summary("model/value_loss", vf_loss / bs)
				tf.scalar_summary("model/entropy", entropy / bs)
				tf.image_summary("model/state", pi.x)
				tf.scalar_summary("model/grad_global_norm", tf.global_norm(grads))
				tf.scalar_summary("model/var_global_norm", tf.global_norm(pi.var_list))
				self.summary_op = tf.merge_all_summaries()

			grads, _ = tf.clip_by_global_norm(grads, 40.0)

			# copy weights from the parameter server to the local model
			self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

			grads_and_vars = list(zip(grads, self.network.var_list))
			inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

			# each worker has a different set of adam optimizer parameters
			opt = tf.train.AdamOptimizer(7e-4)
			self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
			self.summary_writer = None
			self.local_steps = 0

	def gather_episode(self, sess):
		self.runner.run()

	def setup(self, sess, summary_writer):
		self.runner.init_runner(sess, summary_writer)
		self.summary_writer = summary_writer

	def pull_batch_from_queue(self):
		"""
self explanatory:  take a rollout from the queue of the thread runner.
"""
		return self.runner.rollout

	def process(self, sess):
		"""
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

		rollout = self.pull_batch_from_queue()
		batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

		should_compute_summary = self.task == 0 and self.local_steps % 11 == 0

		if should_compute_summary:
			fetches = [self.summary_op, self.train_op, self.global_step]
		else:
			fetches = [self.train_op, self.global_step]

		feed_dict = {
			self.local_network.x: batch.si,
			self.ac: batch.a,
			self.adv: batch.adv,
			self.r: batch.r
		}

		fetched = sess.run(fetches, feed_dict=feed_dict)

		if should_compute_summary:
			self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
			self.summary_writer.flush()
		self.local_steps += 1