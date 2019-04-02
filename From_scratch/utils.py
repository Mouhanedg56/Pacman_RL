import numpy as np 
import tensorflow as tf

import gym
from atari_util import PreprocessAtari

def processState(states):
	return np.reshape(states, [84 * 84 * 4])

def updateTargetGraph(tfVars, tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx, var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx + total_vars//2].assign(var.value()*tau + (1 - tau) * tfVars[idx + total_vars//2].value()))

	return op_holder

def updateTarget(op_holder, sess):
	for op in op_holder:
		sess.run(op)

def make_env():
	env = gym.make("MsPacmanDeterministic-v4")
	env = PreprocessAtari(env, height=84, width=84, dim_order='tensorflow', color=False, n_frames=4, reward_scale=0.01)
	return env