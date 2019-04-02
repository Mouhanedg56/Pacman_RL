import tensorflow as tf 
import numpy as np
import os 
import time

# from DqnAgent import Qnetwork
from Noisydqn import QnetworkNoisy
from utils import make_env, updateTargetGraph, updateTarget, processState
from buffer import experience_buffer

# params
# Best models [-4, -7, -14, 11]
num_episodes = 5
load_model = True
models = os.listdir("Noisydqn\\RL_project\\dqnNoisy")[1:]
# path = "Noisydqn\\model-steps-941-reward-36.ckpt"
h_size = 512
path_ = None
for path in models:

	path  = ".".join(path.split(".")[:-1])
	if path == path_: continue

	env = make_env()
	n_actions = env.action_space.n

	tf.reset_default_graph()
	mainQN = QnetworkNoisy(h_size, n_actions)
	targetQN = QnetworkNoisy(h_size, n_actions)

	init= tf.global_variables_initializer()
	saver = tf.train.Saver()

	jList = []
	rList = []
	total_steps = 0

	with tf.Session() as sess:
		sess.run(init)
		if load_model:
			print('Loading Model....')
			# ckpt = tf.train.get_checkpoint_state(path)
			saver.restore(sess, "Noisydqn\\RL_project\\dqnNoisy\\" + path)
			
		for i in range(num_episodes):

			s = env.reset()
			s = processState(s)
			d = False
			rAll = 0
			j = 0

			while not d:

				j += 1
				a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]
				env.render(mode='human')
				time.sleep(0.025)

				s1, r, d, _ = env.step(a)
				s1 = processState(s1)
				total_steps += 1

				rAll += r
				s = s1


			print("Episode " + str(i) + ": episode steps: " + str(j) + ", episode reward: " + str(rAll))

			jList.append(j)
			rList.append(rAll)
		print(total_steps, np.mean(rList))
	path_ = path
		
	print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")