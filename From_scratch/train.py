import tensorflow as tf 
import numpy as np
import os 

from DqnAgent import Qnetwork
from utils import make_env, updateTargetGraph, updateTarget, processState
from buffer import experience_buffer

# params
batch_size = 32
update_freq = 4
y = 0.99
startE = 1
endE = 0.1
annealing_steps = 10000
num_episodes = 10000
pre_train_steps = 10000
max_epLength = 1500
load_model = False
path = "dqn/"
h_size = 512
tau = 0.001

env = make_env()
n_actions = env.action_space.n

tf.reset_default_graph()
mainQN = Qnetwork(h_size, n_actions)
targetQN = Qnetwork(h_size, n_actions)

init= tf.global_variables_initializer()
saver = tf.train.Saver()
trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)
myBuffer = experience_buffer()

e = startE
stepDrop = (startE - endE) / annealing_steps

jList = []
rList = []
total_steps = 0

if not os.path.exists(path): os.makedirs(path)

with tf.Session() as sess:
	sess.run(init)
	if load_model:
		print('Loading Model....')
		ckpt = tf.train.get_checkpoint_state(path)
		saver.restore(sess, ckpt.model_checkpoint_path)

	for i in range(num_episodes):

		episodeBuffer = experience_buffer()

		s = env.reset()
		s = processState(s)
		d = False
		rAll = 0
		j = 0

		while j < max_epLength:

			j += 1
			if np.random.rand(1) < e or total_steps < pre_train_steps:
				a = np.random.randint(0, n_actions)
			else:
				a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]

			s1, r, d, _ = env.step(a)
			s1 = processState(s1)
			total_steps += 1
			episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1,5]))

			if total_steps > pre_train_steps:
				if e > endE:
					e -= stepDrop

				if total_steps % (update_freq) == 0:
					trainBatch = myBuffer.sample(batch_size)

					Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:,3])})
					Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:,3])})
					end_multiplier = -(trainBatch[:,4] - 1)
					doubleQ = Q2[range(batch_size), Q1]
					targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)

					_ = sess.run(mainQN.updateModel,\
					 feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),\
					  mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

					updateTarget(targetOps, sess)

			rAll += r
			s = s1

			if d == True: 
				break
		print("Episode " + str(i) + ": episode steps: " + str(j) + ", episode reward: " + str(rAll))

		myBuffer.add(episodeBuffer.buffer)
		jList.append(j)
		rList.append(rAll)
		if i % 50 == 0:
			saver.save(sess, path + '/model-' + str(i) + '.ckpt')
			print('Saved Model')
		if len(rList) % 10 == 0:
			print(total_steps, np.mean(rList[-10:]), e)
	saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")


rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)