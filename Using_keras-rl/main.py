import sys
sys.path.append('keras-rl')

import gym
import numpy as np 

from model import NoisyDQN
from processor import AtariProcessor

from keras.optimizers import Adam

from rl.memory import PrioritizedMemory
from rl.policy import GreedyQPolicy
from rl.agents.dqn import DQNAgent
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
from rl.layers import NoisyNetDense

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

env = gym.make('MsPacmanDeterministic-v4')
np.random.seed(231)
env.seed(231)
nb_actions = env.action_space.n
input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])

agent = NoisyDQN(input_shape, nb_actions)
model = agent.model
memory = PrioritizedMemory(limit=1000000, alpha=.6, start_beta=.4, end_beta=1., steps_annealed=30000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()
policy = GreedyQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=True, enable_dueling_network=True, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1., n_step=3, custom_model_objects={"NoisyNetDense":NoisyNetDense})

dqn.compile(Adam(lr=.00025/4), metrics=['mae'])

folder_path = 'models/NoisyNSteps/'

weights_filename = folder_path + 'final_noisynet_nstep_pdd_dqn_MsPacmanDeterministic-v4_weights.h5f'
checkpoint_weights_filename = folder_path + 'final_noisynet_nstep_dqn_MsPacmanDeterministic-v4_weights_{step}.h5f'
log_filename = folder_path + 'final_noisynet_nstep_dqn_MsPacmanDeterministic-v4_REWARD_DATA.txt'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000000)]
callbacks += [TrainEpisodeLogger(log_filename)]

dqn.fit(env, callbacks=callbacks, nb_steps=30000000, verbose=0, nb_max_episode_steps=20000)