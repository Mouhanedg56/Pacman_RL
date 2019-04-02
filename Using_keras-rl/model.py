import sys
sys.path.append('keras-rl')

from keras.models import Model
from keras.layers import Flatten, Convolution2D, Input
from rl.layers import NoisyNetDense


class NoisyDQN():
	def __init__(self, input_shape, nb_actions):
		self.frame = Input(shape=(input_shape))
		self.cv1 = Convolution2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first')(self.frame)
		self.cv2 = Convolution2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first')(self.cv1)
		self.cv3 = Convolution2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first')(self.cv2)
		self.dense= Flatten()(self.cv3)
		self.dense = NoisyNetDense(512, activation='relu')(self.dense)
		self.buttons = NoisyNetDense(nb_actions, activation='linear')(self.dense)
		self.model = Model(inputs=self.frame,outputs=self.buttons)