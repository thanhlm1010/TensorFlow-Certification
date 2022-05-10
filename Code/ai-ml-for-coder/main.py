import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss='mean_squared_error')


