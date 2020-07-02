"""Keras"""
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector

"""
Table of Contents #

1. Keras.layers
    Input() -- Is used to instantiate a Keras tensor.
    Dense() -- Just your regular densely-connected NN layer.
    Reshape() -- Reshapes a tensor to the specified shape.
    LSTM() -- Long Short-Term Memory layer - Hochreiter 1997.
    Lambda() -- Wraps arbitrary expressions as a Layer object.
    Dropout() -- Applies Dropout to the input.
    Activation() -- Applies an activation function to an output.
    

"""

"1 - Keras.layers"
# shape - A shape tuple (integers), not including the batch size. For instance, shape=(32,)
tf.keras.Input(
    shape=None, batch_size=None, name=None
)
# Dense implements the operation: output = activation(dot(input, kernel) + bias)
# where activation is the element-wise activation function passed as the activation argument
# kernel is a weights matrix created by the layer, and bias is a bias vector created by the laye
# Units: Positive integer, dimensionality of the output space
tf.keras.layers.Dense(
    units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None
)
# Layer that reshapes inputs into the given shape.
model.add(tf.keras.layers.Reshape((-1, 2, 2)))
model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))
# units - Positive integer, dimensionality of the output space.
# return_state - Boolean. Whether to return the last state in addition to the output. Default: False.
tf.keras.layers.LSTM(
    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,return_state=False
)

LSTM_cell = LSTM(n_a, return_state = True, return_sequences=False)
# Dropout randomly sets input units to 0 with a frequency of rate at each step during training time.
# Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
tf.keras.layers.Dropout(
    rate, noise_shape=None, seed=None, **kwargs
)
# Activation
X_act = Activation('relu')(X)