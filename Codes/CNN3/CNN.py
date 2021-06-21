import keras
from keras.backend import squeeze
import keras.backend as K


class CNN(keras.Model):
    def __init__(self,
                 concentration,
                 max_pool_size=(10, 2),
                 max_stride_size=(10, 2),
                 noPrior_filters=128,
                 noPriorConv_size=5,
                 noPriorConv_stride=1,
                 noPriorMax_size=2,
                 noPriorMax_stride=2,
                 filters=(128, 256, 512, 1024),
                 fc_units=128):
        super(CNN, self).__init__()

        self.concentration = concentration

        self.Norm = keras.layers.BatchNormalization()
        self.Relu = keras.layers.ReLU()
        self.Maxp = keras.layers.MaxPool2D(pool_size=max_pool_size,
                                           strides=max_stride_size)
        self.Mul = keras.layers.multiply

        self.noPriorConv = keras.layers.convolutional.Conv1D(filters=noPrior_filters,
                                                             kernel_size=noPriorConv_size,
                                                             strides=noPriorConv_stride,
                                                             use_bias=True,
                                                             kernel_initializer='glorot_normal',
                                                             activation='relu')
        self.noPriorNorm = keras.layers.BatchNormalization()
        self.noPriorMax = keras.layers.MaxPool1D(pool_size=noPriorMax_size,
                                                 strides=noPriorMax_stride)

        self.CNN_layers = []
        for f in filters:
            self.CNN_layers.append(keras.layers.Conv1D(filters=f,
                                                       kernel_size=3,
                                                       strides=1,
                                                       use_bias=True,
                                                       kernel_initializer='glorot_normal'))
            self.CNN_layers.append(keras.layers.MaxPool1D(pool_size=2, strides=2))
            self.CNN_layers.append(keras.layers.BatchNormalization())
            self.CNN_layers.append(keras.layers.Activation(activation='relu'))

        self.Flatten = keras.layers.Flatten()
        self.FC_1 = keras.layers.Dense(units=fc_units,
                                       activation='relu')
        self.FC_2 = keras.layers.Dense(units=1)

    def call(self, inputs, **kwargs):
        seq = inputs

        # Calculate occupancy
        concentration = self.concentration
        outputs = self.Relu(self.Norm(seq))
        outputs = self.Maxp(outputs)
        outputs = squeeze(outputs, axis=2)
        concentration = K.tile(concentration, [K.shape(outputs)[1]])
        concentration = K.tile(concentration, [K.shape(outputs)[0]])
        concentration = K.reshape(concentration, (-1, K.shape(outputs)[1], K.shape(outputs)[-1]))
        outputs = self.Mul([outputs, concentration])

        # No prior scan to extract interaction information
        # outputs = K.expand_dims(outputs, -1)
        outputs = self.noPriorConv(outputs)
        outputs = self.noPriorNorm(outputs)
        outputs = self.noPriorMax(outputs)

        # CNN
        for layer in self.CNN_layers:
            outputs = layer(outputs)

        # Full connect layer
        outputs = self.Flatten(outputs)
        outputs = self.FC_1(outputs)
        outputs = self.FC_2(outputs)

        return outputs
