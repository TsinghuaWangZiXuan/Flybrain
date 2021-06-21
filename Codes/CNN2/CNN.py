import keras
from keras.backend import squeeze
import keras.backend as K


class CNN(keras.Model):
    def __init__(self,
                 concentration,
                 dropout_rate=0.1,
                 max_pool_size=(2, 2),
                 max_stride_size=(2, 1),
                 noPrior_filters=64,
                 noPriorConv_size=10,
                 tf_num=None,
                 noPriorConv_stride=1,
                 noPriorMax_size=(2, 1),
                 noPriorMax_stride=(2, 2)):
        super(CNN, self).__init__()

        self.concentration = concentration

        self.Norm = keras.layers.BatchNormalization()
        self.Relu = keras.layers.ReLU()
        self.Maxp = keras.layers.MaxPool2D(pool_size=max_pool_size,
                                           strides=max_stride_size)

        self.noPriorConv = keras.layers.convolutional.Conv2D(filters=noPrior_filters,
                                                             kernel_size=(noPriorConv_size, tf_num),
                                                             strides=(noPriorConv_stride, tf_num),
                                                             use_bias=True,
                                                             kernel_initializer='glorot_normal',
                                                             kernel_regularizer=keras.regularizers.l2(0.01),
                                                             activation='relu')
        self.noPriorMax = keras.layers.MaxPool2D(pool_size=noPriorMax_size,
                                                 strides=noPriorMax_stride)

        self.Dropout = keras.layers.Dropout(rate=dropout_rate)

        self.Flatten = keras.layers.Flatten()

        self.FC = keras.layers.Dense(units=1,
                                     kernel_regularizer=keras.regularizers.l2(0.01),
                                     activation='sigmoid')

    def call(self, inputs, training=True):
        # Compress inputs using max pooling layer
        seq = inputs
        outputs = self.Relu(self.Norm(seq))
        outputs = self.Maxp(outputs)
        outputs = squeeze(outputs, axis=2)

        # Calculate occupancy
        concentration = self.concentration
        concentration = K.tile(concentration, [K.shape(outputs)[1]])
        concentration = K.tile(concentration, [K.shape(outputs)[0]])
        concentration = K.reshape(concentration, (-1, K.shape(outputs)[1], K.shape(outputs)[-1]))
        outputs = keras.layers.multiply([outputs, concentration])

        # No prior convolution
        outputs = K.expand_dims(outputs, -1)
        outputs = self.noPriorConv(outputs)
        outputs = self.noPriorMax(outputs)

        # Drop out to avoid overfitting
        outputs = self.Dropout(outputs)

        # Full connect layer
        outputs = self.Flatten(outputs)
        outputs = self.FC(outputs)
        return outputs
