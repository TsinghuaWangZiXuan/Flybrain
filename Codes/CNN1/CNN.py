import keras
from keras.backend import squeeze
import keras.backend as K
from Scanner import Scanner
import numpy as np


class CNN(keras.Model):
    def __init__(self,
                 dropout_rate=0.1,
                 max_pool_size=(2, 2),
                 max_stride_size=(2, 1),
                 tf_num=278,
                 filters=(64, 64, 64, 128, 128, 256),
                 n_classes=19):
        super(CNN, self).__init__()
        self.tf_num = tf_num

        # Load whole pwm
        pwm = np.load(file="./data/PWM.npy")
        self.Scannner = Scanner(pwm)
        self.Scan = keras.layers.Conv2D(filters=2*tf_num,
                                        kernel_size=(4, 21),
                                        trainable=False,
                                        use_bias=False,
                                        kernel_initializer=self.Scannner)

        self.Norm = keras.layers.BatchNormalization()
        self.Relu = keras.layers.ReLU()
        self.Maxp = keras.layers.MaxPool2D(pool_size=max_pool_size,
                                           strides=max_stride_size)

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

        self.Dropout = keras.layers.Dropout(rate=dropout_rate)

        self.Flatten = keras.layers.Flatten()
        self.FC_1 = keras.layers.Dense(units=1024, activation='relu')
        self.FC_2 = keras.layers.Dense(units=n_classes,
                                       activation='softmax')

    def call(self, inputs, training=True):
        # Scan DNA
        inputs = self.Scan(inputs)
        inputs = K.exp(inputs)

        # Compress inputs using max pooling layer
        seq = inputs  # (batch_size, 1, 2980, 556)
        outputs = squeeze(seq, axis=1)  # (b, 2980, 556)
        outputs = K.reshape(outputs, [outputs.shape[0], outputs.shape[1], 2, self.tf_num])  # (b, 2980, 2, 278)
        outputs = self.Relu(self.Norm(outputs))
        outputs = self.Maxp(outputs)
        outputs = squeeze(outputs, axis=2)  # (b, 1490, 278)

        # CNN
        for layer in self.CNN_layers:
            outputs = layer(outputs)

        # Drop out to avoid overfitting
        outputs = self.Dropout(outputs)

        # Full connect layer
        outputs = self.Flatten(outputs)
        outputs = self.FC_1(outputs)
        outputs = self.FC_2(outputs)
        return outputs
