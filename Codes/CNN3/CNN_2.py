import keras


class CNN(keras.Model):
    def __init__(self,
                 filters=(128, 256, 512, 1024),
                 fc_units=128):
        super(CNN, self).__init__()

        self.CNN_layers = []
        for f in filters:
            self.CNN_layers.append(keras.layers.Conv1D(filters=f,
                                                       kernel_size=7,
                                                       strides=1,
                                                       padding='same',
                                                       use_bias=True,
                                                       kernel_initializer='glorot_normal'))
            self.CNN_layers.append(keras.layers.MaxPool1D(pool_size=2, strides=2))
            self.CNN_layers.append(keras.layers.BatchNormalization())
            self.CNN_layers.append(keras.layers.Activation(activation='relu'))
            self.CNN_layers.append(keras.layers.Dropout(rate=0.2))

        self.Attention = keras.Sequential(
            [
                keras.layers.Dense(units=128, activation='tanh'),
                keras.layers.Dense(units=1, activation='tanh')
            ]
        )

        self.Flatten = keras.layers.Flatten()
        self.FC = keras.layers.Dense(units=1)

    def call(self, inputs, **kwargs):
        # No prior scan to extract interaction information
        # outputs = K.expand_dims(outputs, -1)

        # CNN
        outputs = inputs
        for layer in self.CNN_layers:
            outputs = layer(outputs)

        # Attention layers
        outputs = self.Attention(outputs)

        # Full connect layer
        outputs = self.Flatten(outputs)
        outputs = self.FC(outputs)

        return outputs
