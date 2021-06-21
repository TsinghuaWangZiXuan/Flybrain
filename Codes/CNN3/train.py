import keras
import tensorflow as tf
from Scanner import Scanner
import numpy as np
from keras.optimizers import Adam
from CNN import CNN
import keras.backend as K
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Prepare GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Load data
    X = np.load(file="./data/X.npy")[:1000]
    Y = np.load(file="./data/Y.npy")[:1000]

    # Load motif information
    pwm = np.load(file="./data/PWM.npy")

    # Load concentration information
    concentration = np.load(file="./data/conc.npy")

    # Transform Y
    scaler = StandardScaler()
    y = scaler.fit_transform(np.expand_dims(Y, axis=-1))

    # Define scanner
    scanner = Scanner(pwm, X)
    x = scanner.scan()

    # Construct model
    batch_size = 32
    epoch = 100
    validation_ratio = 0.2
    model = CNN(concentration=K.constant(concentration))
    optimizer = Adam(lr=0.01)
    loss_func = 'mse'
    reduce_lr = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.2,
                                                   patience=5,
                                                   min_lr=0.001),
                 keras.callbacks.EarlyStopping(min_delta=0.0001,
                                               patience=1000)]

    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=tf.keras.metrics.MeanSquaredError(),
                  run_eagerly=True)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    # Train model
    history = model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epoch,
        validation_split=validation_ratio,
        callbacks=reduce_lr,
        shuffle=True
    )

    # Plot metrics
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Save model
    model.save_weights("./model/my_model.h5")
