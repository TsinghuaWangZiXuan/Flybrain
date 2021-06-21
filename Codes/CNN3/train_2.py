import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam
from CNN_2 import CNN
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Prepare GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Load data
    x = np.load(file="./data/X.npy")[:10000]
    y = np.load(file="./data/Y.npy")[:10000]

    # Transform X
    x = np.swapaxes(x, axis1=1, axis2=2)

    # Transform Y
    scaler = StandardScaler()
    y = scaler.fit_transform(np.expand_dims(y, axis=-1))
    y = np.squeeze(y)

    # Construct model
    batch_size = 32
    epoch = 20
    validation_ratio = 0.2
    model = CNN()
    optimizer = Adam(lr=0.0001)
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
