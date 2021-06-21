import keras
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras.optimizers import Adam
from CNN import CNN
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Prepare GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Load data
    x = np.load(file="./data/X.npy")
    y = np.load(file="./data/y_birch_20.npy")
    n_classes = 19

    # Remove class 5
    index = np.where(y == 5)[0]
    x = np.delete(x, index, axis=0)
    y = np.delete(y, index, axis=0)
    y[y == 19] = 5

    x = np.expand_dims(x, axis=-1)

    # Binarize label
    y = label_binarize(y, classes=range(n_classes))

    # Construct model
    batch_size = 32
    epoch = 20
    validation_ratio = 0.2
    learning_rate = 0.001
    model = CNN()

    optimizer = Adam(lr=learning_rate)
    loss_func = 'categorical_crossentropy'
    reduce_lr = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   factor=0.2,
                                                   patience=5,
                                                   min_lr=0.0001)]
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  run_eagerly=True)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    history = model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epoch,
        validation_split=validation_ratio,
        callbacks=reduce_lr,
        shuffle=True
    )

    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model.save_weights("./model/my_model.h5")
