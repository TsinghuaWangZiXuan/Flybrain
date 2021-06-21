import keras
from matplotlib.colors import rgb2hex

from Scanner import Scanner
import numpy as np
from keras.optimizers import Adam
from CNN import CNN
import keras.backend as K
import os
import matplotlib.pyplot as plt

# Prepare GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Load data
    feature = np.load(file="./data/X.npy")
    label = np.load(file="./data/Y.npy")

    # Load whole pwm
    PWM = np.load(file="./data/PWM.npy")

    # Load concentration
    concentration = np.load(file="./data/conc.npy")

    # Motif list
    motif_list = ['acj6', 'br', 'C15', 'CG12236', 'crc', 'exd',
                  'ftz-f1', 'lola', 'mirr', 'Oli', 'otp', 'scrt',
                  'SREBP', 'usp']

    # Define candidate transcription factors
    whole_tf_num = len(motif_list)
    tf_len = 21
    index_list = [[1, 8]]  # br, C15, exd, mirr, otp
    history = []

    # Define train data
    split_point = 54
    X = feature[:split_point]
    Y = label[:split_point]

    y = K.constant(Y)

    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(len(index_list))])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex

    for indx, index in enumerate(index_list):

        # Define current PWM
        curr_motif = [motif_list[i] for i in index]
        p = np.zeros([4, tf_len, 1, 2 * len(index)])
        c = np.zeros([len(index)])
        for i in range(len(index)):
            p[:, :, :, i] = PWM[:, :, :, index[i]]
            p[:, :, :, i + len(index)] = PWM[:, :, :, whole_tf_num + index[i]]
            c[i] = concentration[index[i]]

        # Transform inputs
        scanner = Scanner(p, X)
        x = scanner.scan()

        # Construct model
        batch_size = 8
        epoch = 300
        validation_ratio = 0.3
        learning_rate = 0.001
        model = CNN(concentration=K.constant(c), tf_num=len(index))
        optimizer = Adam(lr=learning_rate)
        loss_func = 'binary_crossentropy'
        reduce_lr = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       factor=0.2,
                                                       patience=5,
                                                       min_lr=0.001)]
        model.compile(loss=loss_func,
                      optimizer=optimizer,
                      metrics=['accuracy'],
                      run_eagerly=True)
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

        h = model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epoch,
            validation_split=validation_ratio,
            callbacks=reduce_lr,
            shuffle=True
        )

        motif = [motif_list[j] for j in index]
        motif = "+".join(motif) if len(motif) > 1 else motif[0]

        # Plot loss
        plt.plot(h.history['val_loss'],
                 label=motif,
                 color=colors[indx])

        history.append(h)

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.show()

    for i, h in enumerate(history):
        motif = [motif_list[j] for j in index_list[i]]
        motif = "+".join(motif) if len(motif) > 1 else motif[0]

        mean_accuracy = np.mean(h.history['accuracy'])
        mean_val_accuracy = np.mean(h.history['val_accuracy'])
        print("{}: mean accuracy = {}, mean validation accuracy = {}".format(motif,
                                                                             mean_accuracy,
                                                                             mean_val_accuracy))

    # Plot accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    model.save_weights("./model/my_model.h5")
