from CNN import CNN
import numpy as np
import keras.backend as K
import os
from Scanner import Scanner

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load data
    x = np.load('./data/X.npy')
    y = np.load("./data/Y.npy")
    pwm = np.load(file="./data/PWM.npy")
    c = np.load(file="./data/conc.npy")

    # Define test data set
    x = x[1000:1100]
    y = y[1000:1100]

    scanner = Scanner(pwm, x)
    x = scanner.scan()

    model = CNN(concentration=K.constant(c))
    model(x)
    model.compile(run_eagerly=True)
    model.load_weights(filepath='model/my_model.h5')
    # model(test_feature)
    pred = np.squeeze(np.asarray(model.predict(x)))
    print(y)
    print(pred)
