from CNN import CNN
import numpy as np
import keras.backend as K
import os
from Scanner import Scanner
from sklearn import metrics
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # Load data
    label = np.load('./data/Y.npy')
    feature = np.load("./data/X.npy")

    # Load pwm
    pwm = np.load(file="./data/PWM.npy")
    conc = np.load(file="./data/conc.npy")

    index = [1, 8]
    p = np.zeros([4, 21, 1, 2 * len(index)])
    c = np.zeros([len(index)])
    for i in range(len(index)):
        p[:, :, :, i] = pwm[:, :, :, index[i]]
        p[:, :, :, i + len(index)] = pwm[:, :, :, 14 + index[i]]
        c[i] = conc[index[i]]

    # Define test data
    split_point = 54
    x = feature[split_point:]
    y = label[split_point:]

    scanner = Scanner(p, x)
    x = scanner.scan()

    # Load model
    model = CNN(concentration=K.constant(c), tf_num=len(index))
    model(x)
    model.compile(run_eagerly=True)
    model.load_weights(filepath='model/my_model.h5')

    # Prediction
    predict = model.predict(x)

    # Plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y, predict, pos_label=1)
    print(metrics.roc_auc_score(y, predict))
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr,
             tpr,
             color='darkorange',
             lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1],
             [0, 1],
             color='navy',
             lw=2,
             linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
