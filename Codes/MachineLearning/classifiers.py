import pickle
from sklearn import svm, metrics
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import DBSCAN, OPTICS, Birch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from Scanner import Scanner
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load data
    y = np.load(file="./y_birch_20.npy")
    x = np.load(file="./x.npy")

    # Load positional weight matrix
    # pwm = np.load(file="./data/PWM.npy")

    # Reduce the dimension of y
    # clustering = Birch(n_clusters=20).fit(y)
    # y = clustering.labels_
    # np.save(arr=y, file='y_birch_20.npy')

    # Transform x
    # scanner = Scanner(pwm, x)
    # x = scanner.multiscan()
    # np.save(arr=x, file='x.npy')

    # Upsampling data for balancing
    # smo = SMOTE(random_state=42)
    # x, y = smo.fit_sample(x, y)

    # Undersampling data for balancing
    rus = RandomUnderSampler(random_state=42)
    x, y = rus.fit_resample(x, y)

    # Shuffle dataset
    # state = np.random.get_state()
    # np.random.shuffle(x)
    # np.random.set_state(state)
    # np.random.shuffle(y)

    # Define training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=0)

    # LR
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(x_train, y_train)
    f = open("model/lr_down.pkl", 'wb')
    pickle.dump(lr_model, f)

    # Evaluate LR model
    y_predict = lr_model.predict(x_test)
    print(y_predict)
    print(confusion_matrix(y_test, y_predict))
    disp = plot_confusion_matrix(lr_model, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('LR Confusion Matrix')
    plt.show()

    # SVM
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(x_train, y_train)
    f = open("model/svm_down.pkl", 'wb')
    pickle.dump(svm_model, f)

    # Evaluate SVM model
    y_predict = svm_model.predict(x_test)
    print(y_predict)
    print(confusion_matrix(y_test, y_predict))
    disp = plot_confusion_matrix(svm_model, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('SVM Confusion Matrix')
    plt.show()

    # MLP
    mlp_model = MLPClassifier()
    mlp_model.fit(x_train, y_train)
    f = open("model/mlp_down.pkl", 'wb')
    pickle.dump(mlp_model, f)

    # Evaluate MLP model
    y_predict = mlp_model.predict(x_test)
    print(y_predict)
    print(confusion_matrix(y_test, y_predict))
    disp = plot_confusion_matrix(mlp_model, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('MLP Confusion Matrix')
    plt.show()

    # RF
    rf_model = RandomForestClassifier(n_estimators=200)
    rf_model.fit(x_train, y_train)
    f = open("model/rf_down.pkl", 'wb')
    pickle.dump(rf_model, f)

    # Evaluate RF model
    y_predict = rf_model.predict(x_test)
    print(y_predict)
    print(confusion_matrix(y_test, y_predict))
    disp = plot_confusion_matrix(rf_model, x_test, y_test, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('RF Confusion Matrix')
    plt.show()
