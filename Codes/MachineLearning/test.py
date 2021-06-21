import pickle

from matplotlib.colors import rgb2hex
from sklearn import svm, metrics
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    plot_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Scanner import Scanner
import seaborn

if __name__ == '__main__':
    # Load data
    y = np.load(file="./y_birch.npy")
    x = np.load(file="./x.npy")

    smo = SMOTE(random_state=42)
    x, y = smo.fit_sample(x, y)

    # Define training set and testing set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5,
                                                        random_state=0)
    models = []
    # Load models
    f = open("model/lr_down.pkl", 'rb')
    models.append(('lr_down', pickle.load(f)))

    f = open("model/lr_up.pkl", 'rb')
    models.append(('lr_up', pickle.load(f)))

    f = open("model/svm_up.pkl", 'rb')
    models.append(('svm_up', pickle.load(f)))

    f = open("model/svm_down.pkl", 'rb')
    models.append(('svm_down', pickle.load(f)))

    f = open("model/mlp_down.pkl", 'rb')
    models.append(('mlp_down', pickle.load(f)))

    f = open("model/mlp_up.pkl", 'rb')
    models.append(('mlp_up', pickle.load(f)))

    f = open("model/rf_down.pkl", 'rb')
    models.append(('rf_down', pickle.load(f)))

    f = open("model/rf_up.pkl", 'rb')
    models.append(('rf_up', pickle.load(f)))

    # Evaluate the model
    class_number = 20
    y_test = label_binarize(y_test, classes=range(class_number))
    plt.figure()
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(8)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex
    indx = 0
    for name, model in models:
        y_predict = model.predict_proba(x_test) if name[:3] == 'mlp' or name[:2] == 'rf' else model.decision_function(x_test)
        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_predict.ravel())
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr,
                 label='{0} micro-average ROC curve (area = {1:0.2f})'.format(name, roc_auc),
                 color=colors[indx])
        indx += 1

    # Plot ROC curve
    plt.plot([0, 1],
             [0, 1],
             color='navy',
             lw=2,
             linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
