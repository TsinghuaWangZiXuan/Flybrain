from sklearn.preprocessing import StandardScaler
from CNN_2 import CNN
import numpy as np
import os

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load data
    x = np.load('./data/X.npy')[:11000]
    y = np.load("./data/Y.npy")[:11000]

    # Transform Y
    scaler = StandardScaler()
    y = scaler.fit_transform(np.expand_dims(y, axis=-1))
    y = np.squeeze(y)

    # Define test data set
    x = x[:500]
    y = y[:500]

    # Transform X
    x = np.swapaxes(x, axis1=1, axis2=2)

    model = CNN()
    model(x)
    model.compile(run_eagerly=True)
    model.load_weights(filepath='model/my_model.h5')
    # model(test_feature)
    pred = np.squeeze(np.asarray(model.predict(x)))
    arr = np.column_stack((y, pred))
    print(arr)

    # Save output file
    with open("./results/prediction.txt", 'w') as out_file:
        for group in arr:
            out_file.write(str(group[0]))
            out_file.write(',')
            out_file.write(str(group[1]))
            out_file.write('\n')