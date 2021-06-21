from matplotlib import pyplot as plt
import pickle

with open('./model/4096/loss.pkl', 'rb') as file:
    losses = pickle.load(file)

    plt.figure()
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.title('Training Loss')
    plt.show()