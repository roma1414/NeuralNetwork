import mnist_loader
from Network import *
import pickle

trainingData, validationData, testData = mnist_loader.load_data_wrapper()

with open("Network_1.pkl", "rb") as f:
    NN = pickle.load(f)

#print(NN.weights[1][0, 0])
NN.SGD(trainingData, 32, 16, 3.0)
#print(NN.weights[1][0, 0])

with open("Network_1.pkl", "wb") as f:
    pickle.dump(NN, f)
