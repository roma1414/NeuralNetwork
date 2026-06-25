from Network import *
import pickle

NN = Network([784, 32, 32, 32, 32, 10])

#print(NN.weights[1][0, 0])

with open("Network_2.pkl", "wb") as f:
    pickle.dump(NN, f)
