import mnist_loader
from Network import *
import pickle

#trainingData, validationData, testData = mnist_loader.load_data_wrapper()

'''with open("Network_1.pkl", "rb") as f:
    NN = pickle.load(f)'''
NN = Network([784, 32, 32, 32, 32, 10])

'''print(NN.weights[1][0, 0])
NN.SGD(trainingData, 8, 8, 3.0, testData=testData)'''
print(NN.weights[1][0, 0])

with open("Network_1.pkl", "wb") as f:
    pickle.dump(NN, f)





'''
print(len(NN.weights))
before = [w.copy() for w in NN.weights]
print(NN.weights[0][1, 1])

NN.SGD(trainingData, 1, 10, 3.0)
print(NN.weights[0][1, 1])

after = NN.weights

for i, (b, a) in enumerate(zip(before, after)):
    print(i, np.max(np.abs(b - a)))'''