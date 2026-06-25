import random
import numpy as np

class Network:

    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Return the output of the network if 'a' is input.
    def FeedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = Sigmoid(np.dot(w, a) + b)
        
        return a
    
    ''' The trainingData is a list of tuples '(x, y)' representing the training inputs and the desired
        outputs. If testData is provided then the network will be evaluated against the test data after each
        epoch, and partial progress printed out.'''
    def SGD(self, trainingData, epochs, miniBatchSize, eta, testData = None):
        if testData:
            nTest = len(testData)

        n = len(trainingData)
        for j in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]
            for miniBatch in miniBatches:
                self.UpdateMiniBatch(miniBatch, eta)
            if testData:
                print(f"Epoch {j}: {self.Evaluate(testData)} / {nTest}")
            else:
                print(f"Epoch {j} complete.")

    """ Updates the network's weights and biases via gradient descent using backpropagation
        to a single mini batch. The minBatch is a list of tuples '(x, y)', and 'eta' is the learning rate."""
    def UpdateMiniBatch(self, miniBatch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            delta_nabla_b, delta_nabla_w = self.Backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    ''' Return a tuple representing the gradient for the cost function (nabla_b, nabla_w)
        Each is a layer-by-layer list similar to weights/biases'''
    def Backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.CostDerivative(activations[-1], y) * \
            SigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.numLayers):
            z = zs[-l]
            sp = SigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def Evaluate(self, testData):
        test_results = [(np.argmax(self.FeedForward(x)), y)
                        for (x, y) in testData]
        return sum(int(x == y) for (x, y) in test_results)

    def CostDerivative(self, outputActivations, y):
        return (outputActivations-y)

# Miscellaneous functions
def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def SigmoidPrime(z):
    return Sigmoid(z)*(1-Sigmoid(z))