import numpy as np


class Network(object):

    def __init__(self, layer_dims, alpha):
        self.layer_dims = layer_dims
        self.n_layers = len(self.layer_dims)
        self.W = {}
        self.b = {}
        self.initialize_params()
        self.A = {}  # cache for backprop
        self.Z = {}  # cache for backprop
        self.alpha = alpha

    def initialize_params(self):
        self.W = {i: np.random.randn(self.layer_dims[i],
                                     self.layer_dims[i-1])*0.01
                  for i in range(1, self.n_layers)}
        self.b = {i: np.zeros((self.layer_dims[i], 1))
                  for i in range(1, self.n_layers)}

    def forward_prop(self, X):
        self.A[0] = X
        for layer in xrange(1, self.n_layers-1):
            self.Z[layer] = np.dot(self.W[layer], self.A[layer - 1]) + self.b[layer]
            self.A[layer] = relu(self.Z[layer])
        L = self.n_layers - 1
        self.Z[L] = np.dot(self.W[L], self.A[L-1]) + self.b[L]
        self.A[L] = sigmoid(self.Z[L])
        return self.A[L]

    def backward_prop(self, Y):
        m = Y.shape[1]
        L = self.n_layers -1
        dA = -(np.divide(Y, self.A[L]) - np.divide(1 - Y, 1 - self.A[L]))
        dW = {}
        db = {}
        L = self.n_layers - 1
        dZ = sigmoid_back(dA, self.Z[L])
        dW[L] = np.dot(dZ, self.A[L].T)/m
        db[L] = np.sum(dZ, axis=1, keepdims=True)/m
        dA = np.dot(self.W[L].T, dZ)
        for layer in reversed(range(1,self.n_layers-1)):
            dZ = relu_back(dA, self.Z[layer])
            A_prev = self.A[layer + 1]
            m = A_prev.shape[1]
            dW[layer] = np.dot(dZ, self.A[layer].T)/m
            db[layer] = np.sum(dZ, axis=1, keepdims=True)/m
            dA = np.dot(self.W[layer].T, dZ)
        return dW, db

    def update_parameters(self, dW, db):
        for i in range(self.n_layers):
            self.W[i] -= self.alpha*dW[i]
            self.b[i] -= self.alpha*db[i]


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(Z, 0)


def sigmoid_back(dA, Z):
    ds = sigmoid(Z)*(1-sigmoid(Z))
    return dA*ds


def relu_back(dA, Z):
    ds = Z > 0
    return dA*ds


def cost(AL, Y):
    m = Y.shape[1]
    cost = -1./m*np.sum((Y*np.log(AL)) + (1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


def predict_from_output(yhat):
    return np.argmax(yhat, axis=0)
