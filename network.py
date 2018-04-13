import numpy as np

class Network(object):

    def init(self, n_layers, n_features, neurons_per_layer):
        self.n_layers = n_layers
        self.n_features = n_features
        self.neurons_per_layer = neurons_per_layer
        self.initialize_params()

    def initialize_params(self):
        self.W = [np.random.randn((self.n_features, self.neurons_per_layer[i]))
                  for i in xrange(self.n_layers)]
        self.b = [np.zeros((1, self.neurons_per_layer[i])) for i in
                  xrange(self.n_layers)]
        self.g = [_relu for i in xrange(self.n_layers - 1)]
        self.g.append(_sigmoid)

    def forward_prop(self, X):
        A[0] = X
        for layer in xrange(1, self.n_layers):
            Z = np.multiply(self.W[layer], A[layer -1]) + self.b[layer]
            A[layer] = g[layer](Z)
        return A[-1]

    def 
