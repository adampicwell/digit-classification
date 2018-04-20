from network import Network, predict_from_output
from helpers import load_training_data, convert_to_xy
import numpy as np

img, lbl = load_training_data()
X, Y = convert_to_xy(img, lbl)
print "Y shape {}".format(Y.shape)

learning_rate = 0.01
layer_dims = [X.shape[0], 30, 20, Y.shape[0]]
net = Network(layer_dims, learning_rate)

X_example = X[:, 0:2]
yhat = net.forward_prop(X_example)
#print yhat
#print predict_from_output(yhat)

dW, db = net.backward_prop(Y[:, 0:2])
print dW
print db
