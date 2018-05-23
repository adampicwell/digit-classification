from network import Network, predict_from_output
from helpers import load_training_data, convert_to_xy
import numpy as np

img, lbl = load_training_data()
X, Y = convert_to_xy(img, lbl)
X_train, Y_train, X_test, Y_test = create_train_set(X, Y)

learning_rate = 0.003
layer_dims = [X.shape[0], 30, 20, Y.shape[0]]
net = Network(layer_dims, learning_rate)


costs = net.train(X, Y, 500)

print costs
