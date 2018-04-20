from network import Network
from helpers import load_training_data, convert_to_xy

img, lbl = load_training_data()
X, Y = convert_to_xy(img, lbl)

learning_rate = 0.01
layer_dims = [30, 20]
net = Network(layer_dims, learning_rate)

