import os
import struct
import numpy as np


def load_training_data(path='data/'):
    train_img_path = os.path.join(path, 'train-images-idx3-ubyte')
    train_lbl_path = os.path.join(path, 'train-labels-idx1-ubyte')

    with open(train_lbl_path, 'rb') as f:
        _, _ = struct.unpack(">II", f.read(8))
        lbl = np.fromfile(f, dtype=np.int8)

    with open(train_img_path, 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl


def convert_to_xy(img, lbl):
    # Convert the images into a vector. Each column is one image.
    X = img.reshape(img.shape[0], -1).T
    # Convert labels to a vector, each row is a digit, each column a sample.
    Y = np.vstack(((lbl == x).astype(int) for x in xrange(10)))
    return X, Y


def create_train_set(X, Y):
    n = .75*X.shape[1]
    idx = np.random.permutation(X.shape[1])
    train_idx, test_idx = idx[:n], idx[n:]
    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]
    X_test = X[:, test_idx]
    Y_test = Y[:, test_idx]
    return X_train, Y_train, X_test, Y_test


def load_test_data(path='data/'):
    test_img_path = os.path.join(path, 't10k-images-idx3-ubyte')
    test_lbl_path = os.path.join(path, 't10k-labels-idx1-ubyte')

    with open(test_lbl_path, 'rb') as f:
        _, _ = struct.unpack(">II", f.read(8))
        lbl = np.fromfile(f, dtype=np.int8)

    with open(test_img_path, 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl


