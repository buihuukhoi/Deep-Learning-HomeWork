import numpy as np


def Relu(input):
    output = np.clip(input, 0.0, None)
    return output


def Softmax(input):
    output = np.exp(input) / np.exp(input).sum(axis=1, keepdims=True)
    return output


dispatcher = {
    'Relu': Relu,
    'Softmax': Softmax,
}