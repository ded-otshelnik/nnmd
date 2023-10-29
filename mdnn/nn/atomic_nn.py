import numpy as np

from collections.abc import Iterable

def expit(x):
    """Sigmoid function

    Args:
        x: array of values 
    """
    return 1 / (1 + np.exp(-x))

class AtomicNN(object):
    def __init__(self, eta, n_iter, hidden_nodes):
        # input neurons amount - g and dg
        self.inodes = 2
        # output neurons amount - energy
        self.onodes = 1
        # hidden layers configuration
        self.hnodes = hidden_nodes

        self.whh = None
        # if 2 or more hidden layers
        if isinstance(self.hnodes, list | np.ndarray | Iterable):
            self.wih = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.hnodes[0], self.inodes))
            self.whh = []
            for i in range(len(self.hnodes) - 1):
                # weights between hidden layers
                self.whh.append(np.random.normal(0.0, pow(self.hnodes[i + 1], -0.5), (self.hnodes[i + 1], self.hnodes[i])))
            # weights between last hidden and output layers
            self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes[len(self.hnodes) - 1]))
        # if only 1 hidden layer
        elif isinstance(self.hnodes, int) and self.hnodes > 0:
            self.wih = np.random.normal(0.0 , pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
            # weights between hidden and output layers
            self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        else:
            raise ValueError("Hidden layers configuration must be consided by positive integer value \
                              or an iterable object of positive integers")

        # training speed
        self.eta = eta
        # activation function
        self.activation_function = expit
        # epochs number
        self.n_iter = n_iter

    def fit(self, g_train, e_train):
        for _ in range(self.n_iter):
            hidden_inputs, hidden_outputs = [], []
            hidden_inputs.append(np.dot(self.wih, g_train.g))
            hidden_outputs.append(self.activation_function(hidden_inputs))
            
            for i in range(1, len(self.hnodes) - 1):
                hidden_inputs.append(np.dot(self.whh[i], hidden_outputs))
                hidden_outputs.append(self.activation_function(hidden_inputs))
            output_inputs = np.dot(self.who, hidden_outputs)
            outputs = self.activation_function(output_inputs)

            error = outputs - e_train
            hidden_errors = np.dot(self.who.T, error)
            for i in range(len(self.hnodes) - 2, -1, -1):
                self.whh[i] += np.dot(hidden_errors * hidden_outputs[i] * (1.0 - hidden_outputs[i]), hidden_inputs[i].T)
                hidden_errors = np.dot(self.whh[i].T, hidden_errors)
            self.wih += self.eta * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(g_train))

    def predict(self, point):
        hidden_inputs, hidden_outputs = [], []
        hidden_inputs.append(np.dot(self.wih, point.g))
        hidden_outputs.append(self.activation_function(hidden_inputs[-1]))
        
        for i in range(1, len(self.hnodes) - 1):
            hidden_inputs.append(np.dot(self.whh[i], hidden_outputs[-1]))
            hidden_outputs.append(self.activation_function(hidden_inputs[-1]))
        output_inputs = np.dot(self.who, hidden_outputs[-1])
        outputs = self.activation_function(output_inputs)
        return outputs