import numpy as np
# get diiferent activations functions 
from scipy.special import expit, softmax

def relu(x):
    return np.where(x < 0, np.zeros(x.shape), x)

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, eta=0.05, n_iter=1, h_activation=expit, o_activation=expit):
        """Create neural network instance

        Args:
            inputnodes (int): input neurons amount
            hiddennodes (int): hidden neurons amount
            outputnodes (int): output neurons amount
            eta (float, optional): training speed. Defaults to 0.05.
            n_iter (int, optional): epochs number. Defaults to 1.
            h_activation: activation func on the hidden layer. Defaults to expit func
            o_activation: activation func on the output layer. Defaults to expit func
        """
        # input neurons amount
        self.inodes = inputnodes
        # hidden neurons amount
        self.hnodes = hiddennodes
        # output neurons amount
        self.onodes = outputnodes
        # weights between input and hidden layers
        self.wih = np.random.normal(0.0 , pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # weights between two hidden layers
        # self.whh = np.random.normal(0.0 , pow(self.hnodes, -0.5), (self.hnodes, self.hnodes))
        # weights between hidden and output layers
        self.who = np.random.normal(0.0 , pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # training speed
        self.eta = eta
        # activation functions 
        self.h_activation_function = h_activation
        self.o_activation_function = o_activation
        # epochs number
        self.n_iter = n_iter

    def train(self, X, y):
        """Training neural network method

        Args:
            X: features
            y: targets
        """

        # convert features and targets to 2-d arrays
        X = np.array(X, ndmin=2).T
        y = np.array(y, ndmin=2).T

        # # compute input signals of the 1st hidden layer
        # h2h_inputs = np.dot(self.wih, X)
        # # compute output signals of the 1st hidden layer
        # h2h_outputs = self.h_activation_function(h2h_inputs)

        # # compute input signals of the 2nd hidden layer
        # hidden_inputs = np.dot(self.whh, h2h_outputs)
        # # compute output signals of the 2nd hidden layer
        # hidden_outputs = self.h_activation_function(hidden_inputs)

        # compute input signals of the 2nd hidden layer
        hidden_inputs = np.dot(self.wih, X)
        # compute output signals of the 2nd hidden layer
        hidden_outputs = self.h_activation_function(hidden_inputs)

        # compute input signals of the output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # compute output signals of the network
        final_outputs = self.o_activation_function(final_inputs)

        # count output errors: target - output
        output_errors = y - final_outputs
        # # compute errors of the 1st hidden layer
        # h2h_errors = np.dot(self.who.T, output_errors)
        # # compute errors of the 2nd hidden layer
        # hidden_errors = np.dot(self.whh.T, h2h_errors)

        # compute errors of the 2nd hidden layer
        hidden_errors = np.dot(self.who.T, output_errors)

        # update weights between hidden and output layers
        self.who += self.eta * np.dot(output_errors * final_outputs * (1.0 - final_outputs), np.transpose(hidden_outputs))
        # # update weights between 1st hidden and 2nd hidden layers
        # self.whh += self.eta * np.dot(h2h_errors * h2h_outputs * (1.0 - h2h_outputs), np.transpose(h2h_outputs))
        # update weights between input and hidden layers
        self.wih += self.eta * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), np.transpose(X))

    def predict(self, X):
        """Prediction method. Return predicted target values according to train data

        Args:
            X: features
        """
        # convert inputs to 2-d arrays
        inputs = np.array(X, ndmin = 2).T

        # # compute input signals of the hidden layer
        # h2h_inputs = np.dot(self.wih, inputs)
        # # compute output signals of the hidden layer
        # h2h_outputs = self.h_activation_function(h2h_inputs)

        # # compute input signals of the hidden layer
        # hidden_inputs = np.dot(self.whh, h2h_outputs)
        # # compute output signals of the hidden layer
        # hidden_outputs = self.h_activation_function(hidden_inputs)

        # compute input signals of the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # compute output signals of the hidden layer
        hidden_outputs = self.h_activation_function(hidden_inputs)
    
        # compute output signals of the hidden layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # compute output signals of the network
        final_outputs = self.o_activation_function(final_inputs)

        return final_outputs
