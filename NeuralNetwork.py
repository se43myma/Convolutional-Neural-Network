import copy
from Layers.Initializers import *

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer) -> None:
        self.optimizer = optimizer      #Optimizer selected for the problem e.g Stochastic descent, Stochastic + momentum, etc.
        self.loss = list()              #List containing loss in each layer
        self.layers = list()            #Architecture of neural network. Contains modules from folder Layers stored in a list. eg: [FullyConnected, ReLU, FullyConnected, Softmax, etc.]
        self.data_layer = list()        #contains list of training data, i.e input data and the corresponding lable data, stored in a list.
        self.loss_layer = None          #function to compute cost of the output layer. e.g: L2-loss, Cross-entropy, etc.
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    #iterate along the network and returns loss, computed from last layer
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = self.input_tensor
        for layer in self.layers:
            self.output_tensor = layer.forward(input_tensor)
            input_tensor = self.output_tensor
        self.output_loss = self.loss_layer.forward(self.output_tensor, self.label_tensor)
        return self.output_loss

    #propogate backward and return error tensor of input layer
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor) 
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    #append a layer to network
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer) 
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    
    #Training the network for #iterations and append the loss after each iteration to the list loss
    def train(self, iterations):
        for _ in range(0,iterations):
            loss = self.forward()
            error_tensor = self.backward()
            self.loss.append(loss)
    
    #test the network for an input and return prediction
    def test(self, input_tensor):
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            input_tensor = output_tensor
        return  output_tensor

