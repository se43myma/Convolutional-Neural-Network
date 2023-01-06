import numpy as np

#fan_in: input dimensions of the weights (/ input size)
#fan_out: output dimensions of weights (/ output size)

class Constant:
    def __init__(self, value) -> None:
        self.value = value
        
    def initialize(self, weight_shape, fan_in, fan_out):
        self.weights = np.full(weight_shape, self.value)
        return self.weights

class UniformRandom:
    def initialize(self, weight_shape, fan_in, fan_out):
        self.weights = np.random.uniform(0, 1, weight_shape)
        return self.weights

class Xavier:
    def initialize(self, weight_shape, fan_in, fan_out):
        sd = np.sqrt(2 / (fan_out + fan_in))
        self.weights = np.random.normal(0, sd, weight_shape)
        return self.weights

class He:            
    def initialize(self, weight_shape, fan_in, fan_out):
        self.weights = Xavier().initialize(weight_shape, fan_in, 0)
        return self.weights
