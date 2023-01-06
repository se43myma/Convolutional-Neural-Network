import numpy as np

#sochastic gradient descent 
class Sgd:
    def __init__(self, learning_rate) -> None:
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

#Sgd with momentum
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate) -> None:
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v

class Adam:
    def __init__(self, learning_rate, mu, rho) -> None:
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1
        self.g = gradient_tensor
        self.v = self.mu * self.v + (1 - self.mu) * self.g
        self.r = self.rho * self.r + (1 - self.rho) * np.multiply(self.g, self.g)
        v = self.v / (1 - self.mu**self.k)
        r = self.r / (1 - self.rho**self.k)
        return weight_tensor - self.learning_rate * v / (np.sqrt(r) + np.finfo(float).eps)

