from Layers.Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    #flatten input to act as fully connected layer
    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return np.ravel(input_tensor).reshape(self.shape[0],-1)

    #Return conv layer from flattened fully connected layer
    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
