import numpy as np
from Layers.Base import *

#Rectified Linear Unit. Activation Function
class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()

    #x if x>0 else 0
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = self.input_tensor * (self.input_tensor > 0)
        return self.output_tensor

    #Return derivative
    #error_previous = 0 if x<=0 or = error
    def backward(self, error_tensor):
        self.error_tensor = error_tensor * (self.output_tensor > 0)
        return self.error_tensor
