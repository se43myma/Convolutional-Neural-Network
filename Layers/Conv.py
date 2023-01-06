from Optimization.Optimizers import *
from Layers.Base import BaseLayer

import copy
import numpy as np
from scipy.signal import correlate, convolve

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels) -> None:
        super().__init__()
        self.trainable = True
        self.ss = stride_shape
        self.cs = convolution_shape
        self.nk = num_kernels
        
        self.weights = np.random.uniform(0, 1, (self.nk, *self.cs))
        self.bias = np.random.uniform(0, 1, (self.nk))
        if isinstance(self.ss, int):
            self.ss = (self.ss, self.ss)

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None

    @property 
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property 
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.deepcopy(value)
        self._optimizer2 = copy.deepcopy(value)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.cs)
        fan_out = self.nk * np.prod(self.cs[1:])

        self.weights = weights_initializer.initialize((self.nk, *self.cs), fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape,fan_in ,fan_in)

    # Returns input tensor for next layer
    def forward(self, input_tensor):
        self.it = input_tensor
        it_batches = input_tensor.shape[0]
        it_channels = input_tensor.shape[1]

        ot_batches = it_batches
        ot_channels = self.nk
        ot_size = self.it.shape[2:]
        self.ot = np.zeros((ot_batches, ot_channels, *ot_size))

        #calculate "same" padding. check for even convolution shape.
        if (self.cs[1] - 1)%2 == 0:
            pad_y_1 = pad_y_2 = int(np.abs((self.cs[1] - 1)/2))
        else:
            pad_y_1 = int(np.ceil(np.abs((self.cs[1] - 1)/2)))
            pad_y_2 = int(np.abs(self.cs[1] - 1 - pad_y_1))

        if len(self.cs) == 3:
            if (self.cs[2] - 1)%2 == 0:
                pad_x_1 = pad_x_2 = int(np.abs((self.cs[2] - 1)/2))
            else:
                pad_x_1 = int(np.ceil(np.abs((self.cs[2] - 1)/2)))
                pad_x_2 = int(np.abs(self.cs[2] - 1 - pad_x_1))

            input_tensor = np.pad(input_tensor, pad_width=((0,0),(0,0),(pad_y_1,pad_y_2),(pad_x_1,pad_x_2)), constant_values=0)
        else:
            input_tensor = np.pad(input_tensor, pad_width=((0,0),(0,0),(pad_y_1,pad_y_2)), constant_values=0)
        
        # store input tensor to use in backward()
        self.pit = input_tensor.copy()

        # Cross-Correlation
        for b in range(ot_batches):
            for c in range(ot_channels):
                self.ot[b][c] = correlate(input_tensor[b], self.weights[c], mode='valid') + self.bias[c]

        # compensate for stride
        if len(self.cs) == 3:
            self.ot = self.ot[:, :, ::self.ss[0], ::self.ss[1]]
        else:
            self.ot = self.ot[:, :, ::self.ss[0]]
        
        return self.ot
    
    # Returns error tensor of previous layer
    def backward(self, error_tensor):
        et_batches = error_tensor.shape[0]
        et_channels = error_tensor.shape[1]

        # Initialise the shapes of gradients. They have shame shape from which they are differentiated.
        self.gradient_input = np.zeros(self.it.shape)
        self.gradient_weights = np.zeros(self.weights.shape)
        self.gradient_bias = np.zeros(self.bias.shape)

        gi_batches = self.gradient_input.shape[0]
        gi_channels = self.gradient_input.shape[1]

        if (self.cs[1] - 1)%2 == 0:
            pad_y_1 = pad_y_2 = int(np.abs((self.cs[1] - 1)/2))
        else:
            pad_y_1 = int(np.ceil(np.abs((self.cs[1] - 1)/2)))
            pad_y_2 = int(np.abs(self.cs[1] - 1 - pad_y_1))

        #dilated tensor(det) and padded plus dilated tensor(pdet) to account for stride 
        det_m = self.ss[0] * error_tensor.shape[2] - self.ss[0] + 1
        # det_m = self.ss[0] * (error_tensor.shape[2] - 1) + error_tensor.shape[2]

        if len(error_tensor.shape) == 4:
            det_n = self.ss[1] * error_tensor.shape[3] - self.ss[1] + 1
            # det_n = self.ss[1] * (error_tensor.shape[3] - 1) + error_tensor.shape[3]
            det = np.zeros((et_batches,et_channels, det_m, det_n))
            det[:,:,::self.ss[0],::self.ss[1]] = error_tensor
            if (self.cs[2] - 1)%2 == 0:
                pad_x_1 = pad_x_2 = int(np.abs((self.cs[2] - 1)/2))
            else:
                pad_x_1 = int(np.ceil(np.abs((self.cs[2] - 1)/2)))
                pad_x_2 = int(np.abs(self.cs[2] - 1 - pad_x_1))
            
            pdet = np.pad(det, pad_width=((0,0),(0,0),(pad_y_1,pad_y_2),(pad_x_1,pad_x_2)), constant_values=0)
        else:
            det = np.zeros((et_batches,et_channels, det_m))
            det[:,:,::self.ss[0]] = error_tensor
            pdet = np.pad(det, pad_width=((0,0),(0,0),(pad_y_1,pad_y_2)), constant_values=0)

        # rearrange weights to match dimensions while convolving
        rearranged_weights = np.swapaxes(self.weights, 0, 1)
        rearranged_weights = np.flip(rearranged_weights, axis=1)        

        # convolve to compute gradient of input
        for b in range(gi_batches):
            for c in range(gi_channels):
                self.gradient_input[b][c] = convolve(pdet[b], rearranged_weights[c], mode='valid')

        # correlate to compute gradient of weights
        for b in range(et_batches):
            for c in range(et_channels):
                self.gradient_weights[c] = correlate(self.pit[b], det[b][c].reshape(1,*det.shape[2:]), mode='valid')
                self.gradient_bias[c] = np.sum(det[b][c])
        
        # calculate updated weights based on optimizers for weights and bias
        if self.optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer2.calculate_update(self.bias, self.gradient_bias)

        return self.gradient_input








