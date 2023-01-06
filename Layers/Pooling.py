from Layers.Base import BaseLayer

import numpy as np
import copy

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape) -> None:
        super().__init__()
        self.ss = stride_shape
        self.ps = pooling_shape

        if isinstance(self.ss, int):
            self.ss = (self.ss, self.ss)

    # returns pooled input tensor
    def forward(self, input_tensor):
        # Poolin valid for only 3d tensors
        if len(input_tensor.shape) == 3:
            return input_tensor
        
        self.it = input_tensor
        batches = input_tensor.shape[0]
        channels = input_tensor.shape[1]
        p_r = self.ps[0]
        p_c = self.ps[1]
        it_r = input_tensor.shape[2]
        it_c = input_tensor.shape[3]
        o_r = int(np.floor((it_r - p_r)/self.ss[0] + 1))
        o_c = int(np.floor((it_c - p_c)/self.ss[1] + 1))


        output_tensor = np.zeros((batches, channels, o_r, o_c))
        # list to store index of maximas
        self.max_locations = list()
        for b in range(batches):
            for c in range(channels):
                for oi, i in enumerate(range(0, it_r - p_r + 1, self.ss[0])):
                    for oj, j in enumerate(range(0, it_c - p_c + 1, self.ss[1])):
                        output_tensor[b, c, oi, oj] = np.max(input_tensor[b, c, i:i+p_r, j:j+p_c]) 
                        #compute relative index(w.r.t input_tensor[b, c, i:i+p_r, j:j+p_c]) where the maximum value occurs
                        max_index = np.argwhere(input_tensor[b, c, i:i+p_r, j:j+p_c] == output_tensor[b, c, oi, oj]) 
                        max_index = max_index[0]
                        # calculate index from relative position
                        max_index[0] += i
                        max_index[1] += j
                        self.max_locations.append(max_index)
        
        return output_tensor

    def backward(self, error_tensor):
        if len(error_tensor) == 3:
            return error_tensor
        prev_error_tensor = np.zeros(self.it.shape)
        batches = self.it.shape[0]
        channels = self.it.shape[1]
        rows = error_tensor.shape[2]
        cols = error_tensor.shape[3]
        # use max_locations to compute error tensor.
        for b in range(batches):
            for c in range(channels):
                count = 0
                for r in range(rows):
                    for j in range(cols):
                        value = error_tensor[b][c][r][j]
                        index = tuple(self.max_locations[count])
                        prev_error_tensor[b][c][index] += value
                        count += 1

        return prev_error_tensor