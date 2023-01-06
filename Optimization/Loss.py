import numpy as np

class CrossEntropyLoss:
    def __init__(self) -> None:
        pass
    
    #compute loss for the prediction_tensor
    #prediction_tensor is the prediction for an input and label_tensor is the expected one-hot encoded vector for the given input
    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        desired_prediction = np.where(label_tensor == 1, self.prediction_tensor + np.finfo(float).eps, label_tensor)
        self.loss = np.sum(-np.log(desired_prediction[desired_prediction != 0.0]))
        return self.loss

    #derivative of cross entropy loss
    def backward(self, label_tensor):
        prediction_epsilon =  self.prediction_tensor + np.finfo(float).eps * np.ones_like(self.prediction_tensor)
        self.error_tensor = - np.divide(label_tensor, prediction_epsilon)
        return self.error_tensor