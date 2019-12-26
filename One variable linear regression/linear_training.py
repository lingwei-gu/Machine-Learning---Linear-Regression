import torch
import pandas as pd


# Assume the linear model is y = slope * x + yint
class Model:
    def __init__(self, training_data):
        self.data = training_data
        self.length = self.data.size()[0]
        self.slope = 1
        self.yint = 0
        self.learning_rate = 0.001

    # calculate the error difference for each training set
    def error_approximation(self, x, y):
        return self.slope * x + self.yint - y

    # calculate the sum of the derivative of the square difference of error
    def cost_function(self, is_yint):
        error = 0

        # It calculates the derivative sum of the approximation to yint
        if is_yint:
            for i in range(self.length):
                error += self.error_approximation(self.data[i][0].item(), self.data[i][1].item())

        # It calculates the derivative sum of the approximation to slope
        else:
            for i in range(self.length):
                error += self.error_approximation(self.data[i][0].item(),
                                                  self.data[i][1].item()) * self.data[i][0].item()
        return error

    # training the model
    def training(self):
        while True:
            error_slope = self.cost_function(False)  # retrieve the error approximation for slope
            error_yint = self.cost_function(True)  # retrieve the error approximation for yint
            if (error_slope + error_yint) < 0.001:  # stop training if the cost function (error) reaches a local minimum
                break

            # update the slope and yint with a learning rate of 0.001
            self.slope -= self.learning_rate * (1 / self.length) * error_slope
            self.yint -= self.learning_rate * (1 / self.length) * error_yint










