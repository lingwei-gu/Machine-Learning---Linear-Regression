import torch
import pandas as pd

data = pd.read_csv('linear_data.csv', delimiter=',')
torch_data = torch.tensor(data.values)

class Model:
    def __init__(self, data):
        self.data = data
        self.length = self.data.size()[0]
        self.slope = 1
        self.yint = 0

    def error_approximation(self, x, y):
        return self.slope * x + self.yint - y

    def cost_function(self, is_yint):
        error = 0
        if is_yint:
            for i in range(self.length):
                error += self.error_approximation(self.data[i][0].item(), self.data[i][1].item())
        else:
            for i in range(self.length):
                error += self.error_approximation(self.data[i][0].item(),
                                                  self.data[i][1].item()) * self.data[i][0].item()
        return error


model = Model(torch_data)

while True:
    error_slope = model.cost_function(False)
    error_yint = model.cost_function(True)
    if (error_slope + error_yint) < 0.001:
        break
    model.slope -= 0.001 * (1 / model.length) * error_slope
    model.yint -= 0.001 * (1 / model.length) * error_yint

print(model.slope, model.yint)





