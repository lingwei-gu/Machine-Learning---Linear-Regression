import torch
import pandas as pd
import linear_training

# read csv file by pandas
data = pd.read_csv('linear_data.csv', delimiter=',')

# convert the data to tensor data structure
torch_data = torch.tensor(data.values)

model = linear_training.Model(torch_data)
model.training()

# read csv file by pandas
train = pd.read_csv('linear_test.csv', delimiter=',')

# convert the data to tensor data structure
torch_train = torch.tensor(train.values)


# to calculate the predicted y values
def calc(x):
    return model.slope * x + model.yint


# calculate the total error
error = 0
for i in range(300):
    error += (calc(torch_train[i][0].item()) - torch_train[i][1].item())

print(error / 300)
