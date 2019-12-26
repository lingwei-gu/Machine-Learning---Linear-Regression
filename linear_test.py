import torch
import pandas as pd

data = pd.read_csv('linear_data.csv', delimiter=',')
torch_data = torch.tensor(data.values)
train = pd.read_csv('linear_test.csv', delimiter=',')
torch_train = torch.tensor(train.values)

def calc(x):
    return 0.9965889892848672 * x - 8.124868668571435e-05

pred = 0
for i in range(300):
    pred += (calc(torch_train[i][0].item()) - torch_train[i][1].item())

print(pred/300)