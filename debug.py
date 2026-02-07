import numpy as np
import torch

# data = np.load("output/samples/stock_ground_truth_24_train.npy")


data = torch.load("dataset/mujoco0.0/original_data.pt", map_location=torch.device('cpu'))
print(data.shape)


# data = torch.load("dataset/mujoco0.0/train_a.pt", map_location=torch.device('cpu'))
# print(data.shape)
#
#
# data = torch.load("dataset/mujoco0.0/train_a.pt", map_location=torch.device('cpu'))
# print(data.shape)
#
#
# data = torch.load("dataset/mujoco0.0/train_b.pt", map_location=torch.device('cpu'))
# print(data.shape)
#
#
# data = torch.load("dataset/mujoco0.0/train_c.pt", map_location=torch.device('cpu'))
# print(data.shape)
#
# data = torch.load("dataset/mujoco0.0/train_d.pt", map_location=torch.device('cpu'))
# print(data.shape)
