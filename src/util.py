import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from baseline_model import CNN
import time
import matplotlib.pyplot as plt

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.moveaxis(npimg, 0, 2))
    plt.show()

def mean_std(dataset):
    first_moment = torch.zeros(3)
    second_moment = torch.zeros(3)
    for i in range(len(dataset)):
        first_moment += dataset[i][0].sum((1,2))
        second_moment += dataset[i][0].pow(2).sum((1,2))
    first_moment /= len(dataset) * 64 ** 2
    second_moment /= len(dataset) * 64 ** 2 
    return first_moment, torch.sqrt(second_moment - first_moment.pow(2))

def dataset_loaders(batch_size = 64, mean = [0.3433, 0.1921, 0.1046], std = [0.4053, 0.2412, 0.2080]):

    trans = [transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)]
    dataset = torchvision.datasets.ImageFolder("../labeled_data", transforms.Compose(trans))
    train_len = len(dataset)*3//5; val_len = len(dataset)//5; test_len = len(dataset) - train_len - val_len
    torch.manual_seed(231)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths = [train_len, val_len, test_len])
    train_load = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size)
    val_load = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size)
    test_load = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)
    return train_load, val_load, test_load
