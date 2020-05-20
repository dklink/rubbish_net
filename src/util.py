import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.moveaxis(npimg, 0, 2))
    plt.show()


def mean_std(dataset):
    first_moment = torch.zeros(3)
    second_moment = torch.zeros(3)
    for i in range(len(dataset)):
        first_moment += dataset[i][0].sum((1, 2))
        second_moment += dataset[i][0].pow(2).sum((1, 2))
    first_moment /= len(dataset) * 64 ** 2
    second_moment /= len(dataset) * 64 ** 2
    return first_moment, torch.sqrt(second_moment - first_moment.pow(2))

def vertical_labels(labels):
    return torch.FloatTensor([labels])

def dataset_loaders(batch_size=128):
    train_mean, train_std = [moment.tolist() for moment in mean_std(dataset=torchvision.datasets.ImageFolder("../train_data", transforms.ToTensor()))]
    trans = [transforms.ToTensor(), transforms.Normalize(mean=train_mean, std=train_std)]
    train_dataset = torchvision.datasets.ImageFolder("../train_data", transforms.Compose(trans), transforms.Compose([transforms.Lambda(vertical_labels)]))
    val_dataset = torchvision.datasets.ImageFolder("../val_data", transforms.Compose(trans), transforms.Compose([transforms.Lambda(vertical_labels)]))
    test_dataset = torchvision.datasets.ImageFolder("../test_data", transforms.Compose(trans), transforms.Compose([transforms.Lambda(vertical_labels)]))
    for ds in [train_dataset, val_dataset, test_dataset]:
        assert ds.class_to_idx == {'not_trash': 0, 'trash': 1}
    train_load = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_load = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_load, val_load, test_load
