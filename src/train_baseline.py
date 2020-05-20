from baseline_model import CNN
from util import dataset_loaders
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

train_load, val_load, test_load = dataset_loaders()
model = CNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2

train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []


for epoch in range(num_epochs):
    correct = 0
    for i, (inputs, labels) in enumerate(train_load):
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1).type(torch.float)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    
        train_loss.append(loss)
        predicted = torch.round(outputs)
        correct += (predicted.data == labels.data).sum().item()
    train_accuracy.append(correct / len(train_load.dataset))

    correct = 0
    for i, (inputs, labels) in enumerate(val_load):
        outputs = model(inputs)
        predicted = torch.round(outputs)
        labels = labels.unsqueeze(1).type(torch.float)

        correct += (predicted.data == labels.data).sum().item()
    val_accuracy.append(correct / len(val_load.dataset))

    print(train_accuracy[-1])
    print(val_accuracy[-1])
plt.plot(np.arange(len(train_loss)), train_loss)
plt.show()

