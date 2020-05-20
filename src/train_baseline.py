from baseline_model import BaselineCNN
from util import dataset_loaders
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import time


def evaluate(model, validation_loader):
    model.eval()
    with torch.no_grad():
        acc = .0
        for i, data in enumerate(validation_loader):
            X = data[0]
            y = data[1]
            predicted = model(X)
            acc += (predicted.round() == y).sum()/float(predicted.shape[0])
    model.train()
    return (acc/(i+1)).detach().item()


print('Loading Datasets...')
train_load, val_load, test_load = dataset_loaders()
print('Done!')

print('Initializing training parameters...')
model = BaselineCNN()
criterion = nn.BCELoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(f'  Model: {model.__class__.__name__}')
print(f'  Criterion: {criterion.__class__.__name__}')
print(f'  Optimizer: {optimizer.__class__.__name__}, lr={lr}')

num_epochs = 3

train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []
epoch_times = []

print('\n----------Begin Training----------\n')
for epoch in range(num_epochs):
    tic = time.time()
    correct = 0
    for i, (inputs, labels) in enumerate(train_load):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data.item())
        predicted = torch.round(outputs)
        correct += (predicted.data == labels.data).sum().item()
    train_accuracy.append(correct / len(train_load.dataset))

    val_accuracy.append(evaluate(model, val_load))
    toc = time.time()
    epoch_times.append(toc-tic)
    print(f"Epoch {epoch+1} executed in {toc-tic: .1f} seconds")
    print(f"  Train accuracy: {train_accuracy[-1]: .1%}")
    print(f"  Val accuracy:   {val_accuracy[-1]: .1%}")

plt.figure()
plt.plot(np.arange(len(train_loss)), train_loss, color = "lightblue")
plt.plot(np.arange(len(train_loss)), gaussian_filter1d(np.array(train_loss), sigma =3), color = 'blue')
plt.xlabel('minibatch #')
plt.ylabel('loss')
plt.show()

plt.figure()
plt.plot(np.arange(num_epochs), train_accuracy, '.-', label='train')
plt.plot(np.arange(num_epochs), val_accuracy, '.-', label='val')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim([.5, 1])
plt.legend()
plt.show()
