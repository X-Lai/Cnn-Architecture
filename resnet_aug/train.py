from helper import _check_acc, Mode, checkpoint_save, checkpoint_load
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from dataset import cifar10
import torchvision.transforms as T
import model as model
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensor_transforms import Pad, RandomFlip, RandomCrop
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def check_accuracy(model):
    if model.training:
        train_acc = _check_acc(model, loader_train_test, Mode.train, device=device, dtype=dtype)
        val_acc = _check_acc(model, loader_val, Mode.val, device=device, dtype=dtype)
        a.append(train_acc)
        b.append(val_acc)
        return val_acc
    else:
        _check_acc(model, loader_test, Mode.test, device=device, dtype=dtype)

def test(model):
    return _check_acc(model, loader_test, Mode.test, device=device, dtype=dtype)

def train(model, optimizer, epochs, print_every=1000, plot_every=20, save_every=20):
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    model = model.to(device=device)
    best_acc = 0.0
    best_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    for e in range(epochs):
        for t, (x,y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss))
                val_acc = check_accuracy(model)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                scheduler.step(val_acc)
                print()

        if e % plot_every == 0 and e > 0:
            plt.plot(np.arange(len(a)), a, 'b', np.arange(len(b)), b, 'y')
            plt.ylim(0,1)
            # plt.show()
            plt.savefig("./logs/Epoch %d.png" % e)

        if e % save_every == 0 and e > 0:
            checkpoint_save(model=None, optimizer=None, state=best_state)
            print("best accuracy = %f" % best_acc)

a = []
b = []
BATCH_SIZE = 128
NUM_TRAIN = 49000
TRAIN = True

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype = torch.float32

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    Pad(size=(3,36,36)),
    RandomCrop(size=(32,32)),
    RandomFlip(h=True),
])

dataset = cifar10('./cifar-10-batches-py', transform=transform)

loader_train = DataLoader(dataset, batch_size=BATCH_SIZE,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_train_test = DataLoader(dataset, batch_size=BATCH_SIZE,
                               sampler=sampler.SubsetRandomSampler(random.sample(range(NUM_TRAIN), 1000)))
loader_val = DataLoader(dataset, batch_size=BATCH_SIZE,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test = DataLoader(dataset, batch_size=BATCH_SIZE,
                         sampler=sampler.SubsetRandomSampler(range(50000, 60000)))

dropout = 0
hidden1 = 128
num_classes = 10

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    model.ResNet(channels=16, layers=9, same_shape=True),
    model.ResNet(channels=32, layers=9, same_shape=False),
    model.ResNet(channels=64, layers=9, same_shape=False),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=8),
    model.Flatten(),
    nn.Linear(64, 10)
)

nn.init.kaiming_normal_(model[0].weight, nonlinearity='relu')
nn.init.kaiming_normal_(model[8].weight, nonlinearity='relu')

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

if TRAIN:
    train(model, optimizer, epochs=300)
else:
    checkpoint_load(model, optimizer)
    test(model)
