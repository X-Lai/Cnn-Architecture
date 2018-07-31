from helper import _check_acc, Mode
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from dataset import cifar10
import torchvision.transforms as T
import model


def check_accuracy(model):
    if model.training:
        _check_acc(model, loader_train_test, Mode.train, device=device, dtype=dtype)
        _check_acc(model, loader_val, Mode.val, device=device, dtype=dtype)
    else:
        _check_acc(model, loader_test, Mode.test, device=device, dtype=dtype)


def train(model, optimizer, epochs, print_every=1000):
    model = model.to(device=device)
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
                check_accuracy(model)
                print()

BATCH_SIZE = 64
NUM_TRAIN = 49000

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
dtype = torch.float32

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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

lr = 0.001
dropout = 0.5
channel0 = 32
channel1 = 32
channel2 = 32
channel3 = 16
channel4 = 16
hidden1 = 1024
hidden2 = 512
hidden3 = 256
num_classes = 10

model = nn.Sequential(
    nn.Conv2d(3, channel0, kernel_size=3, stride=1, padding=1),
    model.bn_relu_conv(channel0, channel1, dropout=dropout),
    model.bn_relu_conv(channel1, channel2, dropout=dropout),
    model.bn_relu_conv(channel2, channel3, dropout=dropout),
    model.bn_relu_conv(channel3, channel4, dropout=dropout),
    model.Flatten(),
    model.affine_relu(channel4*32*32, hidden1, dropout=dropout),
    model.affine_relu(hidden1, hidden2, dropout=dropout),
    model.affine_relu(hidden2, hidden3, dropout=dropout),
    model.affine_relu(hidden3, num_classes)
)
optimizer = optim.Adam(model.parameters(), lr=lr)

train(model, optimizer, epochs=50)
