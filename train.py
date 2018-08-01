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
    T.ToPILImage(),
    T.Pad(padding=(2,2,2,2)),
    T.RandomCrop(size=32),
    T.RandomHorizontalFlip(),
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
dropout = 0
channel1 = 32
channel2 = 32
channel3 = 64 
channel4 = 64
hidden1 = 128
num_classes = 10

model = nn.Sequential(
    # model.conv_relu_conv_relu_pool(in_channel=3, mid_channel=channel1, kernel_size1=3, stride1=1,
    #                                padding1=1, out_channel=channel2, kernel_size2=3, stride2=1,
    #                                padding2=1, pool_kernel_size=2, pool_stride=2, pool_padding=0,
    #                                bn=True, dropout=dropout),
    # model.conv_relu_conv_relu_pool(in_channel=channel2, mid_channel=channel3, kernel_size1=3, stride1=1,
    #                                padding1=1, out_channel=channel4, kernel_size2=3, stride2=1,
    #                                padding2=1, pool_kernel_size=2, pool_stride=2, pool_padding=0,
    #                                bn=True, dropout=dropout),
    model.conv_relu_pool(in_channel=3, out_channel=20, conv_kernel_size=5,
                         conv_stride=1, conv_padding=2, pool_kernel_size=2,
                         pool_stride=2, pool_padding=0, bn=False, dropout=dropout),
    model.conv_relu_pool(in_channel=20, out_channel=50, conv_kernel_size=3,
                         conv_stride=1, conv_padding=1, pool_kernel_size=2,
                         pool_stride=2, pool_padding=0, bn=False, dropout=dropout),
    model.Flatten(),
    model.affine_relu(channel4*8*8, hidden1, bn=False, dropout=dropout),
    model.affine_relu(hidden1, num_classes, bn=False, dropout=dropout)
)
#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

train(model, optimizer, epochs=200)
