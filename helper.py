from enum import Enum
import torch

class Mode(Enum):
    train = 0
    val = 1
    test = 2

def _check_acc(model, loader, mode, device, dtype):
    if mode == Mode.train:
        print('checking accuracy in training set')
    elif mode == Mode.val:
        print('checking accuracy in validation set')
    else:
        print('checking accuracy in test set')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('accuracy %d / %d (%.2f/100)' % (num_correct, num_samples, acc*100))
