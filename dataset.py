from torch.utils.data import Dataset
import pickle
import os.path as osp
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='latin1')
    return dict

class cifar10(Dataset):
    def __init__(self, root, transform=False):
        self.images = None
        self.labels = None
        self.root = root
        self.transform = transform

        self.filenames = []
        for i in range(1,6):
            self.filenames.append(osp.join(root, 'data_batch_'+str(i)))
        self.filenames.append(osp.join(root, 'test_batch'))

        self._preload()
        self.len = 60000

    def _preload(self):
        self.images = []
        self.labels = []
        for filename in self.filenames:
            dict = unpickle(filename)
            data = np.transpose(dict['data'].reshape(10000,3,32,32), (0,2,3,1))
            #print('data.shape = %s' % str(data.shape))
            self.images = self.images + list(data)
            self.labels = self.labels + dict['labels']
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        # print('image.shape = %s' % str(image.data.shape))
        if self.transform:
            image = self.transform(image)
        # print(image.data.shape)
        # print(image)
        # input()
        image = image.view(3,32,32)
        # image = augment(image, self.aug_train)
        # print(image.shape)
        # print(image)
        # input()
        return image, label

    def __len__(self):
        return self.len
