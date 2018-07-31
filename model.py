import torch.nn as nn

class conv_relu_pool(nn.Module):
    def __init__(self, in_channel, out_channel, conv_kernel_size, conv_stride, conv_padding,
                 pool_kernel_size, pool_stride, pool_padding):
        super(conv_relu_pool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=conv_kernel_size, stride=conv_stride,
                              padding=conv_padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                 padding=pool_padding)

    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        pool = self.pool(relu)
        return pool

class conv_relu_conv_relu_pool(nn.Module):
    def __init__(self, in_channel1, out_channel1, kernel_size1, stride1, padding1,
                 in_channel2, out_channel2, kernel_size2, stride2, padding2,
                 pool_kernel_size, pool_stride, pool_padding):
        super(conv_relu_conv_relu_pool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel1, out_channels=out_channel1,
                              kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.relu = nn.ReLU()
        self.conv_relu_pool = conv_relu_pool(in_channel=in_channel2, out_channel=out_channel2,
                                             conv_kernel_size=kernel_size2, conv_stride=stride2,
                                             conv_padding=padding2, pool_kernel_size=pool_kernel_size,
                                             pool_stride=pool_stride, pool_padding=pool_padding)

    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        result = self.conv_relu_pool(relu)
        return result

class bn_relu_conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dropout=0):
        super(bn_relu_conv, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride, padding= padding)

    def forward(self, x):
        bn = self.bn(x)
        relu = self.relu(bn)
        dropout = self.dropout(relu)
        conv = self.conv(dropout)
        return conv

class affine_relu(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0):
        super(affine_relu, self).__init__()
        self.affine = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.relu(self.affine(x)))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
