import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, channels, layers, same_shape=True):
        super(ResNet, self).__init__()
        self.resnet = nn.ModuleList([])
        self.resnet.append(Residual(channels, same_shape=same_shape))
        for it in range(layers-1):
            self.resnet.append(Residual(channels, same_shape=True))
    def forward(self, x):
        out = x
        for resnet in self.resnet:
            out = resnet(out)
        return out

class Residual(nn.Module):
    def __init__(self, channels, same_shape=True, weight_init=True):
        super(Residual, self).__init__()
        stride = 1
        in_channels = channels
        self.same_shape = same_shape
        if not same_shape:
            stride = 2
            in_channels = channels // 2
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=channels,
                                   kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, stride=1, padding=1)
        if weight_init:
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            if not same_shape:
                nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.same_shape:
            return out + x
        return out + self.conv3(x)

class conv_relu_pool(nn.Module):
    def __init__(self, in_channel, out_channel, conv_kernel_size, conv_stride, conv_padding,
                 pool_kernel_size, pool_stride, pool_padding, bn=True, dropout=0):
        super(conv_relu_pool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=conv_kernel_size, stride=conv_stride,
                              padding=conv_padding)
        self.BN = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride,
                                 padding=pool_padding)

    def forward(self, x):
        conv = self.conv(x)
        if self.BN:
            conv = self.bn(conv)
        relu = self.relu(conv)
        relu = self.dropout(relu)
        pool = self.pool(relu)
        return pool

class conv_relu_conv_relu_pool(nn.Module):
    def __init__(self, in_channel, mid_channel, kernel_size1, stride1, padding1,
                 out_channel, kernel_size2, stride2, padding2,
                 pool_kernel_size, pool_stride, pool_padding, bn=True, dropout=0):
        super(conv_relu_conv_relu_pool, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel,
                              kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.BN = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv_relu_pool = conv_relu_pool(in_channel=mid_channel, out_channel=out_channel,
                                             conv_kernel_size=kernel_size2, conv_stride=stride2,
                                             conv_padding=padding2, pool_kernel_size=pool_kernel_size,
                                             pool_stride=pool_stride, pool_padding=pool_padding,
                                             bn=bn, dropout=dropout)

    def forward(self, x):
        conv = self.conv(x)
        if self.BN:
            conv = self.bn(conv)
        relu = self.relu(conv)
        relu = self.dropout(relu)
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
    def __init__(self, in_dim, out_dim, bn=True, dropout=0):
        super(affine_relu, self).__init__()
        self.affine = nn.Linear(in_dim, out_dim)
        self.BN = bn
        if bn == True:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        affine = self.affine(x)
        if self.BN:
            affine = self.bn(affine)
        return self.dropout(self.relu(affine))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
