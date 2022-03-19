from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


from SiamFDA.core.xcorr import xcorr_depthwise

class FDAM(nn.Module):
    def __init__(self):
        super(FDAM, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )
        

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class DepthwiseFDAM(FDAM):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=1, weighted=False):
        super(DepthwiseFDAM, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiFDAM(FDAM):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiFDAM, self).__init__()
        channel = 256 * 3
        reduction = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_template = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc_search = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        for i in range(len(in_channels)):
            self.add_module('box' + str(i + 2), DepthwiseFDAM(in_channels[i], in_channels[i], cls_out_channels))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))


    def forward(self, z_fs, x_fs):

        b, c, _, _ = z_fs.size()
        y = self.avg_pool(z_fs).view(b, c)
        y = self.fc_template(y).view(b, c, 1, 1)
        z_fs = z_fs + z_fs * y.expand_as(z_fs)
        
        b1, c1, _, _ = x_fs.size()
        y1 = self.avg_pool(x_fs).view(b1, c1)
        y1 = self.fc_search(y1).view(b1, c1, 1, 1)
        x_fs = x_fs + x_fs * y1.expand_as(x_fs)

        z_fs = torch.split(z_fs, 256, dim=1)
        x_fs = torch.split(x_fs, 256, dim=1)
         
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box'+str(idx))
            c, l = box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))

        def avg(lst):
            return sum(lst) / len(lst)

        return avg(cls), avg(loc)
        
        
if __name__ == '__main__':
    z_test = torch.rand([32, 256 * 3, 7, 7])
    x_test = torch.rand([32, 256 * 3, 15, 15])
    z_fs = torch.split(z_test, 256, dim=1)
    x_fs = torch.split(x_test, 256, dim=1)
    cls = []
    loc = []
    for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
        print('test')