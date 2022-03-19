# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch
from torch.nn import functional as F


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x

class AdjustWithGAM(nn.Module):
    def __init__(self, in_channels, out_channels):

        
        super(AdjustWithGAM, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
            self.gam = GAM(out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))
                self.add_module('gam'+str(i+2),
                                GAM(out_channels[i]))
                        
       

    def forward(self, features):
        if self.num == 1:
            return self.gam(self.downsample(features))
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                gam_layer = getattr(self, 'gam'+str(i+2))
                out.append(gam_layer(adj_layer(features[i])))

            concat_feature = torch.cat((out[0], out[1], out[2]), 1)
            return concat_feature

class GAM(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        super(GAM, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
                             
        self.g = conv_nd(in_channels=self.inter_channels, out_channels=1,
                         kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        batch_size = x.size(0)
        theta_x = self.theta(x)
        g_x = self.g(theta_x).view(batch_size, 1, -1)
        g_x = F.softmax(g_x, dim=-1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        
        
        f = torch.matmul(theta_x, g_x)
        f = torch.mul(theta_x, f)
        f = f.view(batch_size, self.inter_channels, *x.size()[2:])
        
        
        W_y = self.W(f)
        z = W_y + x
        
        return z

if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20, 20)

        net = AdjustAllLayer([3], [2])
        out = net(img)
        print(out.size())
