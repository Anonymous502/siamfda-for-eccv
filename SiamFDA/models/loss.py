# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F

from SiamFDA.core.config import cfg
from SiamFDA.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.FDAM.FDAM:
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)




def get_cls_sigmoid(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return pred, label

def sigmoid_focal_loss(pred, label):
    gamma = 2
    alpha = 0.25
    pred = pred.permute(0, 2, 3, 1).contiguous()
    batch_size = pred.shape[0]
    pred = pred.view(-1)
    pred = torch.sigmoid(pred)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()

    pred_pos, label_pos = get_cls_sigmoid(pred, label, pos)
    pred_neg, label_neg = get_cls_sigmoid(pred, label, neg)

    pred_final = torch.cat((pred_pos, pred_neg), 0)
    label_final = torch.cat((label_pos, label_neg), 0)


    term1 = (1 - pred_final) ** gamma * torch.log(pred_final)
    term2 = pred_final ** gamma * torch.log(1 - pred_final)

    loss = sum(- label_final * term1 * alpha - (1 - label_final) * term2 * (1 - alpha))
    loss = loss / batch_size
    return loss
    
    

if __name__ == '__main__':
    test = torch.tensor([1, 0, 1, -1, 0])
    pred_test = torch.tensor(([0.9, 0.9, 0.2, 0.241, 0.4]))
    sigmoid_focal_loss(pred_test, test)
