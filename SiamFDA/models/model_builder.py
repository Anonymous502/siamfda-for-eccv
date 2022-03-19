# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from SiamFDA.core.config import cfg
from SiamFDA.models.loss import select_iou_loss, sigmoid_focal_loss
from SiamFDA.models.backbone import get_backbone
from SiamFDA.models.head import get_fdam_head
from SiamFDA.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build fdam head
        if cfg.FDAM.FDAM:
            self.head = get_fdam_head(cfg.FDAM.TYPE,
                                     **cfg.FDAM.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }


    def log_softmax(self, cls):
        if cfg.FDAM.FDAM:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.head(zf, xf)

        # get loss
        #use focal loss
        cls_loss = sigmoid_focal_loss(cls, label_cls)

        # loc loss with iou loss
        loc_loss = select_iou_loss(loc, label_loc, label_cls)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
