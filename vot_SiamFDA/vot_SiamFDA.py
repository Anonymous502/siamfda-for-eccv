import sys
import cv2
import torch
import numpy as np

from os.path import join

from SiamFDA.core.config import cfg
from SiamFDA.models.model_builder import ModelBuilder
from SiamFDA.tracker.tracker_builder import build_tracker
from SiamFDA.utils.bbox import get_axis_aligned_bbox
from SiamFDA.utils.model_load import load_pretrain


from . import vot
from .vot import Rectangle, Polygon, Point






import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


cfg_root = "Path/to/SiamFDA/experiments/siamfda_r50_l234"
model_file = join(cfg_root, 'snapshot/checkpoint_e20.pth')
cfg_file = join(cfg_root, 'config.yaml')

def warmup(model):
    for i in range(10):
        model.template(torch.FloatTensor(1,3,127,127).cuda())

def setup_tracker():
    cfg.merge_from_file(cfg_file)

    model = ModelBuilder()
    model = load_pretrain(model, model_file).cuda().eval()

    tracker = build_tracker(model)
    warmup(model)
    return tracker


tracker = setup_tracker()

handle = vot.VOT("polygon")
region = handle.region()
try:
    region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                       region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
except:
    region = np.array(region)

cx, cy, w, h = get_axis_aligned_bbox(region)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

im = cv2.imread(image_file)  # HxWxC
# init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
tracker.init(im, gt_bbox_)

while True:
    img_file = handle.frame()
    if not img_file:
        break
    im = cv2.imread(img_file)
    outputs = tracker.track(im)
    pred_bbox = outputs['bbox']
    result = Rectangle(*pred_bbox)
    score = outputs['best_score']

    handle.report(result, score)
