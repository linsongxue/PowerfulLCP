from model_unit import CBAMModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from models import Darknet
from torchsummaryX import summary
from utils.utils import wh_iou
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from aux_net import *
from utils.datasets import *
from utils.utils import *
from utils.parse_config import *
from torchvision import models
import time

hyp = {'diou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.00579,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}


# def anchors_transform(anchors):
#     anchors = anchors.flatten().astype(np.int)
#     anchors = [str(num) for num in anchors]
#     anchors = ','.join(anchors)
#     return anchors
#
#
# def write_cfg(cfg_file, module_defs):
#     with open(cfg_file, 'w') as f:
#         for module_def in module_defs:
#             f.write(f"[{module_def['type']}]\n")
#             for key, value in module_def.items():
#                 if (key != 'type') and (key != 'anchors'):
#                     f.write(f"{key}={value}\n")
#                 elif key == 'anchors':
#                     value = anchors_transform(value)
#                     f.write(f"{key}={value}\n")
#             f.write("\n")
#
#
# cfg = 'cfg/yolov3.cfg'
# mask_cfg_path = '/'.join(cfg.split('/')[:-1]) + '/mask' + cfg.split('/')[-1]
# model = Darknet(mask_cfg_path)
# print(model.state_dict()['module_list.0.MaskConv2d.weight'])
# def mask_converted(mask_cfg='cfg/maskyolov3.cfg',
#                    weight_path='converted.pt',
#                    target='maskconverted.pt'):
#     last_darknet = 75
#     mask_weight = OrderedDict()
#     origin_weight = torch.load(weight_path)['model']
#     for k, v in origin_weight.items():
#         key_list = k.split('.')
#         idx = key_list[1]
#         if int(idx) < last_darknet and key_list[2] == 'Conv2d':
#             key_list[2] = 'Mask' + key_list[2]
#         key = '.'.join(key_list)
#         mask_weight[key] = v
#
#     model = Darknet(mask_cfg)
#     model.load_state_dict(mask_weight, strict=False)
#     chkpt = {'epoch': -1,
#              'best_fitness': None,
#              'training_results': None,
#              'model': model.state_dict(),
#              'optimizer': None}
#     torch.save(chkpt, target)
#     return mask_weight


# mask_converted()
# # for k, v in a.items():
# #     print(k)
#
# model = Darknet('cfg/maskyolov3.cfg')
# model.eval()
# img = torch.randn(8, 3, 416, 416)
# out, _ = model(img)
# out = non_max_suppression(out, conf_thres=0, iou_thres=0)
# print(out[0].shape)
# weight = torch.load('maskconverted.pt')['model']
# # weight = mask_converted('converted.pt')
# model.load_state_dict(weight, strict=True)
# aux = AuxNetUtils(model, hyp)
# print(aux.conv_layer_dict)
# print(aux.down_sample_layer)
# weight = torch.Tensor(4, 2, 3, 3)
# a = torch.zeros(2)
# a[0] = 1
# a = a.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(weight)
# print(a)
# mloss = torch.randn(4)
# results = torch.randn(7)
# with open('test.txt', 'a') as f:
#     f.write(('\n' + '%10s' * 10 + '\n') %
#             ('Stage', 'Epoch', 'DIoU', 'obj', 'cls', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))
#     f.write(('%10s' * 2 + '%10.3g' * 8) % (
#                 'FiTune ' + '0', '%g/%g' % (0, 14), *mloss, *results[:4]) + '\n')
#     f.write(('\n' + '%10s' * 10 + '\n') %
#             ('Stage', 'Change', 'MSELoss', 'PrunedLoss', 'AuxLoss', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))
#     f.write(('%10s' * 2 + '%10.3g' * 8) %
#             ('Pruning ' + '0', str(32) + '->' + str(32 - 20), *mloss, *results[:4]) + '\n')
# import logging
#
# logging.basicConfig(filename='logging.txt', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger()
# logger.info(('%10s' * 10 ) %
#             ('Stage', 'Epoch', 'DIoU', 'obj', 'cls', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))
# logger.info(('%10s' * 2 + '%10.3g' * 8) %
#             ('FiTune ' + '0', '%g/%g' % (0, 14), *mloss, *results[:4]))
# a = 3
# print(a % 2 == 1)

# model = Darknet('cfg/ssd-res50.cfg')
# summary(model, torch.Tensor(1, 3, 300, 300))

# a = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
# b, c, d = list(zip(*a))
# print(b)
# print(c)
# print(d)

a = torch.ones(2, 10, 20, 20)
b = torch.zeros(2, 10, 20, 20)
MSE = nn.MSELoss(reduction='sum')
l = MSE(a, b)
print(l)