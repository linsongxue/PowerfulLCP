import torch.nn as nn
import torch
from torchvision.ops import roi_align
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.distributed as dist

import traceback
import argparse
from tqdm import tqdm
import math
from collections import OrderedDict
import os.path as path
import os
import time
import logging
import numpy as np

import test
from models import Darknet
from model_unit import MaskConv2d
from email_error import send_error_report
from utils.utils import compute_loss, init_seeds, xywh2xyxy
from utils.parse_config import parse_data_cfg, parse_model_cfg
from utils.datasets import LoadImagesAndLabels
from utils.torch_utils import select_device

aux_weight = "../weights/DCP/aux.pt"  # weight for aux net
aux_trained = path.exists(aux_weight)
progress_chkpt = "../weights/DCP/pruning.pt"
progress_result = "./DCP/pruning_progress.txt"
last = "../weights/DCP/last.pt"
loss_analyse = "./DCP/loss_analyse.txt"

logging.basicConfig(filename=loss_analyse, filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()

hyp = {'diou': 3.5,  # giou loss gain
       'giou': 3.5,
       'cls': 37.5,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.001,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.9,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'joint_loss': 1.0,
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)


def anchors_transform(anchors):
    anchors = anchors.flatten().astype(np.int)
    anchors = [str(num) for num in anchors]
    anchors = ','.join(anchors)
    return anchors


def write_cfg(cfg_file, module_defs):
    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if (key != 'type') and (key != 'anchors'):
                    f.write(f"{key}={value}\n")
                elif key == 'anchors':
                    value = anchors_transform(value)
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file


def mask_cfg_and_converted(mask_replace_layer,
                           cfg='cfg/yolov3-voc.cfg',
                           weight_path='../weights/converted-voc.pt',
                           target='../weights/maskconverted-voc.pt'):
    mask_cfg = '/'.join(cfg.split('/')[:-1]) + '/mask' + cfg.split('/')[-1]
    origin_mdfs = parse_model_cfg(cfg)
    mask_mdfs = []
    mask_mdfs.append(origin_mdfs.pop(0))
    for i, mdf in enumerate(origin_mdfs):
        if str(i) in mask_replace_layer:
            mdf['type'] = 'maskconvolutional'
        mask_mdfs.append(mdf)
    write_cfg(mask_cfg, mask_mdfs)

    mask_weight = OrderedDict()
    origin_weight = torch.load(weight_path)['model']
    for k, v in origin_weight.items():
        key_list = k.split('.')
        idx = key_list[1]
        if idx in mask_replace_layer and key_list[2] == 'Conv2d':
            key_list[2] = 'Mask' + key_list[2]
            key = '.'.join(key_list)
            mask_weight[key] = v
            mask_weight[key.replace('weight', 'selected_channels_mask')] = torch.ones(v.size(1), dtype=torch.float32)
        else:
            key = '.'.join(key_list)
            mask_weight[key] = v

    model = Darknet(mask_cfg)
    model.load_state_dict(mask_weight, strict=True)
    if target is not None:
        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}
        torch.save(chkpt, target)

    return mask_cfg, model.state_dict()


class DCPAuxNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DCPAuxNet, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, input, targets):
        boxes = targets[:, [0, 2, 3, 4, 5]]
        _, _, h, w = input.shape
        boxes[:, [2, 4]] *= h
        boxes[:, [1, 3]] *= w
        o_h, o_w = torch.mean(boxes[:, 4]), torch.mean(boxes[:, 3])
        boxes[:, 1:] = xywh2xyxy(boxes[:, 1:])
        feat = roi_align(input, boxes, output_size=(o_h, o_w))
        out = self.bn(feat)
        out = self.act(out)
        out = self.pooling(out)
        out = self.linear(out.squeeze())

        return out


class AuxNetUtils(object):

    def __init__(self, model):
        self.model = model
        self.num_classes = int(self.model.module_defs[-1]['classes'])
        self.__down_sample = ['12', '37', '62', '75']
        self.aux_in_layer = self.__down_sample
        self.__fusion_layer = ['87', '99']
        self.layer_info = OrderedDict()
        self.conv_layer_dict = OrderedDict()
        self.prune_guide = OrderedDict()
        self.pruning_layer = []
        self.mask_replace_layer = []
        self.analyze()

    def child_analyse(self, name, child):
        """
        Analyse the module to get some useful params
        :param child:Module, the module of all net
        :return:Dictionary, {"in_channels": Int,
                             "out_channels": Int,
                             "kernel_size": Int,
                             "stride": Int,
                             "padding": Int}
        """
        try:
            module = child.Conv2d
            analyse_dict = {"in_channels": module.in_channels,
                            "out_channels": module.out_channels,
                            "kernel_size": module.kernel_size[0],
                            "stride": module.stride[0],
                            "padding": (module.kernel_size[0] - 1) // 2,
                            "bias": False if module.bias is None else True,
                            "shortcut": False}
        except AttributeError:
            try:
                analyse_dict = {"shortcut": True,
                                "from": self.model.module_defs[int(name)]['from']}
            except KeyError:
                analyse_dict = None

        return analyse_dict

    def analyze(self):
        """
        对模型进行分析，将所有关键信息提取，主要是完成了self.layer_info和self.down_sample_layer的填充
        两个都是列表，第一个内容是字典，第二个记录了下采样的层序号
        :return: None
        """
        # 梳理每一层的信息
        for name, child in self.model.module_list.named_children():
            if isinstance(child, nn.Sequential):  # DarkNet后面的第一层，因为选用的是输入特征图，所以要多用一层
                info = self.child_analyse(name, child)
                self.layer_info[name] = info
            else:
                self.layer_info[name] = None

        # 为每一个需要裁剪的层分配辅助网络的附着层
        last_darknet_layer = 75
        for name, value in self.layer_info.items():
            if int(name) <= last_darknet_layer and not value["shortcut"]:
                if int(name) <= int(self.__down_sample[0]):
                    self.conv_layer_dict[name] = self.__down_sample[0]
                elif int(name) <= int(self.__down_sample[1]):
                    self.conv_layer_dict[name] = self.__down_sample[1]
                elif int(name) <= int(self.__down_sample[2]):
                    self.conv_layer_dict[name] = self.__down_sample[2]
                elif int(name) <= int(self.__down_sample[3]):
                    self.conv_layer_dict[name] = self.__down_sample[3]

        # 需要裁剪的层
        self.pruning_layer = list(self.conv_layer_dict.keys())[::-1]
        self.pruning_layer.pop(-1)
        # 需要将原有网络替换为MaskConv2d
        self.mask_replace_layer = list(self.conv_layer_dict.keys()) + self.__fusion_layer

        # 生成监督剪枝的顺序向量
        for layer in self.pruning_layer:
            guide_item = {'pruned': False,
                          'base_channels': self.layer_info[layer]['in_channels'],
                          'retain_channels': self.layer_info[layer]['in_channels']}
            if self.layer_info[str(int(layer) - 1)]['shortcut']:
                from_index = int(self.layer_info[str(int(layer) - 1)]['from'])
                guide_item['pre_link'] = str(from_index + int(layer))
            self.prune_guide[layer] = guide_item
        for k, v in self.prune_guide.items():
            if 'pre_link' in v.keys():
                self.prune_guide[v['pre_link']]['tail_link'] = k
                v.pop('pre_link')

        self.prune_guide['37']['sync'] = '99'
        self.prune_guide['62']['sync'] = '87'

        self.pruning_layer = self.pruning_layer[::-1]

    def compute_retain_channels(self, layer, rate):
        if self.prune_guide[layer]['pruned']:
            raise Exception

        self.prune_guide[layer]['retain_channels'] = math.floor(
            self.prune_guide[layer]['base_channels'] * (1 - rate))
        self.prune_guide[layer]['pruned'] = True
        return self.prune_guide[layer]['retain_channels']

    def load_state(self, prune_guide):
        self.prune_guide = prune_guide

    def state(self):
        return self.prune_guide

    def next_prune_layer(self, layer):
        idx = self.pruning_layer.index(layer)
        try:
            return self.pruning_layer[idx + 1]
        except IndexError:
            return 'end'

    def creat_aux_model(self, layer_name):
        aux_model = DCPAuxNet(self.layer_info[layer_name]['out_channels'], self.num_classes)

        return aux_model


class HookUtils(object):

    def __init__(self):
        self.origin_features = {}
        self.prune_features = {}
        for i in range(torch.cuda.device_count()):
            self.origin_features['gpu{}'.format(i)] = []
            self.prune_features['gpu{}'.format(i)] = []

    def hook_origin_output(self, module, input, output):
        self.origin_features['gpu{}'.format(output.device.index)].append(output)

    def hook_prune_output(self, module, input, output):
        self.prune_features['gpu{}'.format(output.device.index)].append(output)

    def cat_to_gpu0(self):
        device_list = list(self.origin_features.keys())
        assert self.origin_features['gpu0'] or self.prune_features['gpu0'], "Did not get hook features!"
        for device in device_list[1:]:
            for i in range(len(self.origin_features['gpu0'])):
                self.origin_features['gpu0'][i] = torch.cat(
                    [self.origin_features['gpu0'][i], self.origin_features[device][i].cuda(0)], dim=0)
            for i in range(len(self.prune_features['gpu0'])):
                self.prune_features['gpu0'][i] = torch.cat(
                    [self.prune_features['gpu0'][i], self.prune_features[device][i].cuda(0)], dim=0)

    def clean_hook_out(self):
        for key in list(self.origin_features.keys()):
            self.origin_features[key] = []
        for key in list(self.prune_features.keys()):
            self.prune_features[key] = []


def loss_for_dcp_aux(out_put, targets):
    loss = torch.zeros(1, device=out_put.device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    ground_truth = targets[:, 1]
    loss += criterion(out_put, ground_truth.long())

    return loss


def train_for_aux(cfg, data_loader, weights, aux_weight, device, resume, epochs):
    init_seeds()
    batch_size = data_loader.batch_size
    accumulate = 64 // batch_size

    model = Darknet(cfg).to(device)
    model_chkpt = torch.load(weights, map_location=device)
    model.load_state_dict(model_chkpt['model'], strict=True)
    del model_chkpt
    aux_util = AuxNetUtils(model)
    hook_util = HookUtils()

    start_epoch = 0

    aux_model_list = []
    pg = []
    for layer in aux_util.aux_in_layer:
        aux_model = aux_util.creat_aux_model(layer)
        aux_model.to(device)
        for v in aux_model.parameters():
            pg += [v]
        aux_model_list.append(aux_model)

    optimizer = optim.SGD(pg, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    del pg

    if resume:
        chkpt = torch.load(aux_weight, map_location=device)

        for i, layer in enumerate(aux_util.aux_in_layer):
            aux_model_list[i].load_state_dict(chkpt['aux_in{}'.format(layer)], strict=True)

        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])

        start_epoch = chkpt['epoch'] + 1

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 3, 2 * epochs // 3], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    handles = []  # 结束训练后handle需要回收
    for name, child in model.module_list.named_children():
        if name in aux_util.aux_in_layer:
            handles.append(child.register_forward_hook(hook_util.hook_origin_output))

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers

    nb = len(data_loader)
    model.nc = 80
    model.hyp = hyp
    model.arc = 'default'
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):

        for aux_model in aux_model_list:
            aux_model.train()
        print(('\n' + '%10s' * 6) % ('Stage', 'Epoch', 'gpu_mem', 'AuxID', 'cls', 'targets'))

        # -----------------start batch-----------------
        pbar = tqdm(enumerate(data_loader), total=nb)
        for i, (imgs, targets, _, _) in pbar:

            if len(targets) == 0:
                continue

            ni = i + nb * epoch
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = model(imgs)
            hook_util.cat_to_gpu0()
            for aux_idx, aux_model in enumerate(aux_model_list):
                pred = aux_model(hook_util.origin_features['gpu0'][aux_idx], targets)
                loss = loss_for_dcp_aux(pred, targets)

                loss *= batch_size / 64

                loss.backward()

                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 3 + '%10.3g' * 3) % (
                    'Train Aux', '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, aux_idx, loss, len(targets))
                pbar.set_description(s)
            # 每个batch后要把hook_out内容清除
            hook_util.clean_hook_out()
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        # -----------------end batches-----------------

        scheduler.step()
        final_epoch = epoch + 1 == epochs
        chkpt = {'epoch': epoch,
                 'optimizer': None if final_epoch else optimizer.state_dict()}
        for i, layer in enumerate(aux_util.aux_in_layer):
            chkpt['aux_in{}'.format(layer)] = aux_model_list[i].state_dict()

        torch.save(chkpt, aux_weight)

        torch.save(chkpt, "../weights/DCP/aux-voc.pt")
        del chkpt

        with open("./DCP/aux_result.txt", 'a') as f:
            f.write(s + '\n')

    # 最后要把hook全部删除
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()


def fine_tune(prune_cfg, data, aux_util, device, train_loader, test_loader, epochs=10):
    with open(progress_result, 'a') as f:
        f.write(('\n' + '%10s' * 10 + '\n') %
                ('Stage', 'Epoch', 'DIoU', 'obj', 'cls', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))

    batch_size = train_loader.batch_size
    img_size = train_loader.dataset.img_size
    accumulate = 64 // batch_size
    hook_util = HookUtils()

    pruned_model = Darknet(prune_cfg, img_size=(img_size, img_size)).to(device)

    chkpt = torch.load(progress_chkpt, map_location=device)
    pruned_model.load_state_dict(chkpt['model'], strict=True)

    current_layer = chkpt['current_layer']
    aux_in_layer = aux_util.conv_layer_dict[current_layer]
    aux_model = aux_util.creat_aux_model(aux_in_layer)
    aux_model.to(device)

    aux_model.load_state_dict(chkpt['aux_in{}'.format(aux_in_layer)], strict=True)
    aux_loss_scalar = max(0.01, pow((int(aux_in_layer) + 1) / 75, 2))

    start_epoch = chkpt['epoch'] + 1

    aux_util.load_state(chkpt['prune_guide'])

    if start_epoch == epochs:
        return current_layer  # fine tune 完毕，返回需要修剪的层名

    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(pruned_model.named_parameters()).items():
        if 'MaskConv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    for v in aux_model.parameters():
        pg0 += [v]  # parameter group 0

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    if chkpt['optimizer'] is not None:
        optimizer.load_state_dict(chkpt['optimizer'])

    del chkpt

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 3, 2 * (epochs // 3)], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        pruned_model = nn.parallel.DistributedDataParallel(pruned_model, find_unused_parameters=True)
        pruned_model.yolo_layers = pruned_model.module.yolo_layers

    # -------------start train-------------
    nb = len(train_loader)
    pruned_model.nc = 80
    pruned_model.hyp = hyp
    pruned_model.arc = 'default'
    for epoch in range(start_epoch, epochs):

        # -------------register hook for model-------------
        for name, child in pruned_model.module.module_list.named_children():
            if name == aux_in_layer:
                handle = child.register_forward_hook(hook_util.hook_prune_output)

        # -------------register hook for model-------------

        pruned_model.train()
        aux_model.train()

        print(('\n' + '%10s' * 7) % ('Stage', 'Epoch', 'gpu_mem', 'DIoU', 'obj', 'cls', 'total'))

        # -------------start batch-------------
        mloss = torch.zeros(4).to(device)
        pbar = tqdm(enumerate(train_loader), total=nb)
        for i, (img, targets, _, _) in pbar:
            if len(targets) == 0:
                continue

            ni = nb * epoch + i
            img = img.to(device).float() / 255.0
            targets = targets.to(device)

            pruned_pred = pruned_model(img)
            pruned_loss, pruned_loss_items = compute_loss(pruned_pred, targets, pruned_model)
            pruned_loss *= batch_size / 64

            hook_util.cat_to_gpu0()

            aux_pred = aux_model(hook_util.prune_features['gpu0'][0], targets)

            aux_loss = loss_for_dcp_aux(aux_pred, targets)
            aux_loss *= aux_loss_scalar * batch_size / 64

            loss = pruned_loss + aux_loss
            loss.backward()

            hook_util.clean_hook_out()
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            pruned_loss_items[2] += aux_loss.item()
            mloss = (mloss * i + pruned_loss_items) / (i + 1)
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 3 + '%10.3g' * 4) % (
                'FiTune ' + current_layer, '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss)
            pbar.set_description(s)
        # -------------end batch-------------

        scheduler.step()
        handle.remove()

        results, _ = test.test(prune_cfg,
                               data,
                               batch_size=batch_size * 2,
                               img_size=416,
                               model=pruned_model,
                               conf_thres=0.1,
                               iou_thres=0.5,
                               save_json=False,
                               dataloader=test_loader)

        """
        chkpt = {'current_layer':
                 'epoch':
                 'model': 
                 'optimizer': 
                 'aux_in12': 
                 'aux_in37':
                 'aux_in62':
                 'aux_in75':
                 'prune_guide':}
        """
        chkpt = torch.load(progress_chkpt, map_location=device)
        chkpt['current_layer'] = current_layer
        chkpt['epoch'] = epoch
        chkpt['model'] = pruned_model.module.state_dict() if type(
            pruned_model) is nn.parallel.DistributedDataParallel else pruned_model.state_dict()
        chkpt['optimizer'] = None if epoch == epochs - 1 else optimizer.state_dict()
        chkpt['aux_in{}'.format(aux_in_layer)] = aux_model.state_dict()
        chkpt['prune_guide'] = aux_util.state()

        torch.save(chkpt, progress_chkpt)

        torch.save(chkpt, last)

        if epoch == epochs - 1:
            torch.save(chkpt, '../weights/backup{}.pt'.format(current_layer))

        del chkpt

        with open(progress_result, 'a') as f:
            f.write(('%10s' * 2 + '%10.3g' * 8) % (
                'FiTune ' + current_layer, '%g/%g' % (epoch, epochs - 1), *mloss, *results[:4]) + '\n')
    # -------------end train-------------
    torch.cuda.empty_cache()
    return current_layer


def channels_select(prune_cfg, data, origin_model, aux_util, device, data_loader, select_layer, pruned_rate):
    with open(progress_result, 'a') as f:
        f.write(('\n' + '%10s' * 10 + '\n') %
                ('Stage', 'Change', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))
    logger.info(('%10s' * 7) % ('Stage', 'Channels', 'Batch', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total'))

    batch_size = data_loader.batch_size
    img_size = data_loader.dataset.img_size
    accumulate = 64 // batch_size
    hook_util = HookUtils()
    handles = []
    n_iter = 15

    pruning_model = Darknet(prune_cfg, img_size=(img_size, img_size)).to(device)
    chkpt = torch.load(progress_chkpt, map_location=device)
    pruning_model.load_state_dict(chkpt['model'], strict=True)

    aux_in_layer = aux_util.conv_layer_dict[select_layer]
    aux_model = aux_util.creat_aux_model(aux_in_layer)
    aux_model.to(device)

    aux_model.load_state_dict(chkpt['aux_in{}'.format(aux_in_layer)], strict=True)
    aux_loss_scalar = max(0.01, pow((int(aux_in_layer) + 1) / 75, 2))

    aux_util.load_state(chkpt['prune_guide'])
    del chkpt

    solve_sub_problem_optimizer = optim.SGD(pruning_model.module_list[int(aux_in_layer)].MaskConv2d.parameters(),
                                            lr=hyp['lr0'],
                                            momentum=hyp['momentum'])

    for name, child in origin_model.module_list.named_children():
        if name == aux_in_layer:
            handles.append(child.register_forward_hook(hook_util.hook_origin_output))

    for name, child in pruning_model.module_list.named_children():
        if name == aux_in_layer:
            handles.append(child.register_forward_hook(hook_util.hook_prune_output))

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        origin_model = torch.nn.parallel.DistributedDataParallel(origin_model, find_unused_parameters=True)
        origin_model.yolo_layers = origin_model.module.yolo_layers
        pruning_model = torch.nn.parallel.DistributedDataParallel(pruning_model, find_unused_parameters=True)
        pruning_model.yolo_layers = pruning_model.module.yolo_layers

    retain_channels_num = aux_util.compute_retain_channels(select_layer, pruned_rate)
    pruning_model.nc = 80
    pruning_model.hyp = hyp
    pruning_model.arc = 'default'
    pruning_model.eval()
    aux_model.eval()
    MSE = nn.MSELoss(reduction='mean')
    mloss = torch.zeros(4).to(device)

    for i_k in range(retain_channels_num):

        data_iter = iter(data_loader)
        pbar = tqdm(range(n_iter), total=n_iter)
        print(('\n' + '%10s' * 7) % ('Stage', 'gpu_mem', 'channels', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total'))
        for i in pbar:

            imgs, targets, _, _ = data_iter.next()

            if len(targets) == 0:
                continue

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = origin_model(imgs)

            _, pruning_pred = pruning_model(imgs)
            pruning_loss, _ = compute_loss(pruning_pred, targets, pruning_model)

            hook_util.cat_to_gpu0()
            mse_loss = torch.zeros(1, device=device)

            aux_pred = aux_model(hook_util.prune_features['gpu0'][0], targets)
            aux_loss = loss_for_dcp_aux(aux_pred, targets)
            mse_loss += MSE(hook_util.prune_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss

            loss.backward()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 3 + '%10.3g' * 4) % (
                'Prune ' + select_layer, '%.3gG' % mem, '%g/%g' % (i_k, retain_channels_num), mse_loss, pruning_loss,
                aux_loss, loss)
            pbar.set_description(s)

            if i % 20 == 0:
                logger.info(('%10s' * 3 + '%10.3g' * 4) %
                            ('Prune' + select_layer, str(i_k), '%g/%g' % (i, n_iter), mse_loss, pruning_loss, aux_loss,
                             loss))

            hook_util.clean_hook_out()

        grad = pruning_model.module.module_list[int(select_layer)].MaskConv2d.weight.grad.detach() ** 2
        grad = grad.sum((2, 3)).sqrt().sum(0)

        if i_k == 0:
            pruning_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[:] = 1e-5
            if 'sync' in aux_util.prune_guide[select_layer].keys():
                sync_layer = aux_util.prune_guide[select_layer]['sync']
                pruning_model.module.module_list[int(sync_layer)].MaskConv2d.selected_channels_mask[
                (-1 * aux_util.prune_guide[select_layer]['base_channels']):] = 1e-5

        selected_channels_mask = pruning_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask
        _, indices = torch.topk(grad * (1 - selected_channels_mask), 1)

        pruning_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[indices] = 1
        if 'sync' in aux_util.prune_guide[select_layer].keys():
            pruning_model.module.module_list[int(sync_layer)].MaskConv2d.selected_channels_mask[
                -(aux_util.prune_guide[select_layer]['base_channels'] - indices)] = 1

        pruning_model.zero_grad()

        pbar = tqdm(range(n_iter), total=n_iter)
        print(('\n' + '%10s' * 7) % ('Stage', 'gpu_mem', 'channels', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total'))
        for i in pbar:

            imgs, targets, _, _ = data_iter.next()

            if len(targets) == 0:
                continue

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = origin_model(imgs)

            _, pruning_pred = pruning_model(imgs)
            pruning_loss, _ = compute_loss(pruning_pred, targets, pruning_model)

            hook_util.cat_to_gpu0()
            mse_loss = torch.zeros(1, device=device)

            aux_pred = aux_model(hook_util.prune_features['gpu0'][0], targets)
            aux_loss = loss_for_dcp_aux(aux_pred, targets)
            mse_loss += MSE(hook_util.prune_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss_scalar * aux_loss

            loss.backward()

            if i % accumulate == 0:
                solve_sub_problem_optimizer.step()
                solve_sub_problem_optimizer.zero_grad()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            mloss = (mloss * i + torch.cat([mse_loss, pruning_loss, aux_loss, loss]).detach()) / (i + 1)
            s = ('%10s' * 3 + '%10.3g' * 4) % (
                'SubProm ' + select_layer, '%.3gG' % mem, '%g/%g' % (i_k, retain_channels_num), *mloss)
            pbar.set_description(s)

            if i % 100 == 0:
                logger.info(('%10s' * 3 + '%10.3g' * 4) %
                            ('SubPro' + select_layer, str(i_k), '%g/%g' % (i, n_iter), mse_loss, pruning_loss, aux_loss,
                             loss))

            hook_util.clean_hook_out()

    for handle in handles:
        handle.remove()

    greedy_indices = pruning_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask < 1
    pruning_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[greedy_indices] = 0

    res, _ = test.test(prune_cfg,
                       data,
                       batch_size=batch_size * 2,
                       img_size=416,
                       model=pruning_model,
                       conf_thres=0.1,
                       iou_thres=0.5,
                       save_json=False,
                       dataloader=None)

    chkpt = torch.load(progress_chkpt, map_location=device)
    chkpt['current_layer'] = aux_util.next_prune_layer(select_layer)
    chkpt['epoch'] = -1
    chkpt['model'] = pruning_model.module.state_dict() if type(
        pruning_model) is nn.parallel.DistributedDataParallel else pruning_model.state_dict()
    chkpt['optimizer'] = None
    chkpt['prune_guide'] = aux_util.state()

    torch.save(chkpt, progress_chkpt)

    torch.save(chkpt, last)
    del chkpt

    with open(progress_result, 'a') as f:
        f.write(('%10s' * 2 + '%10.3g' * 8) %
                ('Pruning ' + select_layer,
                 str(aux_util.prune_guide[select_layer]['base_channels']) + '->' + str(retain_channels_num), *mloss,
                 *res[:4]) + '\n')

    torch.cuda.empty_cache()


def get_thin_model(cfg, data, origin_weights, img_size, batch_size, prune_rate, aux_epochs=50, ft_epochs=15,
                   resume=False, cache_images=False, start_layer='75'):
    init_seeds()

    # -----------------dataset-----------------
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']

    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_labels=True,
                                  cache_images=cache_images)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               num_workers=nw,
                                               shuffle=True,  # Shuffle=True unless rectangular training is used
                                               pin_memory=True,
                                               collate_fn=dataset.collate_fn)

    test_loader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, img_size, batch_size * 2,
                                                                  hyp=hyp,
                                                                  rect=True,
                                                                  cache_labels=True,
                                                                  cache_images=cache_images),
                                              batch_size=batch_size * 2,
                                              num_workers=nw,
                                              pin_memory=True,
                                              collate_fn=dataset.collate_fn)
    # -----------------dataset-----------------

    # -----------get trained aux net-----------
    if aux_trained:
        aux_chkpt = torch.load(aux_weight)
        if aux_chkpt["epoch"] + 1 != aux_epochs:
            del aux_chkpt
            train_for_aux(cfg, train_loader, origin_weights, aux_weight, device, resume=True, epochs=aux_epochs)
        else:
            del aux_chkpt
    else:
        train_for_aux(cfg, train_loader, origin_weights, aux_weight, device, resume=False, epochs=aux_epochs)
    # -----------get trained aux net-----------

    # ----------init model and aux util----------
    origin_model = Darknet(cfg).to(device)
    chkpt = torch.load(origin_weights, map_location=device)
    origin_model.load_state_dict(chkpt['model'], strict=True)
    aux_util = AuxNetUtils(origin_model)
    del chkpt
    # ----------init model and aux net----------

    mask_cfg, init_state_dict = mask_cfg_and_converted(aux_util.mask_replace_layer, cfg, origin_weights, target=None)

    # ----------start from first layer----------
    if not resume:
        first_progress = {'current_layer': start_layer,
                          'epoch': -1,
                          'model': init_state_dict,
                          'optimizer': None,
                          'prune_guide': aux_util.state()}
        aux_chkpt = torch.load(aux_weight)
        for k, v in aux_chkpt.items():
            if 'aux' in k:
                first_progress[k] = v
        del aux_chkpt
        torch.save(first_progress, progress_chkpt)

        with open(progress_result, 'a') as f:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write('\n' + t + '\n')
        # ----------start from first layer----------

    layer = start_layer
    if start_layer == aux_util.pruning_layer[-1]:
        return mask_cfg, aux_util

    while int(layer) < int(aux_util.pruning_layer[-1]):
        layer = fine_tune(mask_cfg, data, aux_util, device, train_loader, test_loader, ft_epochs)
        channels_select(mask_cfg, data, origin_model, aux_util, device, train_loader, layer, prune_rate)

    return mask_cfg, aux_util


def prune(mask_cfg, progress_weights, mask_replace_layer):
    only_in = mask_replace_layer[-3:]
    mask_replace_layer = mask_replace_layer[: -2]

    device_in = torch.device('cpu')
    model = Darknet(mask_cfg)
    chkpt = torch.load(progress_weights, map_location=device_in)
    model.load_state_dict(chkpt['model'])

    new_cfg = parse_model_cfg(mask_cfg)

    for layer in mask_replace_layer[:-1]:
        assert isinstance(model.module_list[int(layer)][0], MaskConv2d), "Not a pruned model!"
        tail_layer = mask_replace_layer[mask_replace_layer.index(layer) + 1]
        assert isinstance(model.module_list[int(tail_layer)][0], MaskConv2d), "Not a pruned model!"
        in_channels_mask = model.module_list[int(layer)][0].selected_channels_mask
        out_channels_mask = model.module_list[int(tail_layer)][0].selected_channels_mask

        in_channels = int(torch.sum(in_channels_mask))
        out_channels = int(torch.sum(out_channels_mask))

        new_cfg[int(layer) + 1]['type'] = 'convolutional'
        new_cfg[int(layer) + 1]['filters'] = str(out_channels)

        new_conv = nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=model.module_list[int(layer)][0].kernel_size,
                             stride=model.module_list[int(layer)][0].stride,
                             padding=model.module_list[int(layer)][0].padding,
                             bias=False)
        thin_weight = model.module_list[int(layer)][0].weight[out_channels_mask.bool()]
        thin_weight = thin_weight[:, in_channels_mask.bool()]
        assert new_conv.weight.numel() == thin_weight.numel(), 'Do not match in shape!'
        new_conv.weight.data.copy_(thin_weight.data)

        new_batch = nn.BatchNorm2d(out_channels, momentum=0.1)
        new_batch.weight.data.copy_(model.module_list[int(layer)][1].weight[out_channels_mask.bool()].data)
        new_batch.bias.data.copy_(model.module_list[int(layer)][1].bias[out_channels_mask.bool()].data)
        new_batch.running_mean.copy_(model.module_list[int(layer)][1].running_mean[out_channels_mask.bool()].data)
        new_batch.running_var.copy_(model.module_list[int(layer)][1].running_var[out_channels_mask.bool()].data)
        new_module = nn.Sequential()
        new_module.add_module('Conv2d', new_conv)
        new_module.add_module('BatchNorm2d', new_batch)
        new_module.add_module('activation', model.module_list[int(layer)][2])
        model.module_list[int(layer)] = new_module

    for layer in only_in:
        assert isinstance(model.module_list[int(layer)][0], MaskConv2d), "Not a pruned model!"
        in_channels_mask = model.module_list[int(layer)][0].selected_channels_mask > 0.1
        in_channels = int(torch.sum(in_channels_mask))

        new_conv = nn.Conv2d(in_channels,
                             out_channels=model.module_list[int(layer)][0].out_channels,
                             kernel_size=model.module_list[int(layer)][0].kernel_size,
                             stride=model.module_list[int(layer)][0].stride,
                             padding=model.module_list[int(layer)][0].padding,
                             bias=False)
        new_conv.weight.data.copy_(model.module_list[int(layer)][0].weight[:, in_channels_mask.bool()].data)

        new_module = nn.Sequential()
        new_module.add_module('Conv2d', new_conv)
        new_module.add_module('BatchNorm2d', model.module_list[int(layer)][1])
        new_module.add_module('activation', model.module_list[int(layer)][2])
        model.module_list[int(layer)] = new_module

    write_cfg('cfg/pruned-yolov3.cfg', new_cfg)
    chkpt = {'epoch': -1,
             'best_fitness': None,
             'training_results': None,
             'model': model.state_dict(),
             'optimizer': None}
    torch.save(chkpt, '../weights/DCP/pruned-converted-voc.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune-rate', type=float, default=0.7)
    parser.add_argument('--start_layer', type=str, default='1')
    parser.add_argument('--aux-epochs', type=int, default=5)
    parser.add_argument('--ft-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-voc.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='../weights/converted-voc.pt', help='initial weights')
    parser.add_argument('--data', type=str, default='data/voc.data', help='*.data path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--resume', action='store_true', help='resume training from pruning.pt')
    opt = parser.parse_args()
    print(opt)
    device = select_device(opt.device, apex=False, batch_size=opt.batch_size)

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

    try:
        mask_cfg, aux_util = get_thin_model(cfg=opt.cfg,
                                            data=opt.data,
                                            origin_weights=opt.weights,
                                            img_size=opt.img_size,
                                            batch_size=opt.batch_size,
                                            prune_rate=opt.prune_rate,
                                            aux_epochs=opt.aux_epochs,
                                            ft_epochs=opt.ft_epochs,
                                            resume=opt.resume,
                                            cache_images=opt.cache_images,
                                            start_layer=opt.start_layer)
    except Exception:
        send_error_report(str(traceback.format_exc()))
