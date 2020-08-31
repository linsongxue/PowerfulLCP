from model_unit import CBAMModule, MaskConv2d
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torchvision.ops import roi_align
from models import Darknet
from utils.utils import wh_iou, bbox_iou, xywh2xyxy, box_iou
from utils.parse_config import parse_model_cfg
from utils.torch_utils import init_seeds, model_info
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy


# AuxNets
class HeadLayer(nn.Module):
    def __init__(self, num_classes, anchors):
        """

        :param num_classes: Int
        :param anchors: Numpy.array
        """
        super(HeadLayer, self).__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_feature = 4 + self.num_classes
        self.num_grid_x = 0
        self.num_grid_y = 0

    def forward(self, input, origin_img_size, var=None):
        bs, _, self.num_grid_y, self.num_grid_x = input.shape  # bs, _, 52, 52
        self._creat_grids(origin_img_size, (self.num_grid_y, self.num_grid_x), input.device, input.dtype)

        # [bs, 756, 52, 52] --> [bs, 9, 84, 52, 52]
        output = input.view(bs, self.num_anchors, self.num_feature, self.num_grid_y, self.num_grid_x)
        # [bs, 9, 84, 52, 52] --> [bs, 9, 52, 52, 84]
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        return output

    def _creat_grids(self, origin_img_size, ng, device, dtype):
        index_grid_y, index_grid_x = torch.meshgrid([torch.arange(ng[0]), torch.arange(ng[1])])
        self.offset_xy = torch.stack([index_grid_x, index_grid_y], dim=2).to(device).type(dtype)

        stride = max(origin_img_size) / max(ng)

        self.anchors_vec = self.anchors / stride
        self.anchors_vec = self.anchors_vec.to(device)
        self.ng = torch.tensor(ng).to(device)


class AuxNet(nn.Module):
    """
    The auxiliary network to help select the important channels
    """

    def __init__(self, in_channels, num_classes, hyp, anchors, origin_img_size, feature_maps_size=52):
        """
        init the network
        :param in_channels: Int
        :param num_classes: Int
        :param anchors: Numpy.array
        :param origin_img_size: Int
        :param feature_maps_size: Int
        """
        super(AuxNet, self).__init__()

        self.feature_maps_size = feature_maps_size
        self.origin_img_size = origin_img_size
        self.nc = num_classes
        self.hyp = hyp
        # Layer 0
        self.layer_0 = nn.Sequential()
        self.layer_0.add_module("Conv2d", nn.Conv2d(in_channels=in_channels,
                                                    out_channels=128,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=False))
        self.layer_0.add_module("BatchNorm2d", nn.BatchNorm2d(128, momentum=0.1))
        self.layer_0.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 1
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("CBAM", CBAMModule(in_channels=128,
                                                   out_channels=256,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   deepwidth=True))
        self.layer_1.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 2
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("CBAM", CBAMModule(in_channels=256,
                                                   out_channels=256,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0,
                                                   deepwidth=False))
        self.layer_2.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 3
        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("CBAM", CBAMModule(in_channels=256,
                                                   out_channels=256,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   deepwidth=True))
        self.layer_3.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 4
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("Conv2d", nn.Conv2d(in_channels=256,
                                                    out_channels=(self.nc + 4) * len(anchors),
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=True))
        # Head layer
        self.head = HeadLayer(self.nc, anchors)

    def forward(self, input1, input2=None):
        """
        generate the output
        :param input: Tensor, feature maps of last layer, size=[bs, channels, width, height]
        :return:
        """

        if input2 is not None:
            input1 = F.interpolate(input1, (self.feature_maps_size, self.feature_maps_size), mode='bilinear',
                                   align_corners=False)
            input2 = F.interpolate(input2, (self.feature_maps_size, self.feature_maps_size), mode='bilinear',
                                   align_corners=False)
            input = torch.cat([input1, input2], dim=1)
        else:
            input = F.interpolate(input1, (self.feature_maps_size, self.feature_maps_size), mode='bilinear',
                                  align_corners=False)
        # TODO:replace the interpolate with average pooling
        x = self.layer_0(input)
        short = self.layer_1(x)
        x = self.layer_2(short)
        x = self.layer_3(x)
        x = x + short
        x = self.layer_4(x)
        output = self.head(x, (self.origin_img_size, self.origin_img_size))

        return output


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


class LCPAuxNet(nn.Module):
    def __init__(self, in_channels, num_classes, anchors, iou_threshold=0.5):
        super(LCPAuxNet, self).__init__()
        self.iou_threshold = iou_threshold
        self.anchors = anchors
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, input, targets, predictions):
        _, _, h, w = input.shape
        o_h, o_w = torch.mean(targets[:, 3] * h), torch.mean(targets[:, 4] * w)
        boxes1, boxes2, loc_loss = self.get_LCP_area(targets.clone().detach(), predictions, self.anchors, w, h)
        feat1 = roi_align(input, boxes1, output_size=(o_h, o_w))
        feat2 = roi_align(input, boxes2, output_size=(o_h, o_w))
        out = self.bn(feat1 + feat2)
        out = self.act(out)
        out = self.pooling(out)
        out = self.linear(out.squeeze())

        return out, loc_loss

    def get_LCP_area(self, targets, predictions, anchors, feature_w, feature_h):
        pred_info = []
        # anchors转化为归一形式
        anchors = torch.from_numpy(anchors).to(targets.dtype).to(targets.device) / 416.0
        # anchors_vec是在特征图上的anchors
        anchors_vec = anchors * torch.tensor([feature_h, feature_w], dtype=anchors.dtype, device=anchors.device)
        # 计算一系列索引值
        gwh = targets[:, 4:6]
        iou_anchors = wh_iou(anchors, gwh)
        _, idx_a = iou_anchors.max(0)
        idx_p = idx_a // 3
        idx_a = idx_a % 3
        idx_b = targets[:, 0].long()
        # 选择特定的框
        for i, p in enumerate(predictions):
            mask = idx_p == i
            idx_x, idx_y = (targets[:, 2] * p.size(-2)).long(), (targets[:, 3] * p.size(-3)).long()
            pred_info.append(p[idx_b[mask], idx_a[mask], idx_y[mask], idx_x[mask]])
        pred_info = torch.cat(pred_info, dim=0)

        # 此处开始pred_info_detach就只有选框作用，不再需要反向传播
        pred_info_detach = pred_info.clone().detach()
        pred_info_detach[:, 0:2] = torch.sigmoid(pred_info_detach[:, 0:2]) + torch.stack([idx_x, idx_y], dim=0).t()
        pred_info_detach[:, 2:4] = torch.exp(pred_info_detach[:, 2:4]).clamp(max=1E3) * anchors_vec[(idx_p + idx_a)]

        # 将标签转化为适应特征图的形式，之前是归一化的
        targets[:, [2, 4]] *= feature_w
        targets[:, [3, 5]] *= feature_h
        # 计算IoU大于0.5的
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        pred_info_detach[:, :4] = xywh2xyxy(pred_info_detach[:, :4])

        boxes_union, boxes_gt = torch.zeros([len(targets), 5], device=targets.device), torch.zeros([len(targets), 5],
                                                                                                   device=targets.device)
        boxes_gt[:, 0] = idx_b
        boxes_gt[:, 1:] = targets[:, 2:]

        boxes_union[:, 0] = idx_b
        boxes_union[:, 1:3] = torch.min(pred_info_detach[:, :2], targets[:, 2:4])
        boxes_union[:, 3:5] = torch.max(pred_info_detach[:, 2:4], targets[:, 4:6])

        giou = bbox_iou(torch.cat([torch.sigmoid(pred_info[:, 0:2]) + torch.stack([idx_x, idx_y], dim=0).t(),
                                   torch.exp(pred_info[:, 2:4]).clamp(max=1E3) * anchors_vec[(idx_p + idx_a)]],
                                  dim=1).t(),
                        targets[:, 2:6], x1y1x2y2=False, GIoU=True)

        return boxes_gt, boxes_union, (1 - giou).mean()


class AuxNetUtils(object):

    def __init__(self, model, hyp, backbone="DarkNet53", neck="FPN", strategy="MFCP", prune_short=False):
        self.model = model
        self.hyp = hyp
        self.num_classes = int(self.model.module_defs[-1]['classes'])
        self.anchors = self.model.module_defs[-1]['anchors']

        if backbone == "DarkNet53":
            self.__down_sample = ['12', '37', '62', '75']
        elif backbone == "DarkNet19":
            self.__down_sample = ['6', '8', '10', '12']
        else:
            raise KeyError

        if neck == "FPN":
            self.__fusion_layer = ['87', '99']
        elif neck == "PAN":
            self.__fusion_layer = ['85', '94']
        elif neck == "FPN-tiny":
            self.__fusion_layer = ['21']
        else:
            raise KeyError

        self.strategy = strategy
        if strategy == "MFCP":
            try:
                self.aux_in_layer = [self.__down_sample[0],
                                     [self.__down_sample[1], self.__fusion_layer[1]],
                                     [self.__down_sample[2], self.__fusion_layer[0]],
                                     self.__down_sample[3]]
            except IndexError:
                self.aux_in_layer = [self.__down_sample[0],
                                     self.__down_sample[1],
                                     [self.__down_sample[2], self.__fusion_layer[0]],
                                     self.__down_sample[3]]
        elif strategy == "DCP" or strategy == "LCP":
            self.aux_in_layer = self.__down_sample
        else:
            raise KeyError

        self.sync_guide = dict()
        try:
            self.sync_guide[self.__down_sample[1]] = self.__fusion_layer[1]
        except Exception:
            pass
        self.sync_guide[self.__down_sample[2]] = self.__fusion_layer[0]

        self.prune_short = prune_short
        self.layer_info = OrderedDict()
        self.conv_layer_dict = OrderedDict()
        # 需要裁剪的层
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

        short_cut_idx = []
        # 为每一个需要裁剪的层分配辅助网络的附着层
        last_darknet_layer = 75
        ignore_short_idx = set()
        for name, value in self.layer_info.items():
            if int(name) <= last_darknet_layer and value["shortcut"]:
                ignore_short_idx.add(int(name) + int(value["from"]) + 1)
                ignore_short_idx.add(int(name) + 1)
            elif int(name) <= int(self.__down_sample[0]):
                self.conv_layer_dict[name] = self.aux_in_layer[0]
                self.pruning_layer.append(name)
            elif int(name) <= int(self.__down_sample[1]):
                self.conv_layer_dict[name] = self.aux_in_layer[1]
                self.pruning_layer.append(name)
            elif int(name) <= int(self.__down_sample[2]):
                self.conv_layer_dict[name] = self.aux_in_layer[2]
                self.pruning_layer.append(name)
            elif int(name) <= int(self.__down_sample[3]):
                self.conv_layer_dict[name] = self.aux_in_layer[3]
                self.pruning_layer.append(name)

        if self.prune_short:
            pass
        else:
            pruning_layer_copy = deepcopy(self.pruning_layer)
            for name in pruning_layer_copy:
                if int(name) in ignore_short_idx:
                    self.pruning_layer.remove(name)

        # 第一层不需要减
        self.pruning_layer.pop(0)
        # 需要将原有网络替换为MaskConv2d
        self.mask_replace_layer = list(self.conv_layer_dict.keys()) + self.__fusion_layer

    def next_prune_layer(self, layer):
        idx = self.pruning_layer.index(layer)
        try:
            return self.pruning_layer[idx + 1]
        except IndexError:
            return 'end'

    def creat_aux_model(self, layer_name, img_size=416, feature_maps_size=52):
        if self.strategy == "MFCP" and isinstance(layer_name, str):
            return AuxNet(self.layer_info[layer_name]['out_channels'],
                          self.num_classes,
                          self.hyp,
                          self.anchors,
                          img_size,
                          feature_maps_size)

        elif self.strategy == "MFCP" and isinstance(layer_name, list):
            return AuxNet(sum([self.layer_info[x]['out_channels'] for x in layer_name]),
                          self.num_classes,
                          self.hyp,
                          self.anchors,
                          img_size,
                          feature_maps_size)

        elif self.strategy == "LCP":
            return LCPAuxNet(self.layer_info[layer_name]['out_channels'], self.num_classes, self.anchors)

        elif self.strategy == "DCP":
            return DCPAuxNet(self.layer_info[layer_name]['out_channels'], self.num_classes)


# MFCP utils-----------------------------------
def build_targets_for_MFCP(model, targets):
    hyp = model.hyp
    iou_thr = hyp['iou_t']  # iou threshold
    head = model.head
    num_anchors = len(head.anchors_vec)
    anchors_index = torch.arange(num_anchors).view((-1, 1)).repeat([1, len(targets)]).view(-1)  #
    ground_truth = targets.repeat([num_anchors, 1])

    gwh = targets[:, 4:6] * head.ng
    # iou = torch.stack([wh_iou(anchor, gwh) for anchor in head.anchors_vec])
    iou = wh_iou(head.anchors_vec, gwh)
    mask_satisfy = iou.view(-1) > iou_thr

    # 正确标签的选择
    ground_truth = ground_truth[mask_satisfy]  # 此后这个变量就是IoU大于超参数的阈值的标签
    txy = ground_truth[:, 2:4]  # xy还需要再修正一下
    twh = ground_truth[:, 4:6]

    # 各种索引
    anchors_index = anchors_index[mask_satisfy]
    anchors_vec = head.anchors_vec[anchors_index]
    batch_index, tcls = ground_truth[:, 0:2].long().t()
    x_index, y_index = txy.long().t()
    index = (batch_index, anchors_index, y_index, x_index)

    # 修正xy
    txy -= txy.floor()
    tbox = torch.cat([txy, twh], dim=1)

    return txy, twh, tcls, tbox, index, anchors_vec


def compute_loss_for_MFCP(output, targets, aux_model):
    ft = torch.cuda.FloatTensor if output[0].is_cuda else torch.Tensor
    lcls, lbox = ft([0]), ft([0])
    if type(aux_model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
        aux_model = aux_model.module  # aux 的超参数是模型内自带的，所以需要脱分布式训练的壳
    hyp = aux_model.hyp
    ft = torch.cuda.FloatTensor if output.is_cuda else torch.Tensor
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([hyp['cls_pw']]), reduction='sum')
    txy, twh, tcls, tbox, index, anchors_vec = build_targets_for_MFCP(aux_model, targets)
    b, a, j, i = index
    nb = len(b)
    if nb:
        pn = output[b, a, j, i]  # predict needed
        pxy = torch.sigmoid(pn[:, 0:2])
        pbox = torch.cat([pxy, torch.exp(pn[:, 2:4]).clamp(max=1E3) * anchors_vec], dim=1)
        DIoU = bbox_iou(pbox.t(), tbox, x1y1x2y2=False, DIoU=True)

        lbox += (1 - DIoU).sum()

        tclsm = torch.zeros_like(pn[:, 4:])
        tclsm[range(len(b)), tcls] = 1.0

        lcls += BCEcls(pn[:, 4:], tclsm)

    lbox *= hyp['diou']
    lcls *= hyp['cls']

    if nb:
        lbox /= nb
        lcls /= (nb * aux_model.nc)

    loss = lbox + lcls

    return loss, torch.cat((lbox, lcls, loss)).clone().detach()


# DCP utils-----------------------------------

def compute_loss_for_DCP(out_put, targets):
    loss = torch.zeros(1, device=out_put.device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    ground_truth = targets[:, 1]
    loss += criterion(out_put, ground_truth.long())

    return loss


# LCP utils-----------------------------------
def compute_loss_for_LCP(out_put, loc_loss, targets):
    loss = torch.zeros(1, device=out_put.device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    ground_truth = targets[:, 1]
    loss += criterion(out_put, ground_truth.long()) + 50 * loc_loss

    return loss


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


def train_aux_for_MFCP(cfg, backbone, neck, data_loader, weights, aux_weight, hyp, device, resume, epochs):
    init_seeds()
    batch_size = data_loader.batch_size
    accumulate = 64 // batch_size
    img_size = data_loader.dataset.img_size

    model = Darknet(cfg).to(device)
    model_chkpt = torch.load(weights, map_location=device)
    model.load_state_dict(model_chkpt['model'], strict=True)
    del model_chkpt
    aux_util = AuxNetUtils(model, hyp, backbone, neck)
    hook_util = HookUtils()

    start_epoch = 0

    aux_model_list = []
    pg0, pg1 = [], []  # optimizer parameter groups
    for layer in aux_util.aux_in_layer:
        aux_model = aux_util.creat_aux_model(layer, img_size)
        aux_model.to(device)
        for k, v in dict(aux_model.named_parameters()).items():
            if 'Conv2d.weight' in k:
                pg1 += [v]  # parameter group 1 (apply weight_decay)
            else:
                pg0 += [v]
        aux_model_list.append(aux_model)

    optimizer = optim.SGD(pg0, lr=aux_util.hyp['lr0'], momentum=aux_util.hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': aux_util.hyp['weight_decay']})
    del pg0, pg1

    if resume:
        chkpt = torch.load(aux_weight, map_location=device)

        for i, layer in enumerate(aux_util.aux_in_layer):
            if isinstance(layer, str):
                aux_model_list[i].load_state_dict(chkpt['aux_in{}'.format(layer)], strict=True)
            else:
                aux_model_list[i].load_state_dict(chkpt['aux_in{}'.format(layer[0])], strict=True)

        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])

        start_epoch = chkpt['epoch'] + 1

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 3, 2 * epochs // 3], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    handles = []  # 结束训练后handle需要回收
    for name, child in model.module_list.named_children():
        if name in aux_util.aux_in_layer:
            handles.append(child.register_forward_hook(hook_util.hook_origin_output))
        elif name in aux_util.aux_in_layer[1] or name in aux_util.aux_in_layer[2]:
            handles.append(child.register_forward_hook(hook_util.hook_origin_output))

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        for i, aux_model in enumerate(aux_model_list):
            aux_model_list[i] = nn.parallel.DistributedDataParallel(aux_model, find_unused_parameters=True)

        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers

    nb = len(data_loader)
    model.nc = 80
    model.hyp = aux_util.hyp
    model.arc = 'default'
    for aux_model in aux_model_list:
        model_info(aux_model, report='summary')
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):

        for aux_model in aux_model_list:
            aux_model.train()
        print(('\n' + '%10s' * 8) % ('Stage', 'Epoch', 'gpu_mem', 'AuxID', 'DIoU', 'cls', 'total', 'targets'))

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
                if aux_idx == 0:
                    pred = aux_model(hook_util.origin_features['gpu0'][0])
                elif aux_idx == 1:
                    pred = aux_model(hook_util.origin_features['gpu0'][1], hook_util.origin_features['gpu0'][-1])
                elif aux_idx == 2:
                    pred = aux_model(hook_util.origin_features['gpu0'][2], hook_util.origin_features['gpu0'][-2])
                else:
                    pred = aux_model(hook_util.origin_features['gpu0'][3])

                loss, loss_items = compute_loss_for_MFCP(pred, targets, aux_model)
                loss *= batch_size / 64

                loss.backward()

                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s = ('%10s' * 3 + '%10.3g' * 5) % (
                    'Train Aux', '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, aux_idx, *loss_items, len(targets))
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
            if isinstance(layer, str):
                chkpt['aux_in{}'.format(layer)] = aux_model_list[i].module.state_dict() if type(
                    aux_model_list[i]) is nn.parallel.DistributedDataParallel else aux_model_list[i].state_dict()
            else:
                chkpt['aux_in{}'.format(layer[0])] = aux_model_list[i].module.state_dict() if type(
                    aux_model_list[i]) is nn.parallel.DistributedDataParallel else aux_model_list[i].state_dict()
        torch.save(chkpt, aux_weight)
        torch.save(chkpt, "../weights/aux-coco.pt")
        del chkpt

        with open("aux_result.txt", 'a') as f:
            f.write(s + '\n')

    # 最后要把hook全部删除
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()


def train_aux_for_DCP(cfg, backbone, neck, data_loader, weights, aux_weight, hyp, device, resume, epochs):
    init_seeds()
    batch_size = data_loader.batch_size
    accumulate = 64 // batch_size

    model = Darknet(cfg).to(device)
    model_chkpt = torch.load(weights, map_location=device)
    model.load_state_dict(model_chkpt['model'], strict=True)
    del model_chkpt
    aux_util = AuxNetUtils(model, hyp, backbone, neck, strategy="DCP")
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
                loss = compute_loss_for_DCP(pred, targets)

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

        torch.save(chkpt, "../weights/DCP/aux-coco.pt")
        del chkpt

        with open("./DCP/aux_result.txt", 'a') as f:
            f.write(s + '\n')

    # 最后要把hook全部删除
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()


def train_aux_for_LCP(cfg, backbone, neck, data_loader, weights, aux_weight, hyp, device, resume, epochs):
    init_seeds()
    batch_size = data_loader.batch_size
    accumulate = 64 // batch_size

    model = Darknet(cfg).to(device)
    model_chkpt = torch.load(weights, map_location=device)
    model.load_state_dict(model_chkpt['model'], strict=True)
    del model_chkpt
    aux_util = AuxNetUtils(model, hyp, backbone, neck, strategy="LCP")
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
        model.train()
        for i, (imgs, targets, _, _) in pbar:

            if len(targets) == 0:
                continue

            ni = i + nb * epoch
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                prediction = model(imgs)

            hook_util.cat_to_gpu0()
            for aux_idx, aux_model in enumerate(aux_model_list):
                pred, loc_loss = aux_model(hook_util.origin_features['gpu0'][aux_idx], targets, prediction)
                loss = compute_loss_for_LCP(pred, loc_loss, targets)

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

        torch.save(chkpt, "../weights/LCP/aux-coco.pt")
        del chkpt

        with open("./LCP/aux_result.txt", 'a') as f:
            f.write(s + '\n')

    # 最后要把hook全部删除
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()


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


def prune(mask_cfg, progress_weights, mask_replace_layer, new_cfg_file, new_weights):
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
        new_cfg[int(layer) + 1]['type'] = 'convolutional'
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

    write_cfg(new_cfg_file, new_cfg)
    chkpt = {'epoch': -1,
             'best_fitness': None,
             'training_results': None,
             'model': model.state_dict(),
             'optimizer': None}
    torch.save(chkpt, new_weights)
