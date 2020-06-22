from model_unit import CBAMModule
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from models import Darknet
from utils.utils import wh_iou, bbox_iou
from utils.parse_config import parse_model_cfg, parse_data_cfg
from utils.torch_utils import init_seeds, model_info
from utils.datasets import LoadImagesAndLabels
from tqdm import tqdm
from collections import OrderedDict
import math
import os.path as path
import os


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


class AuxNetUtils(object):

    def __init__(self, model, hyp):
        self.model = model
        self.hyp = hyp
        self.num_classes = int(self.model.module_defs[-1]['classes'])
        self.anchors = self.model.module_defs[-1]['anchors']
        self.__down_sample = ['12', '37', '62', '75']
        self.__fusion_layer = ['87', '99']
        self.aux_in_layer = ['12', ['37', '99'], ['62', '87'], '75']
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
                    self.conv_layer_dict[name] = [self.__down_sample[1], self.__fusion_layer[1]]
                elif int(name) <= int(self.__down_sample[2]):
                    self.conv_layer_dict[name] = [self.__down_sample[2], self.__fusion_layer[0]]
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

    def compute_retain_channels(self, layer, rate):
        if self.prune_guide[layer]['pruned']:
            raise Exception

        if 'tail_link' in self.prune_guide[layer].keys():
            sample_layer = self.prune_guide[layer]['tail_link']
            if not self.prune_guide[sample_layer]['pruned']:
                raise Exception
            if self.prune_guide[sample_layer]['base_channels'] == self.prune_guide[sample_layer]['retain_channels']:
                raise Exception
            retain_channels = self.prune_guide[sample_layer]['retain_channels']
            self.prune_guide[layer]['retain_channels'] = retain_channels
            self.prune_guide[layer]['pruned'] = True

            return self.prune_guide[layer]['retain_channels']

        else:
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

    def creat_aux_model(self, layer_name, img_size, feature_maps_size=52):
        if isinstance(layer_name, str):
            aux_model = AuxNet(self.layer_info[layer_name]['out_channels'],
                               self.num_classes,
                               self.hyp,
                               self.anchors,
                               img_size,
                               feature_maps_size)
            return aux_model

        elif isinstance(layer_name, list):
            in_channels = sum([self.layer_info[x]['out_channels'] for x in layer_name])
            aux_model = AuxNet(in_channels,
                               self.num_classes,
                               self.hyp,
                               self.anchors,
                               img_size,
                               feature_maps_size)
            return aux_model


# utils-----------------------------------

def build_targets_for_aux(model, targets):
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


def compute_loss_for_aux(output, targets, aux_model):
    ft = torch.cuda.FloatTensor if output[0].is_cuda else torch.Tensor
    lcls, lbox = ft([0]), ft([0])
    if type(aux_model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
        aux_model = aux_model.module  # aux 的超参数是模型内自带的，所以需要脱分布式训练的壳
    hyp = aux_model.hyp
    ft = torch.cuda.FloatTensor if output.is_cuda else torch.Tensor
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([hyp['cls_pw']]), reduction='sum')
    txy, twh, tcls, tbox, index, anchors_vec = build_targets_for_aux(aux_model, targets)
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


def train_for_aux(cfg, data_loader, weights, aux_weight, img_size, hyp, device, resume, epochs):
    init_seeds()
    batch_size = data_loader.batch_size
    accumulate = 64 // batch_size

    model = Darknet(cfg).to(device)
    model_chkpt = torch.load(weights, map_location=device)
    model.load_state_dict(model_chkpt['model'], strict=True)
    del model_chkpt
    aux_util = AuxNetUtils(model, hyp)
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

                loss, loss_items = compute_loss_for_aux(pred, targets, aux_model)
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
        torch.save(chkpt, "../weights/aux-voc.pt")
        del chkpt

        with open("aux_result.txt", 'a') as f:
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
