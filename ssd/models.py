import torch
import torch.nn as nn
import math
from itertools import product
from model_unit import MaskConv2d
from utils.parse_config import parse_model_cfg
from torchsummaryX import summary


def create_modules(module_defs, img_size):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []
    num_classes = 0

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=pad,
                                                   groups=int(mdef['groups']) if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))

        elif mdef['type'] == 'maskconvolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('MaskConv2d', MaskConv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=size,
                                                        stride=stride,
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))

        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            filters = output_filters[-1]
            modules = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])

        elif mdef['type'] == 'shortcut':  # nn.ReLU placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])
            if mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU(inplace=True))

        elif mdef['type'] == 'ssd':
            filters = output_filters[-1]
            modules = SSDHead(in_channels=output_filters[-1],
                              num_classes=int(mdef['classes']),
                              num_anchors=int(mdef['anchors']),
                              feature_size=int(mdef['feature']),
                              img_size=img_size,
                              min_size=int(mdef['min']),
                              max_size=int(mdef['max']))
            num_classes = int(mdef['classes'])


        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    assert num_classes > 0, "Number of class has something wrong!"

    return module_list, routs, num_classes


class SSDHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors, feature_size, img_size, min_size, max_size):
        super(SSDHead, self).__init__()
        self.location = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.conference = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3, stride=1, padding=1)
        self.feature_size = feature_size
        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        if num_anchors == 6:
            self.aspect_ratios = [2, 3]
        elif num_anchors == 4:
            self.aspect_ratios = [2]
        else:
            raise KeyError

    def _prior_boxes(self, device, dtype):
        mean = []
        for i, j in product(range(self.feature_size), repeat=2):
            # unit center x,y
            cx = (j + 0.5) / self.feature_size
            cy = (i + 0.5) / self.feature_size

            # aspect_ratio: 1
            # rel size: min_size
            s_k = self.min_size / self.img_size
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (self.max_size / self.img_size))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in self.aspect_ratios:
                mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]

        out = torch.tensor(mean, dtype=dtype, device=device).view(-1, 4)
        out.clamp_(min=0, max=1)
        return out

    def forward(self, input):
        priors = self._prior_boxes(input.device, input.dtype)
        loc = self.location(input)
        conf = self.conference(input)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        conf = conf.permute(0, 2, 3, 1).contiguous()

        output = (loc, conf, priors)

        return output


class SSD(nn.Module):
    def __init__(self, cfg, img_size):
        super(SSD, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs, self.num_cls = create_modules(self.module_defs, img_size)

    def forward(self, input):
        layer_outputs = []
        output = []
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'maxpool', 'maskconvolutional']:
                input = module(input)
            elif mtype == 'route':
                layers = int(mdef['layers'])
                input = layer_outputs[layers]
                # layers = [int(x) for x in mdef['layers'].split(',')]
                # assert len(layers) == 1, "SSD don not support two route two layers"
                # input = layer_outputs[layers[0]]
            elif mtype == 'shortcut':
                input = module(input + layer_outputs[int(mdef['from'])])
            elif mtype == 'ssd':
                output.append(module(input))
            layer_outputs.append(input if i in self.routs else [])

        loc, conf, priors = list(zip(*output))

        loc = torch.cat([l.view(l.size(0), -1) for l in loc], 1)
        conf = torch.cat([c.view(c.size(0), -1) for c in conf], 1)
        priors = torch.cat([p for p in priors], 0)

        if self.training:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_cls + 1),
                      priors)
        else:
            output = None

        return output


# model = SSD('cfg/ssd-res50.cfg', 300)
# print(model)
# model.train()
# # print(model.module_defs)
# # print(model)
# # summary(model, torch.Tensor(1, 3, 300, 300))
# a = torch.ones(1, 3, 300, 300)
# out = model(a)
# print(out[0].shape)
# print(out[1].shape)
# print(out[2].shape)

# chkpt = torch.load('resnet50.pth')
# for k, v in chkpt.items():
#     print(k)
# for k, v in model.state_dict().items():
#     print(k)
from torchvision import models

modelr = models.resnet50(pretrained=True)
for k, v in modelr.named_children():
    print(k, '-->', isinstance(v, nn.Sequential))
