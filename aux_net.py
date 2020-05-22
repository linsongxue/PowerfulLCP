from model_unit import CBAMModule
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
from models import Darknet
from utils.utils import wh_iou, bbox_iou
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler


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

    def __init__(self, in_channels, num_classes, anchors, origin_img_size, feature_maps_size=52):
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
        self.layer_1.add_module("CBAMConv2d", CBAMModule(in_channels=128,
                                                         out_channels=256,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1,
                                                         deepwidth=True))
        self.layer_1.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 2
        self.layer_2 = nn.Sequential()
        self.layer_2.add_module("CBAMConv2d", CBAMModule(in_channels=256,
                                                         out_channels=256,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0,
                                                         deepwidth=False))
        self.layer_2.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 3
        self.layer_3 = nn.Sequential()
        self.layer_3.add_module("CBAMConv2d", CBAMModule(in_channels=256,
                                                         out_channels=256,
                                                         kernel_size=3,
                                                         stride=1,
                                                         padding=1,
                                                         deepwidth=True))
        self.layer_3.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        # Layer 4
        self.layer_4 = nn.Sequential()
        self.layer_4.add_module("Conv2d", nn.Conv2d(in_channels=256,
                                                    out_channels=(num_classes + 4) * len(anchors),
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=1,
                                                    bias=True))
        # Head layer
        self.head = HeadLayer(num_classes, anchors)

    def forward(self, input):
        """
        generate the output
        :param input: Tensor, feature maps of last layer, size=[bs, channels, width, height]
        :return:
        """

        input = F.interpolate(input, (self.feature_maps_size, self.feature_maps_size), mode='bilinear',
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
        if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
            model = model.module
        self.model = model
        self.num_classes = int(self.model.module_defs[-1]['classes'])
        self.anchors = self.model.module_defs[-1]['anchors']
        self.hyp = hyp
        self.hook_out = {}
        for i in range(torch.cuda.device_count()):
            self.hook_out['gpu{}'.format(i)] = []
        self.layer_info = []  # 元素是字典
        self.short_cut_info = {}
        self.down_sample_layer = []  # 元素是str
        self.conv_layer = []  # 最后会变成字典，key是conv层的名字，value是joint loss序号
        self.pruning_layer = []  # 需要实行剪枝的层
        self.aux_list = []  # 元素是nn.Module
        self.analyze()

    def hook_forward(self, module, input, output):
        self.hook_out['gpu{}'.format(input[0].device.index)].append(input[0])

    def clean_hook_out(self):
        for key in list(self.hook_out.keys()):
            self.hook_out[key] = []

    @staticmethod
    def child_analyse(child):
        """
        Analyse the module to get some useful params
        :param child:Module, the module of all net
        :return:Dictionary, {"in_channels": Int,
                             "out_channels": Int,
                             "kernel_size": Int,
                             "stride": Int,
                             "padding": Int}
        """
        analyse_dict = {"shortcut": True}  # 短接层初始化
        for name, module in child.named_children():
            if isinstance(module, nn.Conv2d):
                analyse_dict = {"in_channels": module.in_channels,
                                "out_channels": module.out_channels,
                                "kernel_size": module.kernel_size[0],
                                "stride": module.stride[0],
                                "padding": (module.kernel_size[0] - 1) // 2,
                                "shortcut": False}

        return analyse_dict

    def analyze(self):
        """
        对模型进行分析，将所有关键信息提取，主要是完成了self.layer_info和self.down_sample_layer的填充
        两个都是列表，第一个内容是字典，第二个记录了下采样的层序号
        :return: None
        """
        last_darknet = 75
        for name, child in self.model.module_list.named_children():
            if int(name) <= last_darknet:  # DarkNet后面的第一层，因为选用的是输入特征图，所以要多用一层
                info = AuxNetUtils.child_analyse(child)
                self.layer_info.append(info)
                if not info["shortcut"]:  # 记录非跨越层
                    self.conv_layer.append(str(name))
                    if info["stride"] == 2 or int(name) == last_darknet:  # 记录下采样层
                        self.down_sample_layer.append(str(name))

        defs = self.model.module_defs
        for idx, definition in enumerate(defs):
            if idx <= last_darknet and definition['type'] == 'shortcut':
                out = idx - 1
                short = idx + int(definition['from'])
                while self.model.module_defs[short]['type'] == "shortcut":
                    short += int(self.model.module_defs[short]['from'])
                if str(short) in self.short_cut_info:
                    self.short_cut_info[str(short)].append(str(out))
                else:
                    self.short_cut_info[str(short)] = [str(out)]

        index = 0
        joint_loss_index = []
        for layer in self.conv_layer:
            if int(layer) < int(self.down_sample_layer[index]):
                joint_loss_index.append(index)
            else:
                index += 1
                joint_loss_index.append(index)

        self.conv_layer = dict(zip(self.conv_layer, joint_loss_index))
        self.conv_layer.pop(str(last_darknet))
        for key, value in self.short_cut_info.items():
            for layer in value:
                self.conv_layer.pop(layer)

        self.pruning_layer = list(self.conv_layer.keys())

    def creat_aux_list(self, img_size, device, feature_maps_size=52, conv_layer_name=None):
        """
        需要在初始化类后调用一次，以创造辅助列表
        :param device:
        :param feature_maps_size:
        :param conv_layer_name:
        :return:
        """
        if conv_layer_name is None:
            for layer in self.down_sample_layer:
                in_channels = self.layer_info[int(layer)]["in_channels"]
                aux_net = AuxNet(in_channels, self.num_classes, self.anchors, img_size, feature_maps_size).to(device)
                aux_net.hyp = self.hyp
                aux_net.nc = self.num_classes
                self.aux_list.append(aux_net)
        else:
            aux_idx = self.conv_layer[conv_layer_name]
            layer = self.down_sample_layer[aux_idx]
            in_channels = self.layer_info[int(layer)]["in_channels"]
            aux_net = AuxNet(in_channels, self.num_classes, self.anchors, img_size, feature_maps_size).to(device)
            aux_net.hyp = self.hyp
            aux_net.nc = self.num_classes

            return aux_net

    def distributed(self):
        for i, net in enumerate(self.aux_list):
            self.aux_list[i] = nn.parallel.DistributedDataParallel(net, find_unused_parameters=True)
            # self.aux_list[i].head = self.aux_list[i].module.head

    def load_aux_weight(self, check_point):
        assert self.aux_list, "Please firstly call 'Class.creat_aux_list'!"
        for i in range(len(self.aux_list)):
            self.aux_list[i].load_state_dict(check_point["aux{}".format(i)], strict=True)

    # def create_optimizer_for_fine_tune(self):
    #     assert self.aux_list, "Please call 'Class.creat_aux_list' firstly!"
    #     if self.optimizer_list:
    #         self.optimizer_list = []
    #     for i, net in enumerate(self.aux_list):
    #         pg0, pg1 = [], []
    #         for k, v in dict(net.named_parameters()).items():
    #             if 'Conv2d.weight' in k:
    #                 pg1 += [v]
    #             else:
    #                 pg0 += [v]
    #         optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)
    #         optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})
    #         self.optimizer_list.append(optimizer)
    #         del pg0, pg1

    def cat_to_gpu0(self):
        device_list = list(self.hook_out.keys())
        for i in range(len(self.hook_out['gpu0'])):
            for device in device_list[1:]:
                self.hook_out['gpu0'][i] = torch.cat([self.hook_out['gpu0'][i], self.hook_out[device][i].cuda(0)],
                                                     dim=0)

    def next_conv_layer(self, current_layer_name):
        if self.layer_info[int(current_layer_name) + 1]["shortcut"]:
            return str(int(current_layer_name) + 2)
        else:
            return str(int(current_layer_name) + 1)

    def next_prune_layer(self, current_layer_name):
        index = self.pruning_layer.index(current_layer_name)
        return self.pruning_layer[index + 1]

    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        assert not GIoU or not DIoU, "Just can choice one mode!"
        box1 = box1.t()
        box2 = box2.t()

        # Get the coordinates of bounding boxes
        if x1y1x2y2:
            # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            # x, y, w, h = box1
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                     (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

        iou = inter_area / union_area  # iou
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
            c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
            c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU

        if DIoU:
            box1_center = box1[0:2]
            box2_center = box2[0:2]
            c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
            c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
            inter_diag = torch.pow(box1_center[0] - box2_center[0], 2) + torch.pow(box1_center[1] - box2_center[1], 2)
            out_diag = torch.pow(c_x1 - c_x2, 2) + torch.pow(c_y1 - c_y2, 2)
            return iou - inter_diag / out_diag

        return iou

    @staticmethod
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

    @staticmethod
    def compute_loss_for_aux(output, aux_model, targets):
        ft = torch.cuda.FloatTensor if output[0].is_cuda else torch.Tensor
        lcls, lbox = ft([0]), ft([0])
        if type(aux_model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
            aux_model = aux_model.module  # aux 的超参数是模型内自带的，所以需要脱分布式训练的壳
        hyp = aux_model.hyp
        ft = torch.cuda.FloatTensor if output.is_cuda else torch.Tensor
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([hyp['cls_pw']]), reduction='sum')
        txy, twh, tcls, tbox, index, anchors_vec = AuxNetUtils.build_targets_for_aux(aux_model, targets)
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

        return loss, torch.cat((lbox, lcls, loss)).detach()

    @staticmethod
    def train_for_aux(cfg, weights, batch_size, accumulate, dataloader, hyp, device, resume, epochs, aux_weight):

        model = Darknet(cfg).to(device)
        chkpt = torch.load(weights, map_location=device)
        model.load_state_dict(chkpt['model'], strict=True)
        del chkpt
        aux_util = AuxNetUtils(model, hyp)

        start_epoch = 0
        s = None
        img_size = dataloader.dataset.img_size
        aux_util.creat_aux_list(img_size, device)

        # -----------------optimizer-----------------
        pg0, pg1 = [], []  # optimizer parameter groups
        for net in aux_util.aux_list:
            for k, v in dict(net.named_parameters()).items():
                if 'Conv2d.weight' in k:
                    pg1 += [v]  # parameter group 1 (apply weight_decay)
                else:
                    pg0 += [v]

        optimizer = optim.SGD(pg0, lr=aux_util.hyp['lr0'], momentum=aux_util.hyp['momentum'], nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': aux_util.hyp['weight_decay']})
        del pg0, pg1
        # -----------------optimizer-----------------

        # --------------load the weight--------------
        if resume:
            chkpt = torch.load(aux_weight, map_location=device)

            aux_util.load_aux_weight(chkpt)

            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])

            start_epoch = chkpt['epoch'] + 1
        # --------------load the weight--------------

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
        scheduler.last_epoch = start_epoch - 1

        # --------------hook for forward--------------
        handles = {}  # 结束训练后handle需要回收
        for name, child in model.module_list.named_children():
            if name in aux_util.down_sample_layer:
                handles[name] = child.register_forward_hook(aux_util.hook_forward)
        # --------------hook for forward--------------

        # ----------------DataParallel----------------
        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            aux_util.distributed()
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            model.yolo_layers = model.module.yolo_layers
        # ----------------DataParallel----------------

        # -----------------start train-----------------
        nb = len(dataloader)
        model.nc = 80
        model.hyp = aux_util.hyp
        model.arc = 'default'
        for epoch in range(start_epoch, epochs):
            # model.eval()
            for net in aux_util.aux_list:
                net.train()
            print(('\n' + '%10s' * 8) % ('Stage', 'Epoch', 'gpu_mem', 'Aux', 'DIoU', 'cls', 'total', 'targets'))

            # -----------------start batch-----------------
            pbar = tqdm(enumerate(dataloader), total=nb)
            for i, (imgs, targets, _, _) in pbar:

                if len(targets) == 0:
                    continue

                ni = i + nb * epoch
                imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
                targets = targets.to(device)

                with torch.no_grad():
                    _ = model(imgs)
                aux_util.cat_to_gpu0()
                for aux_idx, feature_maps in enumerate(aux_util.hook_out['gpu0']):
                    pred = aux_util.aux_list[aux_idx](feature_maps)

                    loss, loss_items = AuxNetUtils.compute_loss_for_aux(pred, aux_util.aux_list[aux_idx], targets)
                    loss *= batch_size / 64

                    loss.backward()

                    mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                    s = ('%10s' * 3 + '%10.3g' * 5) % (
                        'Train Aux', '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, aux_idx, *loss_items, len(targets))
                    pbar.set_description(s)

                # 每个batch后要把hook_out内容清除
                aux_util.clean_hook_out()
                if ni % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            # -----------------end batches-----------------

            scheduler.step()
            final_epoch = epoch + 1 == epochs
            chkpt = {'epoch': epoch,
                     'optimizer': None if final_epoch else optimizer.state_dict()}
            for i, net in enumerate(aux_util.aux_list):
                chkpt['aux{}'.format(i)] = net.module.state_dict() if type(
                    net) is nn.parallel.DistributedDataParallel else net.state_dict()
            torch.save(chkpt, aux_weight)

            del chkpt

            with open("aux_result.txt", 'a') as f:
                f.write(s + '\n')

        # 最后要把hook全部删除
        for key, handle in handles.items():
            handle.remove()
        # 把辅助网络列表也都删除
        aux_util.aux_list = []
        torch.cuda.empty_cache()
