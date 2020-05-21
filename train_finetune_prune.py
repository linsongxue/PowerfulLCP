import argparse
import os.path as path
import os
import math
from copy import deepcopy
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test
from models import Darknet
from utils.datasets import LoadImagesAndLabels
from utils.utils import compute_loss, init_seeds
from utils.torch_utils import select_device
from utils.parse_config import parse_data_cfg
from aux_net import AuxNetUtils

mixed_precision = True
try:
    from apex import amp
except:
    mixed_precision = False  # not installed

hyp = {'diou': 3.5,  # giou loss gain
       'giou': 3.5,
       'cls': 37.5,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 49.5,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.006,  # initial learning rate (SGD=1E-3, Adam=9E-5)
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

aux_weight = "../weights/aux.pt"  # weight for aux net
aux_trained = path.exists(aux_weight)
progress_model_all = "../weights/pruning.pt"
progress_result = "./pruning_progress.txt"


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


def fine_tune(cfg, data, pruned_model, aux_util, device, train_loader, test_loader, current_layer_name,
              start_epoch=0, epochs=10, optimizer_state=None, aux_optimizer_state=None):
    batch_size = train_loader.batch_size
    img_size = train_loader.dataset.img_size
    accumulate = 64 // batch_size

    aux_idx = aux_util.conv_layer[current_layer_name]
    hook_layer = aux_util.down_sample_layer[aux_idx]
    aux_loss_scalar = max(0.01, pow((int(hook_layer) + 1) / 75, 2))

    aux_net = aux_util.creat_aux_list(img_size, device, conv_layer_name=current_layer_name)
    chkpt_aux = torch.load(aux_weight, map_location=device)
    aux_net.load_state_dict(chkpt_aux['aux{}'.format(aux_idx)])
    del chkpt_aux

    # ----------init optimizer for pruned model----------
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(pruned_model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0
    pruned_optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    pruned_optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1
    # ----------init optimizer for pruned model----------

    # -------------init optimizer for aux net-------------
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(aux_net.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0
    aux_optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    aux_optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1
    # -------------init optimizer for aux net-------------

    if optimizer_state is not None and aux_optimizer_state is not None:
        pruned_optimizer.load_state_dict(optimizer_state)
        aux_optimizer.load_state_dict(aux_optimizer_state)

    p_scheduler = lr_scheduler.MultiStepLR(pruned_optimizer, milestones=[epochs // 2], gamma=0.1)
    a_scheduler = lr_scheduler.MultiStepLR(aux_optimizer, milestones=[epochs // 2], gamma=0.1)
    p_scheduler.last_epoch = start_epoch - 1
    a_scheduler.last_epoch = start_epoch - 1

    # ----------------apex and distributed----------------
    if mixed_precision:
        pruned_model, pruned_optimizer = amp.initialize(pruned_model, pruned_optimizer, opt_level='O1', verbosity=0)
        aux_net, aux_optimizer = amp.initialize(aux_net, aux_optimizer, opt_level='O1', verbosity=0)
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        pruned_model = torch.nn.parallel.DistributedDataParallel(pruned_model, find_unused_parameters=True)
        pruned_model.yolo_layers = pruned_model.module.yolo_layers
        aux_net = torch.nn.parallel.DistributedDataParallel(aux_net, find_unused_parameters=True)
        # aux_net.head = aux_net.module.head
    # ----------------apex and distributed----------------

    # -------------start train-------------
    nb = len(train_loader)
    pruned_model.nc = 80
    pruned_model.hyp = hyp
    pruned_model.arc = 'default'
    for epoch in range(start_epoch, epochs):

        # -------------register hook for model-------------
        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            for name, child in pruned_model.module.module_list.named_children():
                if name == hook_layer:
                    handle = child.register_forward_hook(aux_util.hook_forward)
        else:
            for name, child in pruned_model.module_list.named_children():
                if name == hook_layer:
                    handle = child.register_forward_hook(aux_util.hook_forward)
        # -------------register hook for model-------------

        pruned_model.train()
        aux_net.train()

        print(('\n' + '%10s' * 8) % ('Stage', 'Epoch', 'gpu_mem', 'DIoU', 'obj', 'cls', 'total', 'targets'))

        # -------------start batch-------------
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

            aux_util.cat_to_gpu0()
            aux_pred = aux_net(aux_util.hook_out['gpu0'][0])
            aux_loss, aux_loss_items = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)
            aux_loss *= aux_loss_scalar * batch_size / 64

            if mixed_precision:
                with amp.scale_loss(pruned_loss, pruned_optimizer) as pruned_scaled_loss:
                    pruned_scaled_loss.backward()
                with amp.scale_loss(aux_loss, aux_optimizer) as aux_scaled_loss:
                    aux_scaled_loss.backward()
            else:
                pruned_loss.backward()
                aux_loss.backward()

            aux_util.clean_hook_out()
            if ni % accumulate == 0:
                pruned_optimizer.step()
                aux_optimizer.step()
                pruned_optimizer.zero_grad()
                aux_optimizer.zero_grad()

            pruned_loss_items[0] += aux_loss_items[0]
            pruned_loss_items[2] += aux_loss_items[1]
            pruned_loss_items[3] += aux_loss_items[2]
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 3 + '%10.3g' * 5) % (
                'Fine tune', '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *pruned_loss_items, len(targets))
            pbar.set_description(s)
        # -------------end batch-------------

        p_scheduler.step()
        a_scheduler.step()
        handle.remove()

        results, _ = test.test(cfg,
                               data,
                               batch_size=batch_size * 2,
                               img_size=416,
                               model=pruned_model,
                               conf_thres=0.1,
                               iou_thres=0.5,
                               save_json=False,
                               dataloader=test_loader)

        with open(progress_result, 'a') as f:
            f.write(('%10s' * 3 + '%10.3g' * 7) % ('Fine tune', '%g/%g' % (epoch, epochs - 1), results) + '\n')

        chkpt = {'current_layer': current_layer_name,
                 'epoch': epoch,
                 'model': pruned_model.module if type(
                     pruned_model) is nn.parallel.DistributedDataParallel else pruned_model,
                 'optimizer': None if epoch == epochs - 1 else pruned_optimizer.state_dict(),
                 'aux_optimizer': None if epoch == epochs - 1 else aux_optimizer.state_dict()}
        torch.save(chkpt, progress_model_all)
        del chkpt

        chkpt_aux = torch.load(aux_weight, map_location=device)
        chkpt_aux['aux{}'.format(aux_idx)] = aux_net.module.state_dict() if type(
                    aux_net) is nn.parallel.DistributedDataParallel else aux_net.state_dict()
        torch.save(chkpt_aux, aux_weight)
        del chkpt_aux
    # -------------end train-------------
    torch.cuda.empty_cache()
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None


def channels_select(cfg, data, origin_model, pruning_model, aux_util, device, data_loader, select_layer,
                    pruned_rate):
    batch_size = data_loader.batch_size
    img_size = data_loader.dataset.img_size
    accumulate = 64 // batch_size

    # ----------get prune layer list----------
    if select_layer in list(aux_util.short_cut_info.keys()):
        select_layer_list = [select_layer]
        select_layer_list += aux_util.short_cut_info[select_layer]
    else:
        select_layer_list = [select_layer]

    feature_num = len(select_layer_list)
    # ----------get prune layer list----------

    # ----------prepare to get origin feature maps----------
    origin_features = []
    pruned_features = []
    handles = []

    def hook_origin(module, input, output):
        origin_features.append(output)

    def hook_pruned(module, input, output):
        pruned_features.append(output)

    for name, child in origin_model.module_list.named_children():
        if name in select_layer_list:
            handles.append(child.register_forward_hook(hook_origin))
    # ----------prepare to get origin feature maps----------

    # ----------prepare to get pruned feature maps----------
    aux_idx = aux_util.conv_layer[select_layer_list[0]]
    hook_layer_aux = aux_util.down_sample_layer[aux_idx]
    for name, child in pruning_model.module_list.named_children():
        if name in select_layer_list:
            handles.append(child.register_forward_hook(hook_pruned))
        elif name == hook_layer_aux:
            handles.append(child.register_forward_hook(aux_util.hook_forward))

    aux_net = aux_util.creat_aux_list(img_size, device, conv_layer_name=select_layer_list)
    chkpt_aux = torch.load(aux_weight, map_location=device)
    aux_net.load_state_dict(chkpt_aux['aux{}'.format(aux_idx)])
    del chkpt_aux
    # ----------prepare to get pruned feature maps----------

    nb = len(data_loader)
    pruning_model.nc = 80
    pruning_model.hyp = hyp
    pruning_model.arc = 'default'
    pruning_model.eval()
    aux_net.eval()
    MSE = nn.MSELoss(reduction='mean')
    pbar = tqdm(data_loader, total=nb)
    for imgs, targets, _, _ in pbar:

        if len(targets) == 0:
            continue

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        with torch.no_grad():
            _ = origin_model(imgs)

        pruning_pred = pruning_model(imgs)
        pruning_loss, _ = compute_loss(pruning_pred, targets, pruning_model)

        aux_pred = aux_net(aux_util.hook_out['gpu0'][0])
        aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

        mse_loss = 0.0
        for i in range(feature_num):
            mse_loss += (MSE(pruned_features[i], origin_features[i]) / feature_num)

        loss = mse_loss + hyp['joint_loss'] * (pruning_loss + aux_loss)

        loss.backward()

        mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
        s = ('%10s' * 2 + '%10.3g' * 5) % (
            'Pruning', '%.3gG' % mem, mse_loss, pruning_loss, aux_loss, loss, len(targets))
        pbar.set_description(s)

        aux_util.clean_hook_out()
        pruned_features = []
        origin_features = []

    aux_util.clean_hook_out()
    pruned_features = []
    origin_features = []

    out_channels = pruning_model.module_list[int(select_layer_list[0])].Conv2d.out_channels
    k = math.ceil(out_channels * pruned_rate)
    pg0, pg1 = [], []
    for layer in select_layer_list:
        grad = pruning_model.module_list[int(layer)].Conv2d.weight.grad.detach()
        grad = torch.sum(grad.square(), dim=(1, 2, 3))
        _, indices = torch.topk(grad, k)
        indices, _ = indices.sort()
        new_module = nn.Sequential()
        new_module.add_module('Conv2d', nn.Conv2d(in_channels=pruning_model.module_list[int(layer)].Conv2d.in_channels,
                                                  out_channels=k,
                                                  kernel_size=pruning_model.module_list[int(layer)].Conv2d.kernel_size,
                                                  stride=pruning_model.module_list[int(layer)].Conv2d.stride,
                                                  padding=pruning_model.module_list[int(layer)].Conv2d.padding,
                                                  bias=pruning_model.module_list[int(layer)].Conv2d.bias))
        new_module.add_module('BatchNorm2d', nn.BatchNorm2d(k, momentum=0.1))
        new_module.add_module('activation', pruning_model.module_list[int(layer)].activation)
        new_module.Conv2d.weight = torch.index_select(pruning_model.module_list[int(layer)].Conv2d.weight, 0, indices)
        new_module.BatchNorm2d.weight = torch.index_select(pruning_model.module_list[int(layer)].BatchNorm2d.weight, 0,
                                                           indices)
        new_module.BatchNorm2d.bias = torch.index_select(pruning_model.module_list[int(layer)].BatchNorm2d.bias, 0,
                                                         indices)
        pg1 += [new_module.Conv2d.weight]
        pg0 += [new_module.BatchNorm2d.weight, new_module.BatchNorm2d.bias]
        pruning_model.module_list[int(layer)] = new_module
        pruning_model.module_defs[int(layer)]['filters'] = k

        next_layer = aux_util.next_conv_layer(layer)
        if pruning_model.module_list[int(next_layer)].Conv2d.in_channels != k:
            new_conv2d = nn.Conv2d(in_channels=k,
                                   out_channels=pruning_model.module_list[int(next_layer)].Conv2d.out_channels,
                                   kernel_size=pruning_model.module_list[int(next_layer)].Conv2d.kernel_size,
                                   stride=pruning_model.module_list[int(next_layer)].Conv2d.stride,
                                   padding=pruning_model.module_list[int(next_layer)].Conv2d.padding,
                                   bias=pruning_model.module_list[int(next_layer)].Conv2d.bias)
            new_conv2d.weight = torch.index_select(pruning_model.module_list[int(next_layer)].Conv2d.weight, 1, indices)
            pruning_model.module_list[int(next_layer)].Conv2d = new_conv2d

    pruning_model.zero_grad()

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    del pg1, pg0

    # ----------update weight----------
    pbar = tqdm(enumerate(data_loader), total=nb)
    for i, (imgs, targets, _, _) in pbar:

        if len(targets) == 0:
            continue

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        with torch.no_grad():
            _ = origin_model(imgs)

        pruning_pred = pruning_model(imgs)
        pruning_loss, _ = compute_loss(pruning_pred, targets, pruning_model)

        aux_pred = aux_net(aux_util.hook_out['gpu0'][0])
        aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

        mse_loss = 0.0
        for i in range(feature_num):
            mse_loss += (MSE(pruned_features[i], origin_features[i]) / feature_num)

        loss = mse_loss + hyp['joint_loss'] * (pruning_loss + aux_loss)

        loss.backward()

        if i % accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()

        mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
        s = ('%10s' * 2 + '%10.3g' * 5) % (
            'Pruning', '%.3gG' % mem, mse_loss, pruning_loss, aux_loss, loss, len(targets))
        pbar.set_description(s)

        aux_util.clean_hook_out()
        pruned_features = []
        origin_features = []

    aux_util.clean_hook_out()
    pruned_features = []
    origin_features = []

    for handle in handles:
        handle.remove()
    # ----------update weight----------

    chkpt = {'current_layer': aux_util.next_prune_layer(select_layer_list[0]),
             'epoch': 0,
             'model': pruning_model,
             'optimizer': None,
             'aux_optimizer': None}
    torch.save(chkpt, progress_model_all)
    del chkpt

    results, _ = test.test(cfg,
                           data,
                           batch_size=batch_size * 2,
                           img_size=416,
                           model=pruning_model,
                           conf_thres=0.1,
                           iou_thres=0.5,
                           save_json=False,
                           dataloader=None)

    with open(progress_result, 'a') as f:
        f.write(('%10s' + '%10.3g' * 7) % ('Pruning', results) + '\n')

    torch.cuda.empty_cache()


def get_thin_model(cfg, data, weights, img_size, batch_size, accumulate, prune_rate, aux_epochs=10, ft_epochs=5,
                   resume=False,
                   cache_images=False):
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

    channels_select_loader = torch.utils.data.DataLoader(LoadImagesAndLabels(train_path, img_size, batch_size // 2,
                                                                             augment=True,
                                                                             hyp=hyp,
                                                                             rect=False,
                                                                             cache_labels=True,
                                                                             cache_images=cache_images),
                                                         batch_size=batch_size // 2,
                                                         num_workers=nw,
                                                         shuffle=True,
                                                         pin_memory=True,
                                                         collate_fn=dataset.collate_fn)
    # -----------------dataset-----------------

    # -----------get trained aux net-----------
    if aux_trained:
        chkpt = torch.load(aux_weight)
        if chkpt["epoch"] + 1 != aux_epochs:
            del chkpt
            AuxNetUtils.train_for_aux(cfg, weights, batch_size, accumulate, train_loader, hyp, device, resume=True,
                                      epochs=aux_epochs, aux_weight=aux_weight)
        else:
            del chkpt
    else:
        AuxNetUtils.train_for_aux(cfg, weights, batch_size, accumulate, train_loader, hyp, device, resume=False,
                                  epochs=aux_epochs, aux_weight=aux_weight)
    # -----------get trained aux net-----------

    # ----------init model and aux net----------
    origin_model = Darknet(cfg).to(device)
    chkpt = torch.load(weights, map_location=device)
    origin_model.load_state_dict(chkpt['model'], strict=True)
    aux_util = AuxNetUtils(origin_model, hyp)

    first_prune = False
    if resume:
        progress_chkpt = torch.load(progress_model_all, map_location=device)
        pruning_model = progress_chkpt['model']
        start_epoch = progress_chkpt['epoch']
        current_layer = progress_chkpt['current_layer']
        optimizer_state = progress_chkpt['optimizer']
        aux_optimizer_state = progress_chkpt['aux_optimizer']
        if start_epoch == ft_epochs - 1:
            first_prune = True
        del progress_chkpt
    else:
        pruning_model = deepcopy(origin_model)
        start_epoch = 0
        current_layer = aux_util.pruning_layer[0]
        optimizer_state = None
        aux_optimizer_state = None
    # ----------init model and aux net----------

    if first_prune:
        channels_select(cfg, data, origin_model, pruning_model, aux_util, device, channels_select_loader, current_layer,
                        prune_rate)
        current_layer = aux_util.next_prune_layer(current_layer)
        start_epoch = 0

    current_layer_index = aux_util.pruning_layer.index(current_layer)
    for layer in aux_util.pruning_layer[current_layer_index:]:
        fine_tune(cfg, data, pruning_model, aux_util, device, train_loader, test_loader, current_layer, start_epoch,
                  ft_epochs, optimizer_state, aux_optimizer_state)
        channels_select(cfg, data, origin_model, pruning_model, aux_util, device, channels_select_loader, layer,
                        prune_rate)

    write_cfg('thin_yolo.cfg', pruning_model.module_defs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune-rate', type=float, default=0.7)
    parser.add_argument('--aux-epochs', type=int, default=10)
    parser.add_argument('--ft-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='../weights/converted.pt', help='initial weights')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--resume', action='store_true', help='resume training from pruning.pt')
    opt = parser.parse_args()
    print(opt)
    device = select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

    get_thin_model(cfg=opt.cfg,
                   data=opt.data,
                   weights=opt.weights,
                   img_size=opt.img_size,
                   batch_size=opt.batch_size,
                   accumulate=opt.accumulate,
                   prune_rate=opt.prune_rate,
                   aux_epochs=opt.aux_epochs,
                   ft_epochs=opt.ft_epochs,
                   resume=opt.resume,
                   cache_images=opt.cache_images)
