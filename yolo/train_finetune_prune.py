import argparse
import os.path as path
import os
import math
from tqdm import tqdm
import numpy as np
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import test
from models import Darknet
from model_unit import MaskConv2d
from utils.datasets import LoadImagesAndLabels
from utils.utils import compute_loss, init_seeds
from utils.torch_utils import select_device
from utils.parse_config import parse_data_cfg, parse_model_cfg
from aux_net import AuxNetUtils, HookUtils, mask_converted
import logging

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

aux_weight = "../weights/aux.pt"  # weight for aux net
aux_trained = path.exists(aux_weight)
progress_chkpt = "../weights/pruning.pt"
progress_result = "./pruning_progress.txt"
last = "../weights/last.pt"
loss_analyse = 'loss_analyse.txt'
logging.basicConfig(filename=loss_analyse, filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


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


def fine_tune(cfg, data, prune_cfg, aux_util, device, train_loader, test_loader, epochs=10):
    with open(progress_result, 'a') as f:
        f.write(('\n' + '%10s' * 10 + '\n') %
                ('Stage', 'Epoch', 'DIoU', 'obj', 'cls', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))

    batch_size = train_loader.batch_size
    img_size = train_loader.dataset.img_size
    accumulate = 64 // batch_size
    need_aux_net = True

    # ----------init pruned model----------
    pruned_model = Darknet(prune_cfg, img_size=(img_size, img_size)).to(device)
    # ----------init pruned model----------

    chkpt = torch.load(progress_chkpt, map_location=device)
    pruned_model.load_state_dict(chkpt['model'], strict=True)
    current_layer_name = chkpt['current_layer']
    start_epoch = chkpt['epoch'] + 1
    if start_epoch == epochs or current_layer_name == 'end':
        return current_layer_name  # fine tune 完毕，返回需要修剪的层名

    optimizer_state = chkpt['optimizer']
    aux_optimizer_state = chkpt['aux_optimizer']
    del chkpt

    # ----------init optimizer for pruned model----------
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(pruned_model.named_parameters()).items():
        if 'MaskConv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0
    pruned_optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    pruned_optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    if optimizer_state is not None:
        pruned_optimizer.load_state_dict(optimizer_state)

    p_scheduler = lr_scheduler.MultiStepLR(pruned_optimizer, milestones=[epochs // 3, 2 * (epochs // 3)], gamma=0.1)
    p_scheduler.last_epoch = start_epoch - 1
    # ----------init optimizer for pruned model----------

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        pruned_model = torch.nn.parallel.DistributedDataParallel(pruned_model, find_unused_parameters=True)
        pruned_model.yolo_layers = pruned_model.module.yolo_layers

    aux_idx = aux_util.conv_layer_dict[current_layer_name]
    if aux_idx == -1:
        need_aux_net = False
    else:
        hook_util = HookUtils()

        hook_layer = aux_util.down_sample_layer[aux_idx]
        aux_loss_scalar = max(0.01, pow((int(hook_layer) + 1) / 75, 2))
        aux_net = aux_util.creat_aux_list(img_size, device, conv_layer_name=current_layer_name)
        chkpt_aux = torch.load(aux_weight, map_location=device)
        aux_net.load_state_dict(chkpt_aux['aux{}'.format(aux_idx)])
        del chkpt_aux

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

        if aux_optimizer_state is not None:
            aux_optimizer.load_state_dict(aux_optimizer_state)

        a_scheduler = lr_scheduler.MultiStepLR(aux_optimizer, milestones=[epochs // 3, 2 * (epochs // 3)], gamma=0.1)
        a_scheduler.last_epoch = start_epoch - 1
        # -------------init optimizer for aux net-------------
        aux_net = torch.nn.parallel.DistributedDataParallel(aux_net, find_unused_parameters=True)

    # -------------start train-------------
    nb = len(train_loader)
    pruned_model.nc = 80
    pruned_model.hyp = hyp
    pruned_model.arc = 'default'
    for epoch in range(start_epoch, epochs):

        # -------------register hook for model-------------
        if need_aux_net:
            for name, child in pruned_model.module.module_list.named_children():
                if name == hook_layer:
                    handle = child.register_forward_hook(hook_util.hook_prune_input)
        # -------------register hook for model-------------

        pruned_model.train()
        if need_aux_net:
            aux_net.train()

        print(('\n' + '%10s' * 8) % ('Stage', 'Epoch', 'gpu_mem', 'DIoU', 'obj', 'cls', 'total', 'targets'))

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

            if need_aux_net:
                hook_util.cat_to_gpu0(model='prune')
                aux_pred = aux_net(hook_util.pruned_features['gpu0'][0])
                aux_loss, aux_loss_items = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)
                aux_loss *= aux_loss_scalar * batch_size / 64
            else:
                aux_loss = torch.zeros(1, dtype=pruned_loss.dtype, device=pruned_loss.device)
                aux_loss_items = torch.zeros(3, dtype=pruned_loss.dtype, device=pruned_loss.device)

            loss = pruned_loss + aux_loss
            loss.backward()

            if need_aux_net:
                hook_util.clean_hook_out(model='prune')
            if ni % accumulate == 0:
                pruned_optimizer.step()
                pruned_optimizer.zero_grad()
                if need_aux_net:
                    aux_optimizer.step()
                    aux_optimizer.zero_grad()

            pruned_loss_items[0] += aux_loss_items[0]
            pruned_loss_items[2] += aux_loss_items[1]
            pruned_loss_items[3] += aux_loss_items[2]
            pruned_loss_items /= 2
            mloss = (mloss * i + pruned_loss_items) / (i + 1)
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 3 + '%10.3g' * 5) % (
                'FiTune ' + current_layer_name, '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, len(targets))
            pbar.set_description(s)
        # -------------end batch-------------

        p_scheduler.step()
        if need_aux_net:
            handle.remove()
            a_scheduler.step()

        results, _ = test.test(cfg,
                               data,
                               batch_size=batch_size * 2,
                               img_size=416,
                               model=pruned_model,
                               conf_thres=0.1,
                               iou_thres=0.5,
                               save_json=False,
                               dataloader=test_loader)

        chkpt = {'current_layer': current_layer_name,
                 'epoch': epoch,
                 'model': pruned_model.module.state_dict() if type(
                     pruned_model) is nn.parallel.DistributedDataParallel else pruned_model.state_dict(),
                 'optimizer': None if epoch == epochs - 1 else pruned_optimizer.state_dict(),
                 'aux_optimizer': None if epoch == epochs - 1 or not need_aux_net else aux_optimizer.state_dict()}

        torch.save(chkpt, progress_chkpt)

        torch.save(chkpt, last)

        if epoch == epochs - 1:
            torch.save(chkpt, '../weights/backup{}.pt'.format(current_layer_name))

        del chkpt

        if need_aux_net:
            chkpt_aux = torch.load(aux_weight, map_location=device)
            chkpt_aux['aux{}'.format(aux_idx)] = aux_net.module.state_dict() if type(
                aux_net) is nn.parallel.DistributedDataParallel else aux_net.state_dict()
            torch.save(chkpt_aux, aux_weight)
            del chkpt_aux

        with open(progress_result, 'a') as f:
            f.write(('%10s' * 2 + '%10.3g' * 8) % (
                'FiTune ' + current_layer_name, '%g/%g' % (epoch, epochs - 1), *mloss, *results[:4]) + '\n')
    # -------------end train-------------
    torch.cuda.empty_cache()
    return current_layer_name


def channels_select(cfg, data, origin_model, prune_cfg, aux_util, device, data_loader, select_layer, pruned_rate):
    with open(progress_result, 'a') as f:
        f.write(('\n' + '%10s' * 10 + '\n') %
                ('Stage', 'Change', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'P', 'R', 'mAP@0.5', 'F1'))
    logger.info(('%10s' * 7) % ('Stage', 'Layer', 'Batch', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total'))

    batch_size = data_loader.batch_size
    img_size = data_loader.dataset.img_size
    accumulate = 64 // batch_size
    need_aux_net = True
    hook_util = HookUtils()
    handles = []

    pruning_model = Darknet(prune_cfg, img_size=(img_size, img_size)).to(device)
    chkpt = torch.load(progress_chkpt, map_location=device)
    pruning_model.load_state_dict(chkpt['model'], strict=True)
    del chkpt

    solve_sub_problem_optimizer = optim.SGD(pruning_model.module_list[int(select_layer)].MaskConv2d.parameters(),
                                            lr=hyp['lr0'],
                                            momentum=hyp['momentum'])

    # ----------prepare to get origin feature maps----------
    for name, child in origin_model.module_list.named_children():
        if name == select_layer:
            handles.append(child.BatchNorm2d.register_forward_hook(hook_util.hook_origin_input))
    # ----------prepare to get origin feature maps----------

    # ----------prepare to get pruned feature maps----------
    aux_idx = aux_util.conv_layer_dict[select_layer]
    hook_layer_aux = aux_util.down_sample_layer[aux_idx]
    if aux_idx == -1:
        need_aux_net = False
    for name, child in pruning_model.module_list.named_children():
        if name == select_layer:
            handles.append(child.BatchNorm2d.register_forward_hook(hook_util.hook_prune_input))
        elif name == hook_layer_aux and need_aux_net:
            handles.append(child.register_forward_hook(hook_util.hook_prune_input))

    if need_aux_net:
        aux_net = aux_util.creat_aux_list(img_size, device, conv_layer_name=select_layer)
        chkpt_aux = torch.load(aux_weight, map_location=device)
        aux_net.load_state_dict(chkpt_aux['aux{}'.format(aux_idx)])
        del chkpt_aux
    # ----------prepare to get pruned feature maps----------

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        origin_model = torch.nn.parallel.DistributedDataParallel(origin_model, find_unused_parameters=True)
        origin_model.yolo_layers = origin_model.module.yolo_layers
        pruning_model = torch.nn.parallel.DistributedDataParallel(pruning_model, find_unused_parameters=True)
        pruning_model.yolo_layers = pruning_model.module.yolo_layers
        if need_aux_net:
            aux_net = torch.nn.parallel.DistributedDataParallel(aux_net, find_unused_parameters=True)

    nb = len(data_loader)
    pruning_model.nc = 80
    pruning_model.hyp = hyp
    pruning_model.arc = 'default'
    pruning_model.eval()
    if need_aux_net:
        aux_net.eval()
    MSE = nn.MSELoss(reduction='mean')
    pbar = tqdm(enumerate(data_loader), total=nb)
    print(('\n' + '%10s' * 7) % ('Stage', 'gpu_mem', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'targets'))
    for i, (imgs, targets, _, _) in pbar:

        if len(targets) == 0:
            continue

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        with torch.no_grad():
            _ = origin_model(imgs)
        hook_util.cat_to_gpu0('origin')

        _, pruning_pred = pruning_model(imgs)
        pruning_loss, _ = compute_loss(pruning_pred, targets, pruning_model)
        hook_util.cat_to_gpu0('prune')

        if need_aux_net:
            aux_pred = aux_net(hook_util.pruned_features['gpu0'][1])
            aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

        mse_loss = torch.zeros(1).to(device)
        mse_loss += MSE(hook_util.pruned_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

        if need_aux_net:
            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss
        else:
            aux_loss = torch.zeros(1)
            loss = hyp['joint_loss'] * mse_loss + pruning_loss

        loss.backward()

        mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
        s = ('%10s' * 2 + '%10.3g' * 5) % (
            'Pruning ' + select_layer, '%.3gG' % mem, mse_loss, pruning_loss, aux_loss, loss, len(targets))
        pbar.set_description(s)

        if i % 100 == 0:
            logger.info(('%10s' * 3 + '%10.3g' * 4) %
                        ('Pruning', select_layer, '%g/%g' % (i, nb - 1), mse_loss, pruning_loss, aux_loss, loss))

        hook_util.clean_hook_out('origin')
        hook_util.clean_hook_out('prune')

    info = aux_util.layer_info[int(select_layer)]
    in_channels = info['in_channels']
    k = math.floor(in_channels * pruned_rate)

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        grad = pruning_model.module.module_list[int(select_layer)].MaskConv2d.weight.grad.detach() ** 2
    else:
        grad = pruning_model.module_list[int(select_layer)].MaskConv2d.weight.grad.detach() ** 2
    grad = grad.sum((2, 3)).sqrt().sum(0)
    _, indices = torch.topk(grad, k, largest=False)
    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        pruning_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[indices] = 0
    else:
        pruning_model.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[indices] = 0

    pruning_model.zero_grad()

    # ----------update weight to solve sub problem----------
    mloss = torch.zeros(4).to(device)
    print(('\n' + '%10s' * 7) % ('Stage', 'gpu_mem', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'targets'))
    for i, (imgs, targets, _, _) in pbar:

        if len(targets) == 0:
            continue

        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)

        with torch.no_grad():
            _ = origin_model(imgs)
        hook_util.cat_to_gpu0('origin')

        _, pruning_pred = pruning_model(imgs)
        pruning_loss, _ = compute_loss(pruning_pred, targets, pruning_model)
        hook_util.cat_to_gpu0('prune')

        if need_aux_net:
            aux_pred = aux_net(hook_util.pruned_features['gpu0'][1])
            aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

        mse_loss = torch.zeros(1).to(device)
        mse_loss += MSE(hook_util.pruned_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

        if need_aux_net:
            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss
        else:
            aux_loss = torch.zeros(1)
            loss = hyp['joint_loss'] * mse_loss + pruning_loss

        loss.backward()

        if i % accumulate == 0:
            solve_sub_problem_optimizer.step()
            solve_sub_problem_optimizer.zero_grad()

        mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
        mloss = (mloss * i + torch.cat([mse_loss, pruning_loss, aux_loss, loss]).detach()) / (i + 1)
        s = ('%10s' * 2 + '%10.3g' * 5) % (
            'SubProm ' + select_layer, '%.3gG' % mem, *mloss, len(targets))
        pbar.set_description(s)

        if i % 100 == 0:
            logger.info(('%10s' * 2 + '%10.3g' * 4) %
                        ('SubProm', select_layer, '%g/%g' % (i, nb - 1), mse_loss, pruning_loss, aux_loss, loss))

        hook_util.clean_hook_out('origin')
        hook_util.clean_hook_out('prune')

    for handle in handles:
        handle.remove()
    # ----------update weight to solve sub problem----------

    if select_layer == aux_util.pruning_layer[-1]:
        current_layer = 'end'
    else:
        current_layer = aux_util.next_prune_layer(select_layer)
    chkpt = {'current_layer': current_layer,
             'epoch': -1,
             'model': pruning_model.module.state_dict() if type(
                 pruning_model) is nn.parallel.DistributedDataParallel else pruning_model.state_dict(),
             'optimizer': None,
             'aux_optimizer': None}

    torch.save(chkpt, progress_chkpt)

    torch.save(chkpt, last)
    del chkpt

    res, _ = test.test(cfg,
                       data,
                       batch_size=batch_size * 2,
                       img_size=416,
                       model=pruning_model,
                       conf_thres=0.1,
                       iou_thres=0.5,
                       save_json=False,
                       dataloader=None)

    with open(progress_result, 'a') as f:
        f.write(('%10s' * 2 + '%10.3g' * 8) %
                ('Pruning ' + select_layer, str(in_channels) + '->' + str(in_channels - k), *mloss, *res[:4]) + '\n')

    torch.cuda.empty_cache()


def get_thin_model(cfg, data, origin_weights, img_size, batch_size, accumulate, prune_rate, aux_epochs=50, ft_epochs=15,
                   resume=False, cache_images=False, start_layer='1'):
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

    channels_select_loader = torch.utils.data.DataLoader(LoadImagesAndLabels(train_path, img_size, batch_size,
                                                                             augment=False,
                                                                             hyp=hyp,
                                                                             rect=False,
                                                                             cache_labels=True,
                                                                             cache_images=cache_images),
                                                         batch_size=batch_size,
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
            AuxNetUtils.train_for_aux(cfg, origin_weights, batch_size, accumulate, train_loader, hyp, device,
                                      resume=True,
                                      epochs=aux_epochs, aux_weight=aux_weight)
        else:
            del chkpt
    else:
        AuxNetUtils.train_for_aux(cfg, origin_weights, batch_size, accumulate, train_loader, hyp, device, resume=False,
                                  epochs=aux_epochs, aux_weight=aux_weight)
    # -----------get trained aux net-----------

    # -----------------get mask cfg-----------------
    mask_cfg = '/'.join(cfg.split('/')[:-1]) + '/mask' + cfg.split('/')[-1]
    if not path.exists(mask_cfg):
        origin_mdfs = parse_model_cfg(cfg)
        mask_mdfs = []
        last_darknet = 75
        for i, mdf in enumerate(origin_mdfs):
            if i <= last_darknet and mdf['type'] == 'convolutional':
                mdf['type'] = 'maskconvolutional'
            mask_mdfs.append(mdf)
        write_cfg(mask_cfg, mask_mdfs)
    # -----------------get mask cfg-----------------

    # ----------init model and aux util----------
    origin_model = Darknet(cfg).to(device)
    chkpt = torch.load(origin_weights, map_location=device)
    origin_model.load_state_dict(chkpt['model'], strict=True)
    aux_util = AuxNetUtils(origin_model, hyp)
    del chkpt
    # ----------init model and aux net----------

    # ----------start from first layer----------
    if not resume:
        init_state_dict = mask_converted(mask_cfg, origin_weights, target=None)

        first_progress = {'current_layer': start_layer,
                          'epoch': -1,
                          'model': init_state_dict,
                          'optimizer': None,
                          'aux_optimizer': None}
        torch.save(first_progress, progress_chkpt)

        with open(progress_result, 'a') as f:
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write('\n' + t + '\n')
        # ----------start from first layer----------

    layer = start_layer
    if start_layer == 'end':
        return mask_cfg, aux_util

    while int(layer) <= int(aux_util.pruning_layer[-1]):
        layer = fine_tune(cfg, data, mask_cfg, aux_util, device, train_loader, test_loader, ft_epochs)
        if layer == 'end':
            break
        channels_select(cfg, data, origin_model, mask_cfg, aux_util, device, channels_select_loader, layer, prune_rate)

    return mask_cfg, aux_util


def pruning(mask_cfg, progress_weights, aux_util):
    device_in = torch.device('cpu')
    model = Darknet(mask_cfg)
    chkpt = torch.load(progress_weights, map_location=device_in)
    model.load_state_dict(chkpt['model'])

    new_cfg = parse_model_cfg(mask_cfg)

    route_layer = ['37', '62']
    route_mask = [None, None]
    for layer in aux_util.pruning_layer[::-1]:
        assert isinstance(model.module_list[int(layer)][0], MaskConv2d), "Not a pruned model!"
        in_channels_mask = model.module_list[int(layer)][0].selected_channels_mask
        out_channels_mask = torch.ones(model.module_list[int(layer)][0].out_channels)
        if layer != aux_util.pruning_layer[-1]:
            out_channels_mask = model.module_list[int(aux_util.next_prune_layer(layer))][0].selected_channels_mask

        in_channels = int(torch.sum(in_channels_mask))
        out_channels = int(torch.sum(out_channels_mask))

        new_cfg[int(layer) + 1]['type'] = 'convolutional'
        new_cfg[int(layer) + 1]['filters'] = str(out_channels)

        new_conv2d = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=model.module_list[int(layer)][0].kernel_size,
                               stride=model.module_list[int(layer)][0].stride,
                               padding=model.module_list[int(layer)][0].padding,
                               bias=False)
        thin_weight = model.module_list[int(layer)][0].weight[out_channels_mask.bool()]
        thin_weight = thin_weight[:, in_channels_mask.bool()]
        new_conv2d.weight.data.copy_(thin_weight.data)

        new_batch = nn.BatchNorm2d(out_channels, momentum=0.1)
        thin_weight = model.module_list[int(layer)][1].weight[out_channels_mask.bool()]
        thin_bias = model.module_list[int(layer)][1].bias[out_channels_mask.bool()]
        thin_mean = model.module_list[int(layer)][1].running_mean[out_channels_mask.bool()]
        thin_var = model.module_list[int(layer)][1].running_var[out_channels_mask.bool()]

        new_batch.weight.data.copy_(thin_weight)
        new_batch.bias.data.copy_(thin_bias)
        new_batch.running_mean.copy_(thin_mean)
        new_batch.running_var.copy_(thin_var)
        new_module = nn.Sequential()
        new_module.add_module('Conv2d', new_conv2d)
        new_module.add_module('BatchNorm2d', new_batch)
        new_module.add_module('activation', model.module_list[int(layer)][2])
        model.module_list[int(layer)] = new_module

        if layer == route_layer[0]:
            mask = torch.cat([torch.ones(model.module_list[96][0].out_channels), in_channels_mask], 0)
            route_mask[0] = mask
        elif layer == route_layer[1]:
            mask = torch.cat([torch.ones(model.module_list[84][0].out_channels), in_channels_mask], 0)
            route_mask[1] = mask

    new_conv2d_87 = nn.Conv2d(in_channels=int(torch.sum(route_mask[1])),
                              out_channels=model.module_list[87][0].out_channels,
                              kernel_size=model.module_list[87][0].kernel_size,
                              stride=model.module_list[87][0].stride,
                              padding=model.module_list[87][0].padding,
                              bias=False)
    new_conv2d_87.weight.data.copy_(model.module_list[87][0].weight.data[:, route_mask[1].bool()])

    new_conv2d_99 = nn.Conv2d(in_channels=int(torch.sum(route_mask[0])),
                              out_channels=model.module_list[99][0].out_channels,
                              kernel_size=model.module_list[99][0].kernel_size,
                              stride=model.module_list[99][0].stride,
                              padding=model.module_list[99][0].padding,
                              bias=False)
    new_conv2d_99.weight.data.copy_(model.module_list[99][0].weight.data[:, route_mask[0].bool()])

    model.module_list[87][0] = new_conv2d_87
    model.module_list[99][0] = new_conv2d_99

    write_cfg('cfg/pruned-yolov3.cfg', new_cfg)
    chkpt = {'epoch': -1,
             'best_fitness': None,
             'training_results': None,
             'model': model.state_dict(),
             'optimizer': None}
    torch.save(chkpt, '../weights/pruned-converted.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune-rate', type=float, default=0.7)
    parser.add_argument('--start_layer', type=str, default='1')
    parser.add_argument('--aux-epochs', type=int, default=50)
    parser.add_argument('--ft-epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=1, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-voc.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='../weights/converted-voc.pt', help='initial weights')
    parser.add_argument('--data', type=str, default='data/voc.data', help='*.data path')
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

    mask_cfg, aux_util = get_thin_model(cfg=opt.cfg,
                                        data=opt.data,
                                        origin_weights=opt.weights,
                                        img_size=opt.img_size,
                                        batch_size=opt.batch_size,
                                        accumulate=opt.accumulate,
                                        prune_rate=opt.prune_rate,
                                        aux_epochs=opt.aux_epochs,
                                        ft_epochs=opt.ft_epochs,
                                        resume=opt.resume,
                                        cache_images=opt.cache_images)

    pruning(mask_cfg, progress_chkpt, aux_util)
