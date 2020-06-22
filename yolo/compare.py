import os
import os.path as path
import math
from tqdm import tqdm
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
from utils.parse_config import parse_data_cfg, parse_model_cfg
from aux_net import AuxNetUtils, HookUtils, mask_converted
import logging

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

aux_weight = "../weights/aux.pt"
compare_analyse = 'compare_analyse.txt'
logging.basicConfig(filename=compare_analyse, filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()


def greedy_channel_select(origin_model, prune_cfg, origin_weights, select_layer, device, aux_util, data_loader,
                          pruned_rate):
    init_state_dict = mask_converted(prune_cfg, origin_weights, target=None)

    prune_model = Darknet(prune_cfg).to(device)
    prune_model.load_state_dict(init_state_dict, strict=True)
    del init_state_dict
    solve_sub_problem_optimizer = optim.SGD(prune_model.module_list[int(select_layer)].MaskConv2d.parameters(),
                                            lr=hyp['lr0'],
                                            momentum=hyp['momentum'])
    hook_util = HookUtils()
    handles = []

    info = aux_util.layer_info[int(select_layer)]
    in_channels = info['in_channels']
    remove_k = math.floor(in_channels * pruned_rate)
    k = in_channels - remove_k

    for name, child in origin_model.module_list.named_children():
        if name == select_layer:
            handles.append(child.BatchNorm2d.register_forward_hook(hook_util.hook_origin_input))

    aux_idx = aux_util.conv_layer_dict[select_layer]
    hook_layer_aux = aux_util.down_sample_layer[aux_idx]
    for name, child in prune_model.module_list.named_children():
        if name == select_layer:
            handles.append(child.BatchNorm2d.register_forward_hook(hook_util.hook_prune_input))
        elif name == hook_layer_aux:
            handles.append(child.register_forward_hook(hook_util.hook_prune_input))

    aux_net = aux_util.creat_aux_list(416, device, conv_layer_name=select_layer)
    chkpt_aux = torch.load(aux_weight, map_location=device)
    aux_net.load_state_dict(chkpt_aux['aux{}'.format(aux_idx)])
    del chkpt_aux

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        prune_model = torch.nn.parallel.DistributedDataParallel(prune_model, find_unused_parameters=True)
        prune_model.yolo_layers = prune_model.module.yolo_layers
        aux_net = torch.nn.parallel.DistributedDataParallel(aux_net, find_unused_parameters=True)

    nb = len(data_loader)
    prune_model.nc = 80
    prune_model.hyp = hyp
    prune_model.arc = 'default'
    prune_model.eval()
    aux_net.eval()
    MSE = nn.MSELoss(reduction='mean')

    greedy = torch.zeros(k)
    for i_k in range(k):
        pbar = tqdm(enumerate(data_loader), total=nb)
        print(('\n' + '%10s' * 8) % ('Stage', 'gpu_mem', 'iter', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'targets'))
        for i, (imgs, targets, _, _) in pbar:
            if len(targets) == 0:
                continue

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = origin_model(imgs)

            _, pruning_pred = prune_model(imgs)
            pruning_loss, _ = compute_loss(pruning_pred, targets, prune_model)
            hook_util.cat_to_gpu0('prune')

            aux_pred = aux_net(hook_util.prune_features['gpu0'][1])
            aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

            mse_loss = torch.zeros(1).to(device)
            mse_loss += MSE(hook_util.prune_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss

            loss.backward()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 3 + '%10.3g' * 5) % (
                'Pruning ' + select_layer, '%.3gG' % mem, '%g/%g' % (i_k, k), mse_loss, pruning_loss, aux_loss, loss,
                len(targets))
            pbar.set_description(s)

            hook_util.clean_hook_out('origin')
            hook_util.clean_hook_out('prune')

        grad = prune_model.module.module_list[int(select_layer)].MaskConv2d.weight.grad.detach().clone() ** 2
        grad = grad.sum((2, 3)).sqrt().sum(0)

        if i_k == 0:
            prune_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[:] = 1e-5
            _, non_greedy_indices = torch.topk(grad, k)
            logger.info('non greedy layer{}: selected==>{}'.format(select_layer, str(non_greedy_indices)))

        selected_channels_mask = prune_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask
        _, indices = torch.topk(grad * (1 - selected_channels_mask), 1)
        prune_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[indices] = 1
        greedy[i_k] = indices
        logger.info('greedy layer{} iter{}: indices==>{}'.format(select_layer, str(i_k), str(indices)))

        prune_model.zero_grad()

        pbar = tqdm(enumerate(data_loader), total=nb)
        mloss = torch.zeros(4).to(device)
        print(('\n' + '%10s' * 8) % ('Stage', 'gpu_mem', 'iter', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'targets'))
        for i, (imgs, targets, _, _) in pbar:

            if len(targets) == 0:
                continue

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = origin_model(imgs)

            _, pruning_pred = prune_model(imgs)
            pruning_loss, _ = compute_loss(pruning_pred, targets, prune_model)
            hook_util.cat_to_gpu0('prune')

            aux_pred = aux_net(hook_util.prune_features['gpu0'][1])
            aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

            mse_loss = torch.zeros(1).to(device)
            mse_loss += MSE(hook_util.prune_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss

            loss.backward()

            solve_sub_problem_optimizer.step()
            solve_sub_problem_optimizer.zero_grad()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            mloss = (mloss * i + torch.cat([mse_loss, pruning_loss, aux_loss, loss]).detach()) / (i + 1)
            s = ('%10s' * 3 + '%10.3g' * 5) % (
                'SubProm ' + select_layer, '%.3gG' % mem, '%g/%g' % (i_k, k), *mloss, len(targets))
            pbar.set_description(s)

            hook_util.clean_hook_out('origin')
            hook_util.clean_hook_out('prune')

    for handle in handles:
        handle.remove()

    logger.info(("greedy layer{}: selected==>{}".format(select_layer, str(greedy))))


def random_greedy_channel_select(origin_model, prune_cfg, origin_weights, select_layer, device, aux_util, data_loader,
                                 pruned_rate):
    init_state_dict = mask_converted(prune_cfg, origin_weights, target=None)

    prune_model = Darknet(prune_cfg).to(device)
    prune_model.load_state_dict(init_state_dict, strict=True)
    del init_state_dict
    solve_sub_problem_optimizer = optim.SGD(prune_model.module_list[int(select_layer)].MaskConv2d.parameters(),
                                            lr=hyp['lr0'],
                                            momentum=hyp['momentum'])
    hook_util = HookUtils()
    handles = []

    info = aux_util.layer_info[int(select_layer)]
    in_channels = info['in_channels']
    remove_k = math.floor(in_channels * pruned_rate)
    k = in_channels - remove_k

    for name, child in origin_model.module_list.named_children():
        if name == select_layer:
            handles.append(child.BatchNorm2d.register_forward_hook(hook_util.hook_origin_input))

    aux_idx = aux_util.conv_layer_dict[select_layer]
    hook_layer_aux = aux_util.down_sample_layer[aux_idx]
    for name, child in prune_model.module_list.named_children():
        if name == select_layer:
            handles.append(child.BatchNorm2d.register_forward_hook(hook_util.hook_prune_input))
        elif name == hook_layer_aux:
            handles.append(child.register_forward_hook(hook_util.hook_prune_input))

    aux_net = aux_util.creat_aux_list(416, device, conv_layer_name=select_layer)
    chkpt_aux = torch.load(aux_weight, map_location=device)
    aux_net.load_state_dict(chkpt_aux['aux{}'.format(aux_idx)])
    del chkpt_aux

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        prune_model = torch.nn.parallel.DistributedDataParallel(prune_model, find_unused_parameters=True)
        prune_model.yolo_layers = prune_model.module.yolo_layers
        aux_net = torch.nn.parallel.DistributedDataParallel(aux_net, find_unused_parameters=True)

    random_num = 100
    prune_model.nc = 80
    prune_model.hyp = hyp
    prune_model.arc = 'default'
    prune_model.eval()
    aux_net.eval()
    MSE = nn.MSELoss(reduction='mean')

    random_greedy = torch.zeros(k)
    for i_k in range(k):
        pbar = tqdm(range(random_num), total=random_num)
        print(('\n' + '%10s' * 8) % ('Stage', 'gpu_mem', 'iter', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'targets'))
        data_iter = iter(data_loader)
        for i in pbar:
            imgs, targets, _, _ = data_iter.next()

            if len(targets) == 0:
                continue

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = origin_model(imgs)

            _, pruning_pred = prune_model(imgs)
            pruning_loss, _ = compute_loss(pruning_pred, targets, prune_model)
            hook_util.cat_to_gpu0('prune')

            aux_pred = aux_net(hook_util.prune_features['gpu0'][1])
            aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

            mse_loss = torch.zeros(1).to(device)
            mse_loss += MSE(hook_util.prune_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss

            loss.backward()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            s = ('%10s' * 3 + '%10.3g' * 5) % (
                'Pruning ' + select_layer, '%.3gG' % mem, '%g/%g' % (i_k, k), mse_loss, pruning_loss, aux_loss, loss,
                len(targets))
            pbar.set_description(s)

            hook_util.clean_hook_out('origin')
            hook_util.clean_hook_out('prune')

        grad = prune_model.module.module_list[int(select_layer)].MaskConv2d.weight.grad.detach() ** 2
        grad = grad.sum((2, 3)).sqrt().sum(0)

        if i_k == 0:
            prune_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[:] = 1e-5

        selected_channels_mask = prune_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask
        _, indices = torch.topk(grad * (1 - selected_channels_mask), 1)
        prune_model.module.module_list[int(select_layer)].MaskConv2d.selected_channels_mask[indices] = 1
        random_greedy[i_k] = indices
        logger.info('random greedy layer{} iter{}: indices==>{}'.format(select_layer, str(i_k), str(indices)))

        prune_model.zero_grad()

        pbar = tqdm(range(random_num), total=random_num)
        mloss = torch.zeros(4).to(device)
        print(('\n' + '%10s' * 8) % ('Stage', 'gpu_mem', 'iter', 'MSELoss', 'PdLoss', 'AuxLoss', 'Total', 'targets'))
        for i in pbar:

            imgs, targets, _, _ = data_iter.next()

            if len(targets) == 0:
                continue

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            with torch.no_grad():
                _ = origin_model(imgs)

            _, pruning_pred = prune_model(imgs)
            pruning_loss, _ = compute_loss(pruning_pred, targets, prune_model)
            hook_util.cat_to_gpu0('prune')

            aux_pred = aux_net(hook_util.prune_features['gpu0'][1])
            aux_loss, _ = AuxNetUtils.compute_loss_for_aux(aux_pred, aux_net, targets)

            mse_loss = torch.zeros(1).to(device)
            mse_loss += MSE(hook_util.prune_features['gpu0'][0], hook_util.origin_features['gpu0'][0])

            loss = hyp['joint_loss'] * mse_loss + pruning_loss + aux_loss

            loss.backward()

            solve_sub_problem_optimizer.step()
            solve_sub_problem_optimizer.zero_grad()

            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            mloss = (mloss * i + torch.cat([mse_loss, pruning_loss, aux_loss, loss]).detach()) / (i + 1)
            s = ('%10s' * 3 + '%10.3g' * 5) % (
                'SubProm ' + select_layer, '%.3gG' % mem, '%g/%g' % (i_k, k), *mloss, len(targets))
            pbar.set_description(s)

            hook_util.clean_hook_out('origin')
            hook_util.clean_hook_out('prune')

    for handle in handles:
        handle.remove()

    logger.info(("random greedy layer{}: selected==>{}".format(select_layer, str(random_greedy))))


def run_compare(cfg, data, prune_cfg, batch_size, origin_weights):
    device = select_device('', apex=None, batch_size=batch_size)

    if device.type != 'cpu' and torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

    init_seeds()

    data_dict = parse_data_cfg(data)
    train_path = data_dict['valid']

    dataset = LoadImagesAndLabels(train_path, 416, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_labels=True,
                                  cache_images=False)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               num_workers=nw,
                                               shuffle=True,  # Shuffle=True unless rectangular training is used
                                               pin_memory=True,
                                               collate_fn=dataset.collate_fn)

    origin_model = Darknet(cfg).to(device)
    chkpt = torch.load(origin_weights, map_location=device)
    origin_model.load_state_dict(chkpt['model'], strict=True)
    aux_util = AuxNetUtils(origin_model, hyp)
    del chkpt

    for layer in aux_util.pruning_layer[7:]:
        greedy_channel_select(origin_model, prune_cfg, origin_weights, layer, device, aux_util, train_loader, 0.75)
        random_greedy_channel_select(origin_model, prune_cfg, origin_weights, layer, device, aux_util, train_loader,
                                     0.75)


if __name__ == "__main__":
    run_compare(cfg='./cfg/yolov3-voc.cfg',
                data='./data/voc.data',
                prune_cfg='./cfg/maskyolov3-voc.cfg',
                batch_size=16,
                origin_weights='../weights/converted-voc.pt')
