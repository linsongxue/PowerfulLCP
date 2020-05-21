import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F


def unique(num_list):  # 检查元素是否为单一
    if len(num_list) == len(set(num_list)):
        return True
    else:
        return False


def parse_prune_idx(module_defs, strategy="threshold"):
    """
    生成剪枝网络层索引
    :param module_defs:
    :param strategy:模式名称
    :return:一个是剪枝网络层索引，另一个是比例剪枝时网络层索引
    （当模式为'rate'时前一个参数会包含比例剪枝索引，后面的正常；当模式为'threshold'时，第二个返回值会变成空列表）
    """
    con_batch_idx = []  # 带batch的层
    con_idx = []  # 纯卷积层
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                con_batch_idx.append(i)
            else:
                con_idx.append(i)

    ignore_idx = set()
    shortcut_couple = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i - 1)
            shortcut_idx = (i + int(module_def['from']))  # 寻找残差层的网络序号
            if module_defs[shortcut_idx]['type'] == 'convolutional':
                ignore_idx.add(shortcut_idx)

                shortcut_couple.append([i - 1, shortcut_idx])
            elif module_defs[shortcut_idx]['type'] == 'shortcut':
                ignore_idx.add(shortcut_idx - 1)

                shortcut_couple.append([i - 1, shortcut_idx - 1])

    # 两个升采样层
    ignore_idx.add(84)
    ignore_idx.add(96)

    prune_idx = [idx for idx in con_batch_idx if idx not in ignore_idx]  # 可以不按照比例剪枝的层

    if strategy == "threshold":
        rate_only_idx = []
        return prune_idx, rate_only_idx
    elif strategy == "rate":
        ignore_idx.remove(84)
        ignore_idx.remove(96)
        rate_only_idx = [idx for idx in ignore_idx if idx in con_batch_idx]
        rate_only_idx.sort()
        prune_idx.extend(rate_only_idx)
        prune_idx.sort()
        assert unique(prune_idx), "解析出现错误"
        return prune_idx, rate_only_idx
    elif strategy == "rateV2":
        return prune_idx, shortcut_couple


class L1BatchNormOptimizer(object):

    @staticmethod
    def update_weight(module_list, prune_idx, sparsity_scalar=0.001):
        """
        To update the batch-norm scalar(after loss backward, before optimizer step)
        :param module_list: Model list class
        :param prune_idx: Index of layer which can be prune in free(list)
        :param sparsity_scalar: Float, the lambda of L1 regularization function
        :return: None
        """

        for idx in prune_idx:
            bn_layer = module_list[idx][1]
            bn_layer.weight.grad.data.add_(sparsity_scalar * torch.sign(bn_layer.weight.data))

    @staticmethod
    def update_bias(module_list, prune_idx, sparsity_scalar=0.01):
        """
        To update the batch-norm bias, the grad of bias add the abs of weight(the influence of weight will be important
        to the final results)
        :param module_list:Model list class
        :param prune_idx:Index of layer which can be prune in free(list)
        :param sparsity_scalar:Float, the lambda of L1 regularization function
        :return:None
        """

        for idx in prune_idx:
            bn_layer = module_list[idx][1]
            bn_layer.bias.grad.data.add_(sparsity_scalar * torch.sign(bn_layer.weight.data))

    @staticmethod
    def collect_data(module_list, index, weight_dict):
        """
        To collect the conv weight and BN weight.
        :param module_list: Model list class
        :param index: The index of layer, just collect one layer
        :param weight_dict: dictionary to save the weight
        :return: None
        """

        conv_layer = module_list[index][0]
        bn_layer = module_list[index][1]
        conv_weight = conv_layer.weight.detach()
        conv_weight = torch.sum(conv_weight, dim=(1, 2, 3))
        bn_weight = bn_layer.weight.detach()
        weight_dict['conv'].append(conv_weight)
        weight_dict['bn'].append(bn_weight)

    @staticmethod
    def update_weightv2(module_list, prune_idx):
        for idx in prune_idx:
            conv_layer = module_list[idx][0]
            bn_layer = module_list[idx][1]
            # weight_l1 代表卷积核的和的torch
            weight_l1 = F.relu(torch.sum(conv_layer.weight.detach(), dim=(1, 2, 3)))
            bn_layer.weight.grad.data.add_(0.01 * (weight_l1.max() - weight_l1) * torch.sign(bn_layer.weight.data))
            del weight_l1

    @staticmethod
    def update_shortcut(module_list, shortcut_couple):
        for idx in shortcut_couple:
            conv_layer1 = module_list[idx[0]][0]
            bn_layer1 = module_list[idx[0]][1]

            conv_layer2 = module_list[idx[1]][0]
            bn_layer2 = module_list[idx[1]][1]

            # 两层的系数做一个平均
            weight_l1 = (F.relu(torch.sum(conv_layer1.weight.detach(), dim=(1, 2, 3))) +
                         F.relu(torch.sum(conv_layer2.weight.detach(), dim=(1, 2, 3)))) / 2

            bn_layer1.weight.grad.data.add_(0.001 * (weight_l1.max() - weight_l1) * torch.sign(bn_layer1.weight.data +
                                                                                               bn_layer2.weight.data))
            bn_layer2.weight.grad.data.add_(0.001 * (weight_l1.max() - weight_l1) * torch.sign(bn_layer1.weight.data +
                                                                                               bn_layer2.weight.data))
            del weight_l1


class WeightPruneUtils(object):

    def __init__(self, weight_path, module_defs, strategy="threshold"):
        device = torch.device('cpu')  # 强制在cpu上进行转换
        check_point = torch.load(weight_path, map_location=device)
        if 'model' in check_point.keys():
            self.weight = check_point['model']
        else:
            self.weight = check_point

        del check_point

        self.strategy = strategy
        self.elements = ['module_list.{}.Conv2d.weight',
                         'module_list.{}.BatchNorm2d.weight',
                         'module_list.{}.BatchNorm2d.bias',
                         'module_list.{}.BatchNorm2d.running_mean',  # 平均数要单独进行一项剪枝步骤
                         'module_list.{}.BatchNorm2d.running_var']
        self.filter_list, self.route_list, self.route_dict, self.conv91_layer = create_channel_list(module_defs)
        self.num_layer = len(self.filter_list)
        self.filter_list_copy = self.filter_list.copy()  # 保存原本

    def _threshold_prune(self, prue_indexes, threshold):
        """
        按照阈值剪枝
        :param prue_indexes: 剪枝索引列表【list】
        :param threshold: 剪枝阈值【浮点数】
        :return:
        """
        progress_bar = tqdm(range(self.num_layer))  # 遍历每一个层
        mask_list = [torch.ones(3).to(torch.bool)]  # 根据BN层系数进行选择的mask，第一层全保存
        threshold = torch.tensor(threshold)

        # 开始逐层剪枝
        for index in progress_bar:
            if index in prue_indexes:  # 索引是需要剪枝的
                mask = torch.abs(self.weight['module_list.{}.BatchNorm2d.weight'.format(index)]) > threshold
                assert mask.sum().item() > 0, "please select a smaller threshold"
                self.filter_list[index] = mask.sum().item()  # item把torch变数

                Correct.biases(self, mask, index)

                for element in self.elements:
                    if len(self.weight[element.format(index)].shape) > 1:
                        self.weight[element.format(index)] = self.weight[element.format(index)][mask]
                        self.weight[element.format(index)] = self.weight[element.format(index)][:, mask_list[-1]]
                    else:
                        self.weight[element.format(index)] = self.weight[element.format(index)][mask]

            elif index in self.route_list:  # 只修改filter_list数，不会改变参数（本层本来就没有参数）
                if len(self.route_dict[index]) > 1:
                    # 一般情况下第一个参数是负数，第二个是正数
                    mask = torch.cat((mask_list[self.route_dict[index][0]], mask_list[self.route_dict[index][1] + 1]))
                else:
                    mask = mask_list[self.route_dict[index][0]]

                self.filter_list[index] = mask.sum().item()

            elif index in self.conv91_layer:
                self.weight['module_list.{}.91conv2d.weight'.format(index)] = \
                    self.weight['module_list.{}.91conv2d.weight'.format(index)][:, mask_list[-1]]

                # 索引不需要剪枝，就把掩膜变成全1模式
                num_channel = self.filter_list[index]
                mask = torch.ones(num_channel).to(torch.bool)

            else:
                # 不需要剪枝层只需把进通道数改变即可
                try:
                    self.weight['module_list.{}.Conv2d.weight'.format(index)] = \
                        self.weight['module_list.{}.Conv2d.weight'.format(index)][:, mask_list[-1]]
                except Exception as error:
                    if isinstance(error, KeyError):
                        print('{} layer without conv have passed'.format(index))
                        pass
                    else:
                        print(error)

                # 索引不需要剪枝，就把掩膜变成全1模式
                num_channel = self.filter_list[index]
                mask = torch.ones(num_channel).to(torch.bool)

            mask_list.append(mask)

    def _rate_prune(self, prune_indexes, prue_rate):
        """
        按照比例剪枝
        :param prue_rate:剪枝比例
        :param prune_indexes: 剪枝索引列表【list】
        :return:
        """
        progress_bar = tqdm(range(self.num_layer))
        mask_list = [torch.ones(3).to(torch.bool)]  # 根据BN层系数进行选择的mask，第一层全保存

        for index in progress_bar:
            if index in prune_indexes:
                feature_size = len(self.weight['module_list.{}.BatchNorm2d.weight'.format(index)])  # 统计batch norm维度
                remain_filters = round(feature_size * prue_rate)  # 确定剩下多少个通道
                self.filter_list[index] = remain_filters

                _, rank_index = torch.topk(torch.abs(self.weight['module_list.{}.BatchNorm2d.weight'.format(index)]),
                                           remain_filters)  # 选出排名靠前的通道索引，以便形成掩膜mask

                # 确定掩膜
                mask = torch.zeros(feature_size)
                mask[rank_index] = 1
                mask = mask.to(torch.bool)

                Correct.biases(self, mask, index)

                for element in self.elements:
                    if len(self.weight[element.format(index)].shape) > 1:
                        self.weight[element.format(index)] = self.weight[element.format(index)][mask]
                        self.weight[element.format(index)] = self.weight[element.format(index)][:, mask_list[-1]]
                    else:
                        self.weight[element.format(index)] = self.weight[element.format(index)][mask]

            elif index in self.route_list:
                if len(self.route_dict[index]) > 1:
                    # 一般情况下第一个参数是负数，第二个是正数
                    mask = torch.cat((mask_list[self.route_dict[index][0]], mask_list[self.route_dict[index][1] + 1]))
                else:
                    mask = mask_list[self.route_dict[index][0]]

                self.filter_list[index] = mask.sum().item()

            elif index in self.conv91_layer:
                self.weight['module_list.{}.91conv2d.weight'.format(index)] = \
                    self.weight['module_list.{}.91conv2d.weight'.format(index)][:, mask_list[-1]]

                num_channel = self.filter_list[index]
                mask = torch.ones(num_channel).to(torch.bool)

            else:
                try:
                    self.weight['module_list.{}.Conv2d.weight'.format(index)] = \
                        self.weight['module_list.{}.Conv2d.weight'.format(index)][:, mask_list[-1]]
                except Exception as error:
                    if isinstance(error, KeyError):
                        print('{} layer without conv have passed'.format(index))
                        pass
                    else:
                        print(error)

                num_channel = self.filter_list[index]
                mask = torch.ones(num_channel).to(torch.bool)

            mask_list.append(mask)

    # def _shortcut(self, prune_indexes, prue_rate):

    def prune(self, prune_indexes, save=True, list_save=True, new_file='pruned_weight.pt', **kwargs):  # 按照模式
        if self.strategy == "threshold":
            self._threshold_prune(prune_indexes, kwargs['threshold'])
        if self.strategy == "rate":
            self._rate_prune(prune_indexes, kwargs['rate'])

        if save:  # 生成两个文件 /一个.pt /一个.txt
            chkpt = {'model': self.weight}
            torch.save(chkpt, new_file)
            if list_save:
                list_file = new_file.replace('.pt', '.txt')
                with open(list_file, 'w') as f:
                    f.write(str(self.strategy) + ' = ')

                    if self.strategy == "threshold":
                        f.write(str(kwargs['threshold']) + '\n')
                    if self.strategy == "rate":
                        f.write(str(kwargs['rate']) + '\n')

                    for i, (original, new) in enumerate(zip(self.filter_list_copy, self.filter_list)):
                        formation = '%10s' + '%5d' + '  channels' + '%10s' + '%5d' + '  channels' + "\n"
                        s = ('layer{0:03d}'.format(i), original, "---->", new)
                        f.write(formation % s)
        else:
            return self.weight

    def generate_filter_list(self):
        return self.filter_list


def generate_cfg(filter_list, new_cfg, module_defs, hyp):
    """
    生成新的cfg文件
    :param filter_list: 新的通道数列表
    :param new_cfg:新cfg文件名
    :param module_defs: 原来的model_defs，list类型
    :param hyp:超参数
    :return: 无
    """

    generate_module_defs = module_defs.copy()
    for index in range(len(filter_list)):
        if generate_module_defs[index]['type'] == 'convolutional':
            generate_module_defs[index]['filters'] = filter_list[index]

    out = [hyp]
    out.extend(generate_module_defs)
    write_cfg(new_cfg, out)


def create_channel_list(module_defs):
    """
    根据model defs产生channel 列表
    :param module_defs: 列表，每个元素是一个字典
    :return: list
    """

    channel_list = []
    route_list = []  # 表示route层的索引序号
    conv91_layer = []  # 表示91分类conv层的索引序号
    route_dict = {}  # 查找每个route层来自哪两个层
    for index, module in enumerate(module_defs):
        if module['type'] == 'convolutional':
            filters = int(module['filters'])
            channel_list.append(filters)

        elif module['type'] == '91convolutional':
            filters = int(module['filters'])
            channel_list.append(filters)
            conv91_layer.append(index)

        elif module['type'] == 'shortcut':
            filters = channel_list[-1]
            channel_list.append(filters)

        elif module['type'] == 'yolo':
            channel_list.append(0)

        elif module['type'] == 'route':
            layers = [int(x) for x in module['layers'].split(',')]
            filters = sum([channel_list[i if i > 0 else i] for i in layers])
            channel_list.append(filters)
            route_list.append(index)
            route_dict[index] = layers

        elif module['type'] == 'upsample':
            filters = channel_list[-1]
            channel_list.append(filters)

    return channel_list, route_list, route_dict, conv91_layer  # 没有初始通道数


def threshold_recommend(weight, prune_indexes, n_thresholds):
    """
    推荐阈值
    :param weight:权重文件（object）
    :param prune_indexes: 剪枝索引
    :param n_thresholds: 需要分的类有多少
    :return:list
    """
    device = torch.device('cpu')
    check_point = torch.load(weight, map_location=device)
    if 'model' in check_point.keys():
        weight = check_point['model']
    else:
        weight = check_point
    data = np.array([])

    for index in prune_indexes:
        scalar = weight['module_list.{}.BatchNorm2d.weight'.format(index)].numpy()
        scalar = np.abs(scalar)
        data = np.concatenate((data, scalar))

    del weight

    data = np.reshape(data, (-1, 1))  # Kmeans聚类只能接受二维数据，因此要将一维扩展为二维
    recommender = KMeans(n_thresholds + 1)
    recommender.fit(data)

    centers = recommender.cluster_centers_
    centers = centers.flatten()  # 将数据平展开为一维
    centers = np.sort(centers)

    thresholds = [round((centers[i] + centers[i + 1]) * 0.5, 2) for i in range(n_thresholds)]

    return thresholds


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


def anchors_transform(anchors):
    anchors = anchors.flatten().astype(np.int)
    anchors = [str(num) for num in anchors]
    anchors = ','.join(anchors)
    return anchors


class Correct(object):

    @staticmethod
    def biases(self, mask, this_index):
        """
        对bias和running_mean进行修改，只修改数据不会修改其他的
        :param self: 类
        :param mask: 本层掩膜
        :param this_index:本层序号
        :return:无返回值
        """
        mask = mask.to(torch.float32)
        activation = F.leaky_relu((1 - mask) * self.weight['module_list.{}.BatchNorm2d.bias'.format(this_index)], 0.1)

        next_layer = [this_index + 1]
        if this_index == 79:
            next_layer.append(84)
        elif this_index == 91:
            next_layer.append(96)

        for layer in next_layer:
            try:
                conv_sum = torch.sum(self.weight['module_list.{}.Conv2d.weight'.format(layer)], dim=(2, 3))
                offset = torch.matmul(conv_sum, activation.reshape(-1, 1)).reshape(-1)
            except Exception as error:
                if isinstance(error, KeyError):
                    pass
                else:
                    raise error

            if next_layer in self.conv91_layer:
                self.weight['module_list.{}.91conv2d.bias'.format(next_layer)].data.add_(offset)
            else:
                try:
                    self.weight['module_list.{}.BatchNorm2d.running_mean'.format(next_layer)].data.sub_(offset)
                except Exception as error:
                    if isinstance(error, KeyError):
                        pass
                    else:
                        raise error

    # @staticmethod
    # def shortcut(self, mask, this_index):
    #
    # conv_sum = torch.sum(self.weight['module_list.{}.Conv2d.weight'.format(layer + 1)], dim=(2, 3))
    # offset = torch.matmul(conv_sum, activation.reshape(-1, 1)).reshape(-1)
    # print("{} shortcut layer skiped".format(layer))


    # @staticmethod
    # def route(self, mask, this_index):
