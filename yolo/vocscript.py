import os
from xml.etree.ElementTree import ElementTree
from tqdm import tqdm
import torch


def read_classname(data_path):
    dic = {}
    with open(data_path, 'r') as f:
        names = f.readlines()
    for i, name in enumerate(names):
        name = name.rsplit()[0]
        dic[name] = i

    return dic


def get_data(tree):
    size = tree.findall('size')[0]
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    data = []
    # TODO: 改变下面的文件名以获取不同的names文件
    dic = read_classname('yolov3/data/voc.names')
    objects = tree.findall('object')
    for object in objects:
        name = object.find('name').text
        xmin = float(object.find('bndbox').find('xmin').text)
        ymin = float(object.find('bndbox').find('ymin').text)
        xmax = float(object.find('bndbox').find('xmax').text)
        ymax = float(object.find('bndbox').find('ymax').text)
        x = round((xmax + xmin) / (2 * width), 8)
        y = round((ymax + ymin) / (2 * height), 8)
        w = round((xmax - xmin) / width, 8)
        h = round((ymax - ymin) / height, 8)
        data.append([dic[name], x, y, w, h])
    return data


# tree = ElementTree()
# tree = tree.parse('./VOC2012/Annotations/2007_000032.xml')
# # data = get_data(tree)
# # print(data)
# print(tree.findall('size')[0].find('width').text)
#
# dic = read_classname('yolov3/data/voc.names')
# print(dic)


# def voc_label(annotations_path, target_path, train=True):
#     files = os.listdir(annotations_path)
#     train_file = annotations_path.replace('Annotations', 'ImageSets') + '/Main/train.txt'
#     val_file = annotations_path.replace('Annotations', 'ImageSets') + '/Main/val.txt'
#     file_list = []
#     if train:
#         target_path = target_path + '/labels_train'
#         with open(train_file, 'r') as file_train:
#             for line in file_train.readlines():
#                 line = line.strip('\n') + '.xml'
#                 file_list.append(line)
#     else:
#         target_path = target_path + '/labels_val'
#         with open(train_file, 'r') as file_train:
#             for line in file_train.readlines():
#                 line = line.strip('\n') + '.xml'
#                 file_list.append(line)
#
#     for file in tqdm(file_list):
#         tree = ElementTree()
#         tree = tree.parse(annotations_path + '/' + file)
#         # filename = tree[1].tag.replace('jpg', 'xml')
#         # assert filename == file, print("don not match the annotations file")
#         datas = get_data(tree)
#         with open(target_path + '/' + file.replace('xml', 'txt'), 'w') as f:
#             for data in datas:
#                 s = str(data).replace('[', ' ').replace(']', ' ')  # 去除[],这两行按数据不同，可以选择
#                 s = s.replace(',', ' ') + '\n'
#                 f.write(s)


def voc_img_label():
    trainval_annotations_path_12 = './VOC2012/Annotations'
    trainval_annotations_path_07 = './VOC2007/Annotations'
    test_annotations_path = './VOC2007-test/Annotations'
    target_path1 = 'voc/labels_train'
    target_path2 = 'voc/labels_test'

    train_file_12 = trainval_annotations_path_12.replace('Annotations', 'ImageSets') + '/Main/train.txt'
    val_file_12 = trainval_annotations_path_12.replace('Annotations', 'ImageSets') + '/Main/val.txt'
    train_file_07 = trainval_annotations_path_07.replace('Annotations', 'ImageSets') + '/Main/train.txt'
    val_file_07 = trainval_annotations_path_07.replace('Annotations', 'ImageSets') + '/Main/val.txt'
    test_file = test_annotations_path.replace('Annotations', 'ImageSets') + '/Main/test.txt'

    train_file_list = []
    test_file_list = []
    with open(train_file_12, 'r') as file_train:
        for line in file_train.readlines():
            line = line.strip('\n') + '.xml'
            train_file_list.append(line)

    with open(val_file_12, 'r') as file_val:
        for line in file_val.readlines():
            line = line.strip('\n') + '.xml'
            train_file_list.append(line)

    with open(train_file_07, 'r') as file_train:
        for line in file_train.readlines():
            line = line.strip('\n') + '.xml'
            train_file_list.append(line)

    with open(val_file_07, 'r') as file_val:
        for line in file_val.readlines():
            line = line.strip('\n') + '.xml'
            train_file_list.append(line)

    with open(test_file, 'r') as file_val:
        for line in file_val.readlines():
            line = line.strip('\n') + '.xml'
            test_file_list.append(line)

    for file in tqdm(train_file_list):
        tree = ElementTree()
        try:
            tree = tree.parse(trainval_annotations_path_12 + '/' + file)
        except FileNotFoundError:
            tree = tree.parse(trainval_annotations_path_07 + '/' + file)
        datas = get_data(tree)
        with open(target_path1 + '/' + file.replace('xml', 'txt'), 'w') as f:
            for data in datas:
                s = str(data).replace('[', ' ').replace(']', ' ')  # 去除[],这两行按数据不同，可以选择
                s = s.replace(',', ' ') + '\n'
                f.write(s)

    for file in tqdm(test_file_list):
        tree = ElementTree()
        tree = tree.parse(test_annotations_path + '/' + file)
        datas = get_data(tree)
        with open(target_path2 + '/' + file.replace('xml', 'txt'), 'w') as f:
            for data in datas:
                s = str(data).replace('[', ' ').replace(']', ' ')  # 去除[],这两行按数据不同，可以选择
                s = s.replace(',', ' ') + '\n'
                f.write(s)

    with open('yolov3/data/voc_train.txt', 'w') as f:
        for line in train_file_list:
            line = line.replace('xml', 'jpg')
            f.write('../voc/images_train/' + line + '\n')

    with open('yolov3/data/voc_test.txt', 'w') as f:
        for line in test_file_list:
            line = line.replace('xml', 'jpg')
            f.write('../voc/images_test/' + line + '\n')


voc_img_label()
