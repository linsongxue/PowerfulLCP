import json
import numpy as np
from pycocotools.coco import COCO


def class91_to_class80():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, -1, 24, 25, -1, -1,
         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, -1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
         53, 54, 55, 56, 57, 58, 59, -1, 60, -1, -1, 61, -1, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, -1, 73, 74, 75,
         76, 77, 78, 79, -1]
    return x


def class91_to_class91():
    x = []
    for i in range(91):
        x.append(i)
    return x


json_file = "../instances_train2017.json"
version = json_file.split('_')[-1].split('.')[0]
root = '../mscoco2014/labels_' + version.replace('2017', '2014') + "/"


def label_generate():
    coco = COCO(json_file)
    ids = coco.getImgIds()
    class_list = class91_to_class80()
    id = ids[1]
    for id in ids:
        img_info = coco.loadImgs(id)
        ann_ids = coco.getAnnIds(id)
        ann_list = coco.loadAnns(ann_ids)
        with open(root + img_info[0]['file_name'].replace('jpg', 'txt'), 'w') as f:
            for ann in ann_list:
                cat = class_list[int(ann['category_id']) - 1]
                if cat == -1:
                    continue
                f.write(str(cat) + ' ')
                bbox = np.array(ann['bbox'])
                bbox[0:2] = bbox[0:2] + bbox[2:4] / 2.0
                bbox[0] = round(bbox[0] / img_info[0]['width'], 8)
                bbox[1] = round(bbox[1] / img_info[0]['height'], 8)
                bbox[2] = round(bbox[2] / img_info[0]['width'], 8)
                bbox[3] = round(bbox[3] / img_info[0]['height'], 8)
                s = str(bbox).replace('[', ' ').replace(']', ' ')  # 去除[],这两行按数据不同，可以选择
                s = s.replace(',', ' ') + '\n'
                f.write(s)


def list_generate():
    coco = COCO(json_file)
    ids = coco.getImgIds()
    with open('./data/' + version.replace('2017', '2014') + '.txt', 'w') as f:
        for id in ids:
            img_name = coco.loadImgs(id)[0]['file_name']
            f.write(root + img_name + '\n')


def genereate():
    list_generate()
    label_generate()


if __name__ == "__main__":
    label_generate()
