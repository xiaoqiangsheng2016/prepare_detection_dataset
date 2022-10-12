#
# test for f_as01
#

import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
np.random.seed(41)

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            # for shape in shapes:
            #     annotation = self._annotation(shape)
            #     self.annotations.append(annotation)
            #     self.ann_id += 1
            img_id = self._image_id(json_path)
            annotation = self._annotation(shapes, img_id)
            self.annotations.append(annotation)
            self.ann_id += 1

            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        # for k, v in self.classname_to_id.items():
        #     category = {}
        #     category['id'] = v
        #     category['name'] = k
        #     self.categories.append(category)
        category = {}
        category["supercategory"] = "person"
        category["id"] = 1
        category["name"] = "person"
        category["keypoints"] = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ]
        category["skeleton"] = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7]
        ]
        self.categories.append(category)

    def _image_id(self, image_path):
        return int(os.path.splitext(os.path.basename(image_path))[0])

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self._image_id(path)
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    # def _annotation(self, shape):
    #     label = shape['label']
    #     points = shape['points']
    #     annotation = {}
    #     annotation['id'] = self.ann_id
    #     annotation['image_id'] = self.img_id
    #     annotation['category_id'] = int(self.classname_to_id[label])
    #     annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
    #     annotation['bbox'] = self._get_box(points)
    #     annotation['iscrowd'] = 0
    #     annotation['area'] = 1.0
    #     return annotation
    def _annotation(self, shapes, img_id):
        points = []
        for shape in shapes:
            points.append((int(shape['points'][0][0]), int(shape['points'][0][1]), int(2)))
        annotation = {}
        annotation['segmentation'] = [[]]
        annotation['num_keypoints'] = len(points)
        annotation['keypoints'] = np.array(points).reshape(-1).tolist()
        annotation['id'] = self.ann_id
        annotation['image_id'] = img_id
        annotation['category_id'] = 1
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y, v in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':

    data_type = "front"
    labelme_path = f"dataset/labelme/f_ca01/{data_type}"
    saved_coco_path = f"./output/f_ca01/{data_type}"

    # 创建文件
    if not os.path.exists(f"{saved_coco_path}/coco/annotations/"):
        os.makedirs(f"{saved_coco_path}/coco/annotations/")
    if not os.path.exists(f"{saved_coco_path}/coco/images/train2017/"):
        os.makedirs(f"{saved_coco_path}/coco/images/train2017")
    if not os.path.exists(f"{saved_coco_path}/coco/images/val2017/"):
        os.makedirs(f"{saved_coco_path}/coco/images/val2017")
    # 获取images目录下所有的joson文件列表
    json_list_path = glob.glob(labelme_path + "/*.json")
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=0.1)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, f'{saved_coco_path}/coco/annotations/person_keypoints_train2017.json')
    for file in train_path:
        # shutil.copy(file.replace("json","jpg"), f"{saved_coco_path}/coco/images/train2017/")
        filename, ext = os.path.splitext(file)
        if os.path.exists(filename + ".jpg"):
            shutil.copy(filename + ".jpg", f"{saved_coco_path}/coco/images/train2017/")
        elif os.path.exists(filename + ".jpeg"):
            shutil.copy(filename + ".jpeg", f"{saved_coco_path}/coco/images/train2017/")
        elif os.path.exists(filename + ".png"):
            shutil.copy(filename + ".png", f"{saved_coco_path}/coco/images/train2017/")

    for file in val_path:
        # shutil.copy(file.replace("json","jpg"), f"{saved_coco_path}/coco/images/val2017/")
        filename, ext = os.path.splitext(file)
        if os.path.exists(filename + ".jpg"):
            shutil.copy(filename + ".jpg", f"{saved_coco_path}/coco/images/val2017/")
        elif os.path.exists(filename + ".jpeg"):
            shutil.copy(filename + ".jpeg", f"{saved_coco_path}/coco/images/val2017/")
        elif os.path.exists(filename + ".png"):
            shutil.copy(filename + ".png", f"{saved_coco_path}/coco/images/val2017/")

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, f'{saved_coco_path}/coco/annotations/person_keypoints_val2017.json')
