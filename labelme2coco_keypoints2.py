#
# gen dataset for alphapose

import os
import json
import numpy as np
import glob
import shutil
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
np.random.seed(41)

class Lableme2CoCo:

    def __init__(self, labels):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.labels = labels
        # self.labels = [
        #     '颈围',
        #     '前腋宽',
        #     '上胸围',
        #     '下胸围',
        #     '胃围',
        #     '腰围',
        #     '腹围',
        #     '臀围',
        #     '第一大腿围',
        #     '第三大腿围',
        #     '第五大腿围',
        #     '膝上围',
        #     '膝盖围',
        #     '膝下围',
        #     '腿肚围',
        #     '腿肚下围',
        #     '脚腕以上',
        #     '脚腕',
        #     '肩胸长',
        #     '乳长',
        #     '腹长',
        #     '头顶',
        #     'BP上',
        #     'BP右',
        #     'BP左',
        #     '裆部',
        #     '地面',
        #     '肚脐',
        #     '肩凸距',
        #     '肩凹距',
        #     '肩宽',
        #     '第一上臂围',
        #     '第二上臂围',
        #     '第三上臂围',
        #     '手肘',
        #     '第一下臂围',
        #     '第二下臂围',
        #     '手腕'
        # ]

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in tqdm(json_path_list):
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

        filename, ext = os.path.splitext(path)
        if os.path.exists(filename + ".jpg"):
            img_ext = ".jpg"
        elif os.path.exists(filename + ".jpeg"):
            img_ext = ".jpeg"
        elif os.path.exists(filename + ".png"):
            img_ext = ".png"
        else:
            raise ValueError("image ext name is error !")

        # img_x = utils.img_b64_to_arr(obj['imageData'])
        img_x = cv2.imread(filename + img_ext)
        image['height'] = img_x.shape[0]
        image['width'] = img_x.shape[1]
        image['id'] = self._image_id(path)

        image['file_name'] = os.path.basename(path).replace(".json", img_ext)
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

    @classmethod
    def get_shape_by_label(cls, shapes, label):
        for shape in shapes:
            if shape['label'] == label:
                return shape
        return None

    def _annotation(self, shapes, img_id):
        points = []
        for label in self.labels:
            shape = self.__class__.get_shape_by_label(shapes, label)
            if shape is None:
                raise ValueError(f'{img_id} {label} is error!')

            if shape['shape_type'] == 'line':
                points.append((int(shape['points'][0][0]), int(shape['points'][0][1]), int(2)))
                points.append((int(shape['points'][1][0]), int(shape['points'][1][1]), int(2)))
            elif shape['shape_type'] == 'point':
                points.append((int(shape['points'][0][0]), int(shape['points'][0][1]), int(2)))
            else:
                print("error shape type!!")

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


def show_sample_kpts(labels, image_file, json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        obj = json.load(f)
    shapes = obj['shapes']
    image = cv2.imread(image_file)
    idx = 0
    for label in labels:
        shape = Lableme2CoCo.get_shape_by_label(shapes, label)
        if shape is None:
            raise ValueError(f'{label} is not exist in json!')

        if shape['shape_type'] == 'line':
            pt1 = (int(shape['points'][0][0]), int(shape['points'][0][1]))
            pt2 = (int(shape['points'][1][0]), int(shape['points'][1][1]))
            cv2.line(image, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)
            cv2.putText(image, str(idx), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
            idx = idx+1
            cv2.putText(image, str(idx), pt2, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
            idx = idx + 1
            # points.append((int(shape['points'][0][0]), int(shape['points'][0][1]), int(2)))
            # points.append((int(shape['points'][1][0]), int(shape['points'][1][1]), int(2)))
        elif shape['shape_type'] == 'point':
            pt1 = (int(shape['points'][0][0]), int(shape['points'][0][1]))
            # points.append((int(shape['points'][0][0]), int(shape['points'][0][1]), int(2)))
            cv2.circle(image, center=pt1, radius=3, color=(255, 0, 0), thickness=-1)
            cv2.putText(image, str(idx), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)
            idx = idx+1
        else:
            print("error shape type!!")

    dsize = (int(image.shape[1]*800/image.shape[0]), int(800))
    image = cv2.resize(image, dsize=dsize)
    cv2.imshow("img", image)
    cv2.waitKey(100000)

if __name__ == '__main__':

    data_type = "side"
    labelme_path = f"dataset/labelme/post_data/{data_type}"
    saved_coco_path = f"./output/kpts_dataset/{data_type}"
    if data_type == 'front':
        labels = [
            '颈围',
            '前腋宽',
            '上胸围',
            '下胸围',
            '胃围',
            '腰围',
            '腹围',
            '臀围',
            '第一大腿围',
            '第三大腿围',
            '第五大腿围',
            '膝上围',
            '膝盖围',
            '膝下围',
            '腿肚围',
            '腿肚下围',
            '脚腕以上',
            '脚腕',
            '肩胸长',
            '乳长',
            '腹长',
            '头顶',
            'BP上',
            'BP右',
            'BP左',
            '裆部',
            '地面',
            '肚脐',
            '肩凸距',
            '肩凹距',
            '肩宽',
            '第一上臂围',
            '第二上臂围',
            '第三上臂围',
            '手肘',
            '第一下臂围',
            '第二下臂围',
            '手腕'
        ]
    elif data_type == 'side':
        labels = [
            '侧面身高',
            '颈围',
            '腋下围',
            '胸围',
            '下胸围',
            '胃围',
            '腰围',
            '腹围',
            '臀围',
            '第一腿围',
            '第三腿围',
            '第五腿围',
            '膝上围',
            '膝盖围',
            '膝下围',
            '腿肚围',
            '脚腕围',
            '锁骨中',
            '腿腹交界',
            '腿肚下围',
            '脚腕以上'
        ]
    else:
        labels = []
        raise ValueError("data_type is error")

    # show_sample_kpts(labels, image_file=os.path.join(labelme_path, "00000001.jpg"), json_file=os.path.join(labelme_path, "00000001.json"))
    # exit(0)

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
    l2c_train = Lableme2CoCo(labels=labels)
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, f'{saved_coco_path}/coco/annotations/person_keypoints_train2017.json')
    for file in tqdm(train_path):
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
    l2c_val = Lableme2CoCo(labels=labels)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, f'{saved_coco_path}/coco/annotations/person_keypoints_val2017.json')
