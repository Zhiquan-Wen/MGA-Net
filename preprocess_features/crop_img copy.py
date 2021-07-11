# -*- coding: UTF-8 -*-
# crop the object region from the image based on the ground truth bounding box
import json
import h5py
from PIL import Image
import os

Dataset_path = './clevr_ref+_1.0/images/'

bbox_path = './data/obj_bbox_train_with_sequential.json'

img_train_path = './clevr_ref+_1.0/images/train'

os.mkdir(os.path.join(Dataset_path, 'obj_bbox_crop_image'))

bbox = json.load(open(bbox_path, 'r'))

img_name = list(bbox.keys())

for img_key in img_name:
    split = img_key.split('_')[1]
    bbox_key = str(int(img_key.split('_')[-1]))
    detected_bbox = list(bbox[img_key].values())
    detected_bbox = [[box[0], box[1], box[0]+box[2]-1, box[1]+box[3]-1] for box in detected_bbox]  # xywh -->xyxy
    img = Image.open(os.path.join(Dataset_path, split, img_key + '.png')).convert('RGB')
    os.mkdir(os.path.join(Dataset_path, 'obj_bbox_crop_image', split, img_key))
    for n, i in enumerate(detected_bbox):
        crop = img.crop(i)
        crop.save(os.path.join(Dataset_path, 'obj_bbox_crop_image', split, img_key, '%d.jpg'%n))




