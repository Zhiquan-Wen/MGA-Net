import os
import numpy as np
import torch
import json
import h5py
import torchvision
from scipy.misc import imread, imresize
import argparse
import torch.nn as nn
from functools import cmp_to_key
import tqdm

# 这份文件专门为clevr_ref+提取obj features

parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', required=True)  # 输入图片的目录，包含所有图片的目录，图片的目录下包含1.jpg,2.jpg...
parser.add_argument('--output_h5_file', required=True)  # 输出h5文件的目录

parser.add_argument('--image_height', default=320, type=int)  # resize 成320*320
parser.add_argument('--image_width', default=320, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--gpu', default=0)

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, input):
        x = input.view(-1)
        return x


def build_model(args):
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    total_model = getattr(torchvision.models, args.model)(pretrained=True)

    model = nn.Sequential(
        total_model.conv1,
        total_model.bn1,
        total_model.relu,
        total_model.maxpool,
        total_model.layer1,
        total_model.layer2,
        total_model.layer3,
        nn.AdaptiveAvgPool2d(1)
    )
    model = model.cuda()
    model.eval()

    return model


def run_batch(cur_batch, model):  # 一张图片的所有bounding box算一个batch
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)  # CLEVR

    image_patch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_patch = (image_patch / 255.0 - mean) / std
    image_patch = torch.from_numpy(image_patch).type_as(torch.FloatTensor())

    image_patch = image_patch.cuda()
    feats = model(image_patch).view(image_patch.shape[0], -1)
    feats = feats.data.cpu().clone().numpy()

    return feats

def compare(cp1, cp2):
        cp_1 = int(cp1)
        cp_2 = int(cp2)

        if cp_1 - cp_2 <= 0:
            return -1
        else:
            return 1

def main(args):
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)  # gpuid
    input_path = {}

    global data_split

    for fn in os.listdir(args.input_image_dir):  # this folder contain all cropped images, such as [CLEVR_train_000000: 1.jpg, 2.jpg, ...; CLEVR_train_000001: 1.jpg, 2.jpg ...]
        name = str(fn)
        data_split = name.split('_')[1]
        for fn_ in os.listdir(os.path.join(args.input_image_dir, fn)):  # CLEVR_train_000000: 1.jpg, 2.jpg ...
            if not fn_.endswith('.jpg'): continue
            if name not in input_path:
                input_path[name] = [os.path.join(args.input_image_dir, name, fn_)]
            else:
                input_path[name].append(os.path.join(args.input_image_dir, name, fn_))  # 搜集每张图片的裁剪下来的图片的path
        keys = input_path[name]
        new_keys = []
        sorted_path = []
        for k in keys:
            new_keys.append(k.split('/')[-1].split('.')[0])  # obtain obj id for orderring，
        new_keys.sort(key=cmp_to_key(compare))

        for k_ in new_keys:
            for keys_ in keys:
                if '/' + k_ +'.jpg' in keys_:
                    sorted_path.append(keys_)  # 确保obj顺序正确
                    break
        input_path[name] = sorted_path

    print("sorted path : {}".format([input_path['CLEVR_%s_004110'%data_split], input_path['CLEVR_%s_000004'%data_split]]))

    kes = input_path.keys()
    kes_ = []
    for j in kes:
        j_ = j.split('_')[-1]
        kes_.append(j_)
    kes_.sort(key=cmp_to_key(compare))
    input_path_ = {}
    for i in kes_:
        key = 'CLEVR_%s_'%(data_split) + i
        input_path_[key] = input_path[key]

    print('sorted input path: {}'.format([{n: input_path_[n]} for n in list(input_path_.keys())[:10]]))

    img_size = (args.image_height, args.image_width)

    num_images = len(kes)

    model = build_model(args)
    print("finish construct model")

    count = 0

    with h5py.File(args.output_h5_file, 'w') as f:
        for name in tqdm.tqdm(input_path_):
            f_dataset = f.create_dataset(name=name, shape=(len(input_path_[name]), 1024))
            cur_batch = []
            for path in input_path_[name]:
                data = path.split('/')[-1]
                img = imread(path, mode='RGB')
                img = imresize(img, img_size, interp='bicubic')
                img = img.transpose(2, 0, 1)[None]
                cur_batch.append(img)
            feats = run_batch(cur_batch, model)
            f_dataset[:, :] = feats
            count += 1

            if count % 1000 == 0:
                print("finish images : [%d/%d]"%(count, num_images))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
