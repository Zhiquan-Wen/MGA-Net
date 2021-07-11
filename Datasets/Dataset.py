import operator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import numpy as np


class CLEVRDataset(Dataset):
    def __init__(self, vis_feat_path, refexps_path, bounding_box_path, answer_obj_path, vocab_path):
        super(CLEVRDataset, self).__init__()
        vis_feat = h5py.File(vis_feat_path, 'r')
        refexps_idx = h5py.File(refexps_path, 'r') 
        answer_obj = json.load(open(answer_obj_path, 'r'))  # {image_name:{refexp_id: answer_obj}} 从0开始索引
        bounding_box = json.load(open(bounding_box_path, 'r'))  # 每张图片对应的每个obj的bounding box {img_name: {obj_id: bounding box}}
        self.vocab = json.load(open(vocab_path, 'r'))  # vocab path 

        self.idx_to_tokens = {v: k for k, v in self.vocab.items()}

        self.img_names = list(refexps_idx.keys())

        # transform the image name to index
        self.idx_to_img_names = {k: v for k, v in enumerate(self.img_names)}
        self.img_names_to_idx = {v: k for k, v in enumerate(self.img_names)}

        self.vis_feat = {}
        self.refexps = {}
        self.answer_obj = {}
        self.bounding_box = {}

        for img_name in self.img_names:
            self.vis_feat[img_name] = vis_feat[img_name].value  # {img_name: [num_obj, 512]} 按顺序排下来的
            self.refexps[img_name] = refexps_idx[img_name].value  # {img_name: [num_ref, max_dim]} # 按refexp顺序排下来的
            self.bounding_box[img_name] = bounding_box[img_name]  # {img_name: {obj_id: bounding box}}
            self.answer_obj[img_name] = answer_obj[img_name]  # {img_name: {origin_ref_idx: [obj_id]}}


        original_ref_idx = []
        for names, values in self.answer_obj.items():
            original_ref_idx += list(values.keys())  # 将所有refexps的原始id统计下来

        self.idx_to_ref = {k: v for k, v in enumerate(original_ref_idx)}  # new ref_idx to original ref_idx :use to get answer obj
        self.ref_to_idx = {v: k for k, v in enumerate(original_ref_idx)}

        # attributes:
        self.vis_dim = 1024
        self.edge_dim = 5
        self.loc_dim = 5

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __getitem__(self, index):
        img_name = self.idx_to_img_names[index]
        img_features = torch.from_numpy(self.vis_feat[img_name]).float()
        refexps = torch.from_numpy(self.refexps[img_name]).long()
        num_obj = len(img_features)
        num_refexps = len(refexps)
        lfeats = torch.from_numpy(self.compute_lfeats(num_obj, img_name)).float()
        same_feats = self.compute_same_feature(num_obj, img_name)
        loc_feats = self.compute_loc_feature(num_obj, img_name)
        answer_obj = torch.LongTensor(list(np.concatenate(np.array(list(self.answer_obj[img_name].values())), 0)))

        return img_features, refexps, lfeats, same_feats, loc_feats, answer_obj, num_obj, num_refexps

    def __len__(self):
        return len(self.img_names)

    def compute_lfeats(self,max_obj_num, img_name):
        # each image is 480*320
        lfeats = np.zeros((max_obj_num, 5))
        bounding_box_all = self.bounding_box[img_name]
        bounding_box_all = np.array(list(bounding_box_all.values()))
        for ix, box in enumerate(bounding_box_all):
            lfeats[ix] = np.array([box[0] / 480., box[1] / 320., (box[0] + box[2] - 1) / 480.,
                                   (box[1] + box[3] - 1) / 320., (box[2] * box[3]) / (480. * 320.)])
        return lfeats

    def compute_loc_feature(self, max_obj_num, img_name):
        vis_features = self.vis_feat[img_name]
        dif_lfeats = np.zeros((max_obj_num, max_obj_num, (5 + 1024 + 5)))
        bounding_box_all = self.bounding_box[img_name]
        bounding_box_all = np.array(list(bounding_box_all.values()))
        for i, box_i in enumerate(bounding_box_all):
            rcx, rcy, rw, rh = box_i[0] + box_i[2] / 2, box_i[1] + box_i[3] / 2, box_i[2], box_i[3]
            for j, box_j in enumerate(bounding_box_all):
                if j != i:
                    edge_feats = np.array([(box_j[0] - rcx) / rw,
                                           (box_j[1] - rcy) / rh,
                                           (box_j[0] + box_j[2] - rcx) / rw,
                                           (box_j[1] + box_j[3] - rcy) / rh,
                                           (box_j[2] * box_j[3]) / (rw * rh)])
                    loc_feats = np.array(
                        [box_j[0] / 480., box_j[1] / 320., (box_j[0] + box_j[2] - 1) / 480.,
                         (box_j[1] + box_j[3] - 1) / 320., (box_j[2] * box_j[3]) / (480. * 320.)])
                    vis_feats = vis_features[j]
                    dif_lfeats[i, j:(j + 1), :1034] = np.concatenate((edge_feats, vis_feats, loc_feats))
        dif_lfeats_ = dif_lfeats.reshape(-1)
        # remove the diagonal values of the matrix
        dif_lfeats_ = np.delete(dif_lfeats_, [i + j for i in range(0, max_obj_num ** 2 * 1034, (max_obj_num + 1) * 1034)
                                             for j in range(0, 1034)]).reshape(max_obj_num*(max_obj_num - 1), 1034)
        dif_lfeats_ = torch.from_numpy(dif_lfeats_).float()
        return dif_lfeats_

    def compute_same_feature(self, max_obj_num, img_name):
        same_feats = np.zeros((max_obj_num, max_obj_num, 2 * 1029))
        vis_features = self.vis_feat[img_name]
        lfeats = self.compute_lfeats(max_obj_num, img_name)
        num_obj = len(vis_features)
        for i in range(num_obj):
            vis_i_feat = vis_features[i]
            lfeat_i = lfeats[i]
            feat_i = np.concatenate((vis_i_feat, lfeat_i))
            for j in range(num_obj):
                if j == i:
                    continue
                else:
                    vis_j_feat = vis_features[j]
                    lfeat_j = lfeats[j]
                    feat_j = np.concatenate((vis_j_feat, lfeat_j))
                    same_feats[i, j: (j + 1), :2 * 1029] = np.concatenate((feat_i, feat_j))
        same_feats_ = same_feats.reshape(-1)
        # remove the diagonal values of the matrix
        same_feats_ = np.delete(same_feats_,
                               [i + j for i in range(0, max_obj_num ** 2 * 2 * 1029, (max_obj_num + 1) * 2 * 1029)
                                for j in range(0, 2 * 1029)]).reshape(max_obj_num*(max_obj_num - 1), 2 * 1029)
        same_feats_ = torch.from_numpy(same_feats_).float()
        return same_feats_
