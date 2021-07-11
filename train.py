from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

from logger1 import Logger
import os
import os.path as osp
import sys
import numpy as np
import time
import logging
import random
from pprint import pprint

import torch
import torch.nn as nn

from Datasets.Dataloader import CLEVRDataLoader
from models.MGA_Nets import GraphMatching
import utils as model_utils
from eval import eval_split
from opt import parse_opt

def settings(opt):
    # set random seed
    seed = opt['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # cudnn

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(opt['gpuid'])


def get_logger(save_path):
    """
    initialize logger
    """
    logger = logging.getLogger('Train')
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "train_test.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger

def write_settings(save_path, opt):
    with open(os.path.join(save_path, 'settings.log'), 'w') as f:
        for k, v in opt.items():
            f.write(str(k) + ":" + str(v) + "\n")

def set_save_path(save_path, vis_features, rnn_layer_num, batch_size, rnn_drop_out, jemb_drop, word_vec_drop, rounds, des):
    save_path = save_path + \
                "log_vis_features_{}_rnn_layer_num_{}_rnn_drop_out_{}_jemb_drop_{}_word_vec_drop_{}_bs{:d}_GGNN_{}_rounds_{}/" \
                    .format(vis_features, rnn_layer_num, rnn_drop_out, jemb_drop, word_vec_drop, batch_size, rounds, des)

    if os.path.exists(save_path):
        print("{} file exist!".format(save_path))
        action = input("Select Action: d (delete) / q (quit):").lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(save_path)
        else:
            raise OSError("Directory {} exits!".format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def main(args):
    opt = vars(args)  # 将args转换为字典
    settings(opt)
    save_path = "./saved_output/"
    save_path_ = set_save_path(save_path, 'resnet101', opt['rnn_num_layers'], opt['batch_size'], opt['rnn_drop_out'], 
                               opt['jemb_drop_out'], opt['word_drop_out'], opt['rounds'], opt['saved_des'])
    logger1 = Logger(os.path.join(save_path_, 'train_tensorboard'))
    logger = get_logger(save_path_)
    logger.info("|===> logger will be saved in {}".format(save_path_))
    write_settings(save_path_, opt)

    train_loader, test_loader = CLEVRDataLoader(opt['train_vis_feat_path'], opt['train_refexps_path'],
                                                opt['train_bounding_box_path'], opt['train_answer_obj_path'],
                                                opt['test_vis_feat_path'], opt['test_refexps_path'], 
                                                opt['test_bounding_box_path'], opt['test_answer_obj_path'],
                                                opt['vocab_path'], opt['batch_size'], 1).get_dataloader()

    logger.info("finish construct the data_loader")

    opt['vocab_size'] = train_loader.dataset.vocab_size
    opt['vis_dim'] = train_loader.dataset.vis_dim
    opt['loc_dim'] = train_loader.dataset.loc_dim
    opt['edge_dim'] = train_loader.dataset.edge_dim

    model = GraphMatching(opt)
    logger.info("finish design the model")

    # set up criterion
    ce_crit = nn.CrossEntropyLoss()
    # move to GPU
    if opt['gpuid'] >= 0:
        model.cuda()
        ce_crit.cuda()

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])

    resume_epoch = 0
    best_val_score = 0

    if opt['resume']:
        data = torch.load(opt['resume'])
        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        resume_epoch = data['epoch']
        best_val_score = resume_epoch['best_acc']

    # start training
    data_time, model_time = 0, 0
    lr = optimizer.param_groups[0]['lr']
    iter = 0
    count = 0
    loss_total = 0
    acc_ = 0
    totals = 0
    truthes = 0
    loss_ = {}
    for epoch in range(resume_epoch, opt['max_epoches']):
        # set mode
        model.train()
        # time
        T = {}
        tic = time.time()
        # load one batch data
        for i, (vis, ref, lfeats, loc_feats, same_feats, answer_obj, num_obj, num_refexps, sent_to_img_feat) in\
                enumerate(train_loader):
            batch_vis_feats = vis.cuda()
            batch_lfeats = lfeats.cuda()
            batch_same_feats = same_feats.cuda()
            batch_loc_feats = loc_feats.cuda()
            labels = ref.cuda()
            batch_num_obj = num_obj
            batch_num_refexps = num_refexps

            gt_ann_ids = answer_obj.cuda()
            gt = answer_obj.detach().cpu().numpy()

            T['data'] = time.time() - tic

            # zero gradient
            optimizer.zero_grad()

            # forward
            tic = time.time()
            scores = model(batch_vis_feats, batch_lfeats, batch_same_feats, batch_loc_feats, labels, batch_num_obj,
                           sent_to_img_feat)
            scores_ = scores.data.cpu().numpy()
            predict = np.argmax(scores_, 1)
            truth = (predict == gt).sum()
            total = scores_.shape[0]
            loss = ce_crit(scores, gt_ann_ids)

            loss.backward()
            model_utils.clip_gradient(optimizer, opt['grad_clip'])
            optimizer.step()
            T['model'] = time.time() - tic
            loss_ = loss.item()
            loss_total += loss_
            count += 1

            data_time += T['data']
            model_time += T['model']

            truthes += truth
            totals += total
            acc_ = truthes / totals * 100

            iter += 1

            if iter % opt['losses_log_every'] == 0 and iter != 0:
                # print stats
                logger.info(
                    'iter[%s](epoch[%s]), train_acc=%.3f, train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
                    % (iter, epoch, acc_, loss_, lr, data_time / opt['losses_log_every'],
                       model_time / opt['losses_log_every']))
                data_time, model_time = 0, 0

            tic = time.time()

        val_loss, acc = eval_split(test_loader, model, ce_crit, 'val', opt)
        logger.info('validation loss: %.2f' % val_loss)
        logger.info('validation acc : %.2f%%\n' % (acc * 100.0))

        # save model if best
        current_score = acc

        if best_val_score is None or current_score > best_val_score:
            best_val_score = current_score
            checkpoint_path = osp.join(save_path_, 'best' + '.pth')
            checkpoint = {}
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['opt'] = opt
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['best_acc'] = best_val_score
            torch.save(checkpoint, checkpoint_path)
            logger.info('model saved to %s' % checkpoint_path)

        train_loss = loss_total / count
        loss_total = 0
        count = 0
        truthes = 0
        totals = 0
        data_tensorboard = {'train_acc': acc_, 'train_loss': train_loss, 'test_loss': val_loss, 'test_acc': acc, 'lr': optimizer.param_groups[0]['lr']}
        for tag, value in data_tensorboard.items():
            logger1.scalar_summary(tag, value, epoch)
        epoch += 1


if __name__ == "__main__":
    args = parse_opt()
    main(args)
