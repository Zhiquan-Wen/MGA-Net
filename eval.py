from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from pprint import pprint

import torch
import torch.nn.functional as F

def eval_split(loader, model, crit, split, opt):
    verbose = opt.get('verbose', True)
    assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'

    # set mode
    model.eval()

    # initialize
    loss_sum = 0
    loss_evals = 0
    acc = 0
    model_time = 0
    data_time = 0
    total_refexps = 0
    T = {}
    total_samples = len(loader.dataset.idx_to_ref)
    with torch.no_grad():
        tic = time.time()
        for i, (vis, ref, lfeats, loc_feats, same_feats, answer_obj, num_obj, num_ref, sent_to_img_feat) in \
                enumerate(loader):

            batch_vis_feats = vis.cuda()
            batch_lfeats = lfeats.cuda()
            batch_same_feats = same_feats.cuda()
            batch_loc_feats = loc_feats.cuda()
            labels = ref.cuda()
            batch_num_obj = num_obj

            gt_ann_ids = answer_obj.cuda()
            gt = answer_obj.detach().cpu().numpy()

            T['data'] = time.time() - tic

            tic = time.time()
            scores = model(batch_vis_feats, batch_lfeats, batch_same_feats, batch_loc_feats, labels, batch_num_obj,
                           sent_to_img_feat)
            T['model'] = time.time() - tic
            scores_ = scores.data.cpu().numpy()
            predict = np.argmax(scores_, 1)
            acc += (predict == gt).sum()
            loss = crit(scores, gt_ann_ids)

            total_refexps += len(labels)

            loss_sum += loss.item() * len(labels)
            loss_evals += len(labels)

            # print
            ix0 = total_refexps
            ix1 = total_samples
            if verbose:
                print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, loss=%.4f, data_time=%.2f, '
                      'model_time=%.2f' % \
                      (split, ix0, ix1, acc * 100.0 / total_refexps, loss_sum / loss_evals, data_time, model_time))
            tic = time.time()
        assert total_refexps == total_samples

    return loss_sum / loss_evals, acc / total_samples
