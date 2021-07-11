from torch.utils.data import DataLoader
from Datasets.Dataset import CLEVRDataset
import torch


def collate_fn(batch):
    img_feats, refexps, lfeats, same_feats, loc_feats, answer, num_obj, num_refexps = zip(*batch)
    img_feats_ = torch.cat(img_feats, 0)
    refexps_ = torch.cat(refexps, 0)
    lfeats_ = torch.cat(lfeats, 0)
    loc_feats_ = torch.cat(loc_feats, 0)
    same_feats_ = torch.cat(same_feats, 0)
    answer_ = torch.cat(answer, 0)
    num_obj_ = list(num_obj)
    num_refexps_ = list(num_refexps)
    max_len_sent = (refexps_ != 0).sum(1).max().data
    refexps_ = refexps_[:, :max_len_sent]
    sent_to_img_feats = []
    for i in range(len(num_refexps_)):
        sent_to_img_feats += [i] * num_refexps_[i]

    return img_feats_, refexps_, lfeats_, loc_feats_, same_feats_, answer_, num_obj_, num_refexps_, sent_to_img_feats


class CLEVRDataLoader(object):
    def __init__(self, train_vis_feat_path, train_refexps_path, train_bounding_box_path, train_answer_obj_path,
                 test_vis_feat_path, test_refexps_path, test_bounding_box_path, test_answer_obj_path,
                 vocab_path, batch_size, n_thread):
        super(CLEVRDataLoader, self).__init__()
        self.batch_size = batch_size
        self.n_thread = n_thread
        train_dataset = CLEVRDataset(train_vis_feat_path, train_refexps_path, train_bounding_box_path,
                                     train_answer_obj_path, vocab_path)
        test_dataset = CLEVRDataset(test_vis_feat_path, test_refexps_path, test_bounding_box_path,
                                    test_answer_obj_path, vocab_path)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=n_thread, collate_fn=collate_fn)

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                                       num_workers=n_thread, collate_fn=collate_fn)

    def get_dataloader(self):
        return self.train_loader, self.test_loader
