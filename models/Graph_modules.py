from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from .att_lang_encoder import *


class Sent2GraphMatching(nn.Module):
    def __init__(self, opt):
        super(Sent2GraphMatching, self).__init__()
        vis_dim = opt['vis_dim']
        loc_dim = opt['loc_dim']
        cxt_dim = opt['word_vec_size']
        jemb_dim = opt['jemb_dim']
        self.jemb_dim = jemb_dim
        jemb_drop_out = opt['jemb_drop_out']
        node_dif_loc_dim = loc_dim
        node_dim = 2 * jemb_dim
        self.setting = opt

        if opt['rounds'] > 0:
            self.same_GRU = nn.GRUCell(input_size=jemb_dim, hidden_size=jemb_dim)

            self.loc_GRU = nn.GRUCell(input_size=jemb_dim, hidden_size=jemb_dim)

        # feature encoding
        self.vis_o_emb = nn.Sequential(
            nn.Linear(vis_dim, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(jemb_drop_out),
            nn.Linear(jemb_dim, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(jemb_dim)
        )

        self.loc_o_emb = nn.Sequential(
            nn.Linear(loc_dim, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(jemb_drop_out),
            nn.Linear(jemb_dim, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(jemb_dim)
        )

        self.vis_same_emb = nn.Sequential(
            nn.Linear((vis_dim + loc_dim) * 2, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(jemb_drop_out),
            nn.Linear(jemb_dim, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(jemb_dim),
        )

        self.loc_rel_emb = nn.Sequential(
            nn.Linear((vis_dim + loc_dim + node_dif_loc_dim), jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(jemb_drop_out),
            nn.Linear(jemb_dim, jemb_dim),
            nn.BatchNorm1d(jemb_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(jemb_dim),
        )

        # encode the language
        self.lang_same_dim = nn.Linear(cxt_dim, jemb_dim)
        self.lang_loc_dim = nn.Linear(cxt_dim, jemb_dim)
        self.lang_vis_dim = nn.Linear(cxt_dim, jemb_dim)
        self.lang_ord_dim = nn.Linear(cxt_dim, jemb_dim)

        # encode the node or edge
        self.node_vis_dim = nn.Linear(jemb_dim, jemb_dim)
        self.node_ord_dim = nn.Linear(node_dim, jemb_dim)
        self.edge_same_dim = nn.Linear(jemb_dim, jemb_dim)
        self.edge_loc_dim = nn.Linear(jemb_dim, jemb_dim)

        # attention
        self.attn_vis = nn.Linear(jemb_dim, 1)
        self.attn_ord = nn.Linear(jemb_dim, 1)
        self.attn_same = nn.Linear(jemb_dim, 1)
        self.attn_loc = nn.Linear(jemb_dim, 1)

        # matching lang
        self.match_lang_same = nn.Linear(cxt_dim, jemb_dim)
        self.match_lang_ord = nn.Linear(cxt_dim, jemb_dim)
        self.match_lang_loc = nn.Linear(cxt_dim, jemb_dim)
        self.match_lang_vis = nn.Linear(cxt_dim, jemb_dim)

        # matching others
        self.match_vis = nn.Linear(jemb_dim, jemb_dim)
        self.match_ord = nn.Linear(node_dim, jemb_dim)
        self.match_same = nn.Linear(jemb_dim, jemb_dim)
        self.match_loc = nn.Linear(jemb_dim, jemb_dim)

    def forward(self, vis_input, loc_input, same_input, loc_rel_input, lang_vis_input, lang_ord_input, lang_same_input,
                lang_loc_input, weights, num_objs, sent_to_img_feats):
        """
            :param sent_to_img_feats: index of the query corresponding to the image (batch_lang, )
            :param num_objs: num of objects of each image in one batch, (batch, )
            :param vis_input: (batch*num_obj, vis_dim)
            :param loc_input: (batch*max_ann*(max_ann-1), loc_dim)
            :param same_input: (batch*max_ann*(max_ann-1), (vis_dim + loc_dim)*2)
            :param loc_rel_input: (batch*max_ann, (vis_dim + loc_dim + node_dif_dim))
            :param lang_same_input: (batch_lang, jemb_dim)
            :param lang_loc_input: (batch_lang, jemb_dim)
            :param lang_ord_input: (batch_lang, jemb_dim)
            :param lang_vis_input: (batch_lang, jemb_dim)
            :param weights: (batch_lang, 4)
            :return:
            """

        vis_input_view = vis_input  # total objs in batch images, 512

        loc_input_view = loc_input  # (total objs in batch images, 5)

        same_input_view = same_input  # (one batch sum(num_objs*(num_objs-1)), 1029*2)

        loc_rel_input_view = loc_rel_input  # (one batch sum(num_objs*(num_objs-1)), 1034)

        # encode the input feature
        loc_input_view = self.loc_o_emb(loc_input_view)  # (total objs in batch images, 512)
        SAME = self.vis_same_emb(same_input_view)  # (one batch sum(num_objs*(num_objs-1)), 512)
        LOC = self.loc_rel_emb(loc_rel_input_view)  # (one batch sum(num_objs*(num_objs-1)), 512)
        vis_input_view = self.vis_o_emb(vis_input_view)  # (total objs in batch images, 512)
        SUB = vis_input_view  # (total objs in batch images, 512)
        ORD = torch.cat((vis_input_view, loc_input_view), 1)  # (total objs in batch images, 1024)

        # encode the lang
        lang_vis = self.lang_vis_dim(lang_vis_input)  # (num refs in one batch images, 512)
        lang_loc = self.lang_loc_dim(lang_loc_input)  # (num refs in one batch images, 512)
        lang_same = self.lang_same_dim(lang_same_input)  # (num refs in one batch images, 512)
        lang_ord = self.lang_ord_dim(lang_ord_input)  # (num refs in one batch images, 512)

        # encode node or edge
        SUB_node_emb = self.node_vis_dim(SUB)  # (total objs in batch images, 512)
        ORD_node_emb = self.node_ord_dim(ORD)  # (total objs in batch images, 512)
        SAME_edge_emb = self.edge_same_dim(SAME)  # (one batch sum(num_objs*(num_objs-1)), 512)
        LOC_edge_emb = self.edge_loc_dim(LOC)  # (one batch sum(num_objs*(num_objs-1)), 512)

        batch, s_dim = lang_vis_input.size()  

        match_scores = Variable(torch.ones(batch, max(num_objs)).cuda() * -230.)

        # encode Sents
        Sents = self.match_lang_vis(lang_vis_input)  # (num refs in one batch images, 512)
        Sents1 = self.match_lang_ord(lang_ord_input)  # (num refs in one batch images, 512)
        Sents2 = self.match_lang_same(lang_same_input)  # (num refs in one batch images, 512)
        Sents3 = self.match_lang_loc(lang_loc_input)  # (num refs in one batch images, 512)

        node_point = 0
        edge_point = 0
        now_id = 0
        for i in range(batch):
            ######################################################################################
            feat_id = sent_to_img_feats[i]  # sent_to_img_feats denotes each ref corresponds to which images
            if now_id == feat_id:
                node_point = node_point
                edge_point = edge_point
            else:
                node_point = node_point + num_objs[feat_id - 1]
                edge_point = edge_point + num_objs[feat_id - 1] * (num_objs[feat_id - 1] - 1)
                now_id = feat_id

            # compute the index of the features 
            #######################################################################################
            max_ann = num_objs[feat_id]

            sub = SUB[node_point: node_point + num_objs[feat_id]]
            ordi = ORD[node_point: node_point + num_objs[feat_id]]
            same = SAME[edge_point:edge_point + (num_objs[feat_id] * (num_objs[feat_id] - 1))]
            loc = LOC[edge_point:edge_point + (num_objs[feat_id] * (num_objs[feat_id] - 1))]

            same_edge_emb = SAME_edge_emb[edge_point: edge_point + (num_objs[feat_id] * (num_objs[feat_id] - 1))]
            loc_edge_emb = LOC_edge_emb[edge_point: edge_point + (num_objs[feat_id] * (num_objs[feat_id] - 1))]
            sub_node_emb = SUB_node_emb[node_point: node_point + num_objs[feat_id]]
            ord_node_emb = ORD_node_emb[node_point: node_point + num_objs[feat_id]]

            # VIS node attention
            lang_vis_ = lang_vis[i]  # fetch one encoded sentence
            lang_vis_expand = lang_vis_.unsqueeze(0).expand(max_ann, self.jemb_dim)
            lang_vis_attn = self.attn_vis(F.tanh(lang_vis_expand + sub_node_emb)).squeeze(1)
            lang_vis_attn = F.softmax(lang_vis_attn, 0)
            lang_vis_attn = lang_vis_attn.unsqueeze(1).expand(max_ann, sub.size()[1])

            sub = sub * lang_vis_attn

            # ORD node attention
            lang_ord_ = lang_ord[i]
            lang_ord_ = lang_ord_.unsqueeze(0).expand(max_ann, self.jemb_dim)
            lang_ord_attn = self.attn_ord(F.tanh(ord_node_emb + lang_ord_)).squeeze(1)
            lang_ord_attn = F.softmax(lang_ord_attn, 0)
            lang_ord_attn = lang_ord_attn.unsqueeze(1).expand(max_ann, ordi.size()[1])

            ordi = ordi * lang_ord_attn

            # SAME edge attention
            lang_same_ = lang_same[i]
            lang_same_ = lang_same_.unsqueeze(0).expand(max_ann*(max_ann - 1), self.jemb_dim)
            lang_same_attn = self.attn_same(F.tanh(lang_same_ + same_edge_emb)).squeeze(1)

            # LOC edge attention
            lang_loc_ = lang_loc[i]
            lang_loc_ = lang_loc_.unsqueeze(0).expand(max_ann*(max_ann - 1), self.jemb_dim)
            lang_loc_attn = self.attn_loc(F.tanh(lang_loc_ + loc_edge_emb)).squeeze(1)

            if self.setting['rounds'] > 0:
                # SAME edge
                lang_same_attn = F.softmax(lang_same_attn.view(max_ann, max_ann-1), 1).view(-1)

                A_same_attn = lang_same_attn.view(max_ann, max_ann-1)
                A_same = A_same_attn.new_zeros(size=(max_ann, max_ann))
                mask_same = torch.eye(n=max_ann, device=A_same.device) < 1
                A_same.masked_scatter_(mask_same, A_same_attn)

                lang_same_attn = lang_same_attn.unsqueeze(1).expand(max_ann*(max_ann-1), same.size()[-1])
                same = same * lang_same_attn
                same = same.view(max_ann, max_ann-1, -1).sum(1)

                H_same = same  # (max_ann, jemb_dim)   

                # LOC edge
                lang_loc_attn =  F.softmax(lang_loc_attn.view(max_ann, max_ann-1), 1).view(-1)

                A_loc_attn = lang_loc_attn.view(max_ann, max_ann-1)
                A_loc = A_loc_attn.new_zeros(size=(max_ann, max_ann))
                mask_loc = torch.eye(n=max_ann, device=A_loc.device) < 1
                A_loc.masked_scatter_(mask_loc, A_loc_attn)

                lang_loc_attn = lang_loc_attn.unsqueeze(1).expand(max_ann*(max_ann-1), loc.size()[-1])
                loc = loc * lang_loc_attn
                loc = loc.view(max_ann, max_ann-1, -1).sum(1)
                H_loc = loc

                for _ in range(self.setting['rounds']):
                    a_same = F.tanh(torch.mm(A_same, H_same))   # H_same = same
                    H_same = self.same_GRU(a_same, H_same)

                    a_loc = F.tanh(torch.mm(A_loc, H_loc))  # H_loc = loc
                    H_loc = self.loc_GRU(a_loc, H_loc)

            else:
                # SAME edge
                lang_same_attn = lang_same_attn.view(max_ann, max_ann-1)
                lang_same_attn = F.softmax(lang_same_attn, 1)
                lang_same_attn = lang_same_attn.view(-1)
                lang_same_attn = lang_same_attn.unsqueeze(1).expand(max_ann*(max_ann - 1), same.size()[-1])

                same = same * lang_same_attn
                H_same = same.view(max_ann, max_ann-1, -1).sum(1)

                # LOC edge
                lang_loc_attn = lang_loc_attn.view(max_ann, max_ann-1)
                lang_loc_attn = F.softmax(lang_loc_attn, 1)
                lang_loc_attn = lang_loc_attn.view(-1)
                lang_loc_attn = lang_loc_attn.unsqueeze(1).expand(max_ann*(max_ann - 1), loc.size()[-1])

                loc = loc * lang_loc_attn
                H_loc = loc.view(max_ann, max_ann-1, -1).sum(1)

            # match scores
            weight = weights[i]  # (4)
            sent = Sents[i].unsqueeze(0)
            sent1 = Sents1[i].unsqueeze(0)
            sent2 = Sents2[i].unsqueeze(0)
            sent3 = Sents3[i].unsqueeze(0)

            sent = F.tanh(sent)
            sent1 = F.tanh(sent1)
            sent2 = F.tanh(sent2)
            sent3 = F.tanh(sent3)

            obj = F.tanh(self.match_vis(sub))
            obj1 = F.tanh(self.match_ord(ordi))
            obj2 = F.tanh(self.match_same(H_same))
            obj3 = F.tanh(self.match_loc(H_loc))

            score = torch.mm(sent, obj.transpose(0, 1))
            score1 = torch.mm(sent1, obj1.transpose(0, 1))
            score2 = torch.mm(sent2, obj2.transpose(0, 1))
            score3 = torch.mm(sent3, obj3.transpose(0, 1))

            match_scores[i, 0:max_ann] = weight[0]*score + weight[1]*score1 + weight[2]*score2 + weight[3]*score3

        return match_scores


