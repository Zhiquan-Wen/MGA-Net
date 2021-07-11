from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from .Graph_modules import *
from .att_lang_encoder import *


class GraphMatching(nn.Module):
    def __init__(self, opt):
        super(GraphMatching, self).__init__()
        hidden_size = opt['rnn_hidden_size']
        num_dirs = 2 if opt['bidirectional'] > 0 else 1
        self.vocab_size = opt['vocab_size']
        cxt_dim = hidden_size * num_dirs

        # language rnn encoder
        self.rnn_encoder = RNNEncoder(
            vocab_size=opt['vocab_size'],
            word_embedding_size=opt['word_embedding_size'],
            word_vec_size=opt['word_vec_size'],
            hidden_size=opt['rnn_hidden_size'],
            bidirectional=opt['bidirectional'] > 0,
            input_dropout_p=opt['word_drop_out'],
            dropout_p=opt['rnn_drop_out'],
            n_layers=opt['rnn_num_layers'],
            rnn_type=opt['rnn_type'],
            variable_lengths=opt['variable_lengths'] > 0
        )

        self.weight_fc = nn.Linear(opt['word_vec_size'], 4)

        # phrase attention
        self.sub_attention = PhraseAttention(cxt_dim)
        self.ord_attention = PhraseAttention(cxt_dim)
        self.same_attention = PhraseAttention(cxt_dim)
        self.loc_attention = PhraseAttention(cxt_dim)

        self.sent_graph_matching = Sent2GraphMatching(opt)

    def forward(self, node_fc7, node_loc, loc_rel_same_feats, loc_rel_loc_feats, labels, num_objs,
                sent_to_img_feats):
        """
        :param node_fc7: (n, max_ann, vis_dim)
        :param node_loc: (n, max_ann, loc_dim)
        :param loc_rel_same_feats:(n, max_ann, max_ann, 517*2)
        :param loc_rel_loc_feats:(n, max_ann, max_ann, 512+5+5)
        :param labels:(n, seq_len)
        :param sent_to_image_ann:
        :param node_mask:(n, max_ann)
        :param edge_mask:(n, max_ann, max_ann)
        :return:
        """

        context, hidden, embedding = self.rnn_encoder(labels)

        embedded = embedding.mean(1)

        _, lang_vis_attn = self.sub_attention(context, embedding, labels)
        _, lang_ord_attn = self.ord_attention(context, embedding, labels)
        _, lang_same_attn = self.same_attention(context, embedding, labels)
        _, lang_loc_attn = self.loc_attention(context, embedding, labels)

        weights = F.softmax(self.weight_fc(embedded), 1)

        match_scores = self.sent_graph_matching(node_fc7, node_loc, loc_rel_same_feats, loc_rel_loc_feats,
                                                lang_vis_attn, lang_ord_attn, lang_same_attn, lang_loc_attn,
                                                weights, num_objs, sent_to_img_feats)
        return match_scores
