from pprint import pprint
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='clevr', help='name of dataset')

    # Dataset
    parser.add_argument('--train_vis_feat_path', type=str, default='./data/resnet101_train_gt.h5', help='vis obj feature path')
    parser.add_argument('--train_refexps_path', type=str, default='./data/train_refexp_detect_idx.h5', help='refexps idx path')
    parser.add_argument('--train_bounding_box_path', type=str, default='./data/obj_bbox_train_with_sequential.json', help='bounding box path')
    parser.add_argument('--train_answer_obj_path', type=str, default='./data/train_refexp_answer_detect.json', help='answer_obj path')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.json', help='vocab path')
    parser.add_argument('--test_vis_feat_path', type=str, default='./data/resnet101_val_gt.h5',   help='vis obj feature path')
    parser.add_argument('--test_refexps_path', type=str, default='./data/test_refexp_detect_idx.h5', help='refexps idx path')
    parser.add_argument('--test_bounding_box_path', type=str, default='./data/obj_bbox_val_with_sequential.json', help='bounding box path')
    parser.add_argument('--test_answer_obj_path', type=str, default='./data/test_refexp_answer_detect.json', help='answer_obj path')

    # Language Encoder Setting
    parser.add_argument('--word_embedding_size', type=int, default=512, help='the encoding size of each token')
    parser.add_argument('--word_vec_size', type=int, default=512, help='further non-linear of word embedding')
    parser.add_argument('--word_drop_out', type=float, default=0, help='word drop out after embedding')
    parser.add_argument('--bidirectional', type=int, default=1, help='bi-rnn')
    parser.add_argument('--rnn_hidden_size', type=int, default=512, help='hidden size of LSTM')
    parser.add_argument('--rnn_type', type=str, default='lstm', help='rnn, gru or lstm')
    parser.add_argument('--rnn_drop_out', type=float, default=0, help='dropout between stacked rnn layers')
    parser.add_argument('--rnn_num_layers', type=int, default=2, help='number of layers in lang_encoder')
    parser.add_argument('--variable_lengths', type=int, default=1, help='use variable length to encode')
    # Number of rounds in GGNN
    parser.add_argument('--rounds', type=int, default=0, help='number of rounds in GGNN, 0 denotes without GGNN')
    # Joint Embedding setting
    parser.add_argument('--jemb_drop_out', type=float, default=0, help='dropout in the joint embedding')
    parser.add_argument('--jemb_dim', type=int, default=512, help='joint embedding layer dimension')
    parser.add_argument('--match_dim', type=int, default=512, help='embedding size for sent to graph matching')

    # Optimization: General
    parser.add_argument('--max_epoches', type=int, default=50, help='max number of iterations to run')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size in number of images per batch')
    parser.add_argument('--grad_clip', type=float, default=0.3, help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    # Evaluation/Checkpointing
    parser.add_argument('--losses_log_every', type=int, default=25,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--resume', type=str, default='', help='checkpoint path')
    parser.add_argument('--saved_des', type=str, default='245_experiments', help='description of saved path')
    # misc
    parser.add_argument('--seed', type=int, default=7, help='random number generator seed to use')
    parser.add_argument('--gpuid', type=int, default=1, help='which gpu to use, -1 = use CPU')
    # parse 
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args


if __name__ == '__main__':
    opt = parse_opt()
    print('opt[\'id\'] is ', opt['id'])
