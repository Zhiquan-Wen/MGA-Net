import json
import h5py
import numpy as np
import argparse
from preprocess import tokenize, build_vocab, encode, decode

'''
处理refexp的数据，首先构建词汇表，然后根据词汇表将每个refexp转换为词汇表的idx，将词汇表存入一个json文件 {word: idx}
所有refexp存入h5py文件里面
'''

parser = argparse.ArgumentParser()
# /home/wenzhiquan/Datasets/clevr_ref+_1.0/refexps/refexp_train/val.json
parser.add_argument('--input_refexps_json', required=True)  # data format : {Image name:{refexps_id: refexps}}
# /home/wenzhiquan/Datasets/clevr_ref+_1.0/refexps/refexp_train/val_idx.h5df
parser.add_argument('--output_refexps_h5df', required=True)
parser.add_argument('--output_vocab_json', required=True)

def main(args):

    data = json.load(open(args.input_refexps_json, 'r'))
    max_length = 0
    all_refexps = []
    for keys in data:
        for ref_id in data[keys]:
            all_refexps.append(data[keys][ref_id])

    for r in all_refexps:
        t = tokenize(
            r,
            punct_to_keep=[',', ';'],
            punct_to_remove=['?', '.']
        )
        if len(t) > max_length:
            max_length = len(t)

    refexp_token_to_idx = build_vocab(
        all_refexps,
        punct_to_keep=[',', ';'],
        punct_to_remove=['?', '.']
    )

    with open(args.output_vocab_json, 'w') as f:
        json.dump(refexp_token_to_idx, f)

    with h5py.File(args.output_refexps_h5df, 'w') as f:
        for keys in data:
            one_image_refexps = []
            # img_name = keys.split('.')[0]
            one_image_refexps_to_idx = []
            img_all_refexps = data[keys]

            for ref_id in img_all_refexps:
                # refexp = img_all_refexps[ref_id]
                # one_image_refexps.append(refexp)
                refexp = img_all_refexps[ref_id]
                one_image_refexps.append(refexp)

            for refexps in one_image_refexps:
                tokens = tokenize(refexps, punct_to_remove=['?', '.'], punct_to_keep=[';', ','])
                refexps_idx = encode(tokens, refexp_token_to_idx)
                one_image_refexps_to_idx.append(refexps_idx)

            for refexp_ in one_image_refexps_to_idx:
                num_null = max_length - len(refexp_)
                if num_null > 0:
                    refexp_ += [refexp_token_to_idx['<NULL>']]*num_null

            one_image_refexps_to_idx_numpy = np.asarray(one_image_refexps_to_idx, dtype=np.int32)

            f.create_dataset(keys, data=one_image_refexps_to_idx_numpy)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




