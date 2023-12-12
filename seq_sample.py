import argparse
import glob
import logging
import os

import numpy as np

from utils import set_logging, db_gen, plot_db, LEN_SEQ, make_dirs


def find_key_idx(seq, th_val=10, th_cnt=10, th_mean=1.):
    cnt = 0
    key_idx = -1
    for i, val in enumerate(seq):
        cnt = cnt + 1 if val > th_val else 0
        if cnt > th_cnt:
            key_idx = i
            break
    if key_idx == -1:
        return key_idx
    mean = 0.
    while key_idx + int(LEN_SEQ / 2) <= len(seq):
        idx_end = key_idx + int(LEN_SEQ / 2)
        mean = np.mean(np.absolute(seq[key_idx - 1:idx_end - 1] - seq[key_idx:idx_end]))
        if mean > th_mean or seq[idx_end] > 220:
            break
        key_idx += int(LEN_SEQ / 2)
    logging.info(f'mean --> {mean}')
    return key_idx


def run(args):
    if args.save_plot:
        make_dirs(args.dir_plot_save)
    if os.path.exists(args.path_out):
        os.remove(args.path_out)

    idx_save = 0
    for subset in args.subsets:
        dir_in_s = os.path.join(args.dir_in, subset)
        paths_src = glob.glob(os.path.join(dir_in_s, '*'))
        for path_src in paths_src:
            logging.info((path_src, idx_save))
            db = db_gen(path_src)
            status = np.array(db['Status'.lower()]).astype(float)
            idx_feat = 0
            feats = np.array(db[args.key_choose.lower()]).astype(float)
            # key_idx = np.nonzero(status)[0][0] if len(np.nonzero(status)[0]) > 0 else len(status) / 2
            key_idx = find_key_idx(feats)
            if subset == 'pos':
                assert key_idx > 0
            if subset == 'neg' and key_idx < 0:
                continue
            seq_len = LEN_SEQ
            idx_s = 0 if key_idx - seq_len * args.shift_rate_left < 0 else int(
                key_idx - seq_len * args.shift_rate_left)
            idx_e = len(status) if idx_s + seq_len > len(status) else int(idx_s + seq_len)
            db['Status'.lower()][int(key_idx)] = 100
            db['Status'.lower()][idx_s] = 255
            db['Status'.lower()][idx_e - 1] = 255
            seq_pick = feats[idx_s:idx_e]
            if len(seq_pick) < seq_len:
                pad = [seq_pick[-1]] * (seq_len - len(seq_pick))  # pad last seq val
                seq_pick = np.append(seq_pick, pad)
            # logging.info((idx_save, np.max(seq_pick)))
            with open(args.path_out, 'a') as f:
                label = '+1' if 'pos' in subset else '-1'
                f.write(label + ' ')
                for feat in seq_pick:
                    f.write(f'{idx_feat + 1}:{feat} ')
                    idx_feat += 1
            with open(args.path_out, 'a') as f:
                f.write('\n')
            if args.save_plot:
                plot_db(db, 0.1, subset, args.dir_plot_save, idx_save)
            idx_save += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR_TFS',
                        type=str)
    parser.add_argument('--path_out',
                        default='/home/manu/tmp/smartsd',
                        type=str)
    parser.add_argument('--subsets', default=['pos', 'neg'])
    parser.add_argument('--shift_rate_left', default=1 / 2)
    parser.add_argument('--key_choose', default='ADC_Forward')
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/smartsd_plot')
    parser.add_argument('--save_plot', default=True)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
