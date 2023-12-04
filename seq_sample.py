import argparse
import glob
import logging
import os
import shutil
import sys

import numpy as np

from funs import set_logging, db_gen, plot_db, LEN_SEQ


def run(args):
    if os.path.exists(args.path_out):
        os.remove(args.path_out)

    for subset in args.subsets:
        dir_in_s = os.path.join(args.dir_in, subset)
        paths_src = glob.glob(os.path.join(dir_in_s, '*'))
        for path_src in paths_src:
            logging.info(path_src)
            db = db_gen(path_src)
            feats = np.array(db['Smoke_Forward'.lower()]).astype(float)
            status = np.array(db['Status'.lower()]).astype(float)
            # logging.info(status)
            # logging.info(np.nonzero(status))
            key_idx = np.nonzero(status)[0][0] if len(np.nonzero(status)[0]) > 0 else len(status) / 2
            # logging.info(key_idx)
            seq_len = LEN_SEQ
            idx_s = 0 if key_idx - seq_len * args.shift_rate_left < 0 else int(key_idx - seq_len * args.shift_rate_left)
            idx_e = len(feats) if idx_s + seq_len > len(feats) else int(idx_s + seq_len)
            seq_pick = feats[idx_s:idx_e]
            db['Status'.lower()][idx_s] = 255
            db['Status'.lower()][idx_e - 1] = 255
            # plot_db(db, 5)
            if len(seq_pick) < seq_len:
                pad = [0] * (seq_len - len(seq_pick))
                seq_pick = np.append(seq_pick, pad)
            logging.info(seq_pick)
            path_out = os.path.join(args.path_out)
            with open(path_out, 'a') as f:
                label = '+1' if 'pos' in subset else '-1'
                f.write(label + ' ')
                for i, feat in enumerate(seq_pick):
                    f.write(f'{i + 1}:{feat} ')
                f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR_TFS',
                        type=str)
    parser.add_argument('--path_out',
                        default='/home/manu/tmp/smartsd',
                        type=str)
    parser.add_argument('--subsets', default=['pos', 'neg'])
    parser.add_argument('--shift_rate_left', default=2 / 4)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
