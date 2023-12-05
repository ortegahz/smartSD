import argparse
import glob
import logging
import os

import numpy as np

from funs import set_logging, db_gen, plot_db, LEN_SEQ, make_dirs


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
            logging.info(path_src)
            db = db_gen(path_src)
            idx_feat = 0
            flag_valid = True
            status = np.array(db['Status'.lower()]).astype(float)
            key_idx = np.nonzero(status)[0][0] if len(np.nonzero(status)[0]) > 0 else len(status) / 2
            seq_len = LEN_SEQ
            idx_s = 0 if key_idx - seq_len * args.shift_rate_left < 0 else int(
                key_idx - seq_len * args.shift_rate_left)
            idx_e = len(status) if idx_s + seq_len > len(status) else int(idx_s + seq_len)
            db['Status'.lower()][idx_s] = 255
            db['Status'.lower()][idx_e - 1] = 255
            for key_choose in args.keys_choose:
                feats = np.array(db[key_choose.lower()]).astype(float)
                # logging.info(status)
                # logging.info(np.nonzero(status))
                # logging.info(key_idx)
                seq_pick = feats[idx_s:idx_e]
                if len(seq_pick) < seq_len:
                    pad = [0] * (seq_len - len(seq_pick))
                    seq_pick = np.append(seq_pick, pad)
                # logging.info(seq_pick)
                logging.info((idx_save, np.max(seq_pick)))
                # if 'pos' in subset and np.max(seq_pick) < 10:
                # if idx_save == 79:
                #     flag_valid = False
                #     continue
                if idx_save == 79 or idx_save == 44 or idx_save == 96:
                    seq_pick = feats[-seq_len:]
                    db['Status'.lower()][-seq_len] = 200
                    db['Status'.lower()][-1] = 200
                with open(args.path_out, 'a') as f:
                    label = '+1' if 'pos' in subset else '-1'
                    f.write(label + ' ')
                    for feat in seq_pick:
                        f.write(f'{idx_feat + 1}:{feat} ')
                        idx_feat += 1
            if args.save_plot:
                plot_db(db, 0.1, args.dir_plot_save, idx_save)
            idx_save += 1
            if flag_valid:
                with open(args.path_out, 'a') as f:
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
    parser.add_argument('--keys_choose', default=['ADC_Forward'])
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
