import argparse
import glob
import logging
import os

import numpy as np

from demos.demo_fft import fft_wrapper
from utils.utils import set_logging, db_gen, plot_db, make_dirs, find_key_idx, seq_pick_process, update_svm_label_file


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
            feats = np.array(db[args.key_choose.lower()]).astype(float)
            # key_idx = np.nonzero(status)[0][0] if len(np.nonzero(status)[0]) > 0 else len(status) / 2
            key_idx = find_key_idx(feats)
            # if subset == 'pos':
            #     assert key_idx > 0
            # if subset == 'neg' and key_idx < 0:
            #     continue
            if key_idx < 0:
                continue
            seq_pick, _, _ = seq_pick_process(feats, key_idx, db=db)
            seq_pick_fft = fft_wrapper(seq_pick)
            # logging.info((idx_save, np.max(seq_pick)))
            update_svm_label_file(seq_pick, args.path_out, subset)
            if args.save_plot:
                plot_db(db, seq_pick_fft, 0.1, subset, args.dir_plot_save, idx_save)
            idx_save += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS',
                        type=str)
    parser.add_argument('--path_out',
                        default='/home/manu/tmp/smartsd_v0',
                        type=str)
    parser.add_argument('--subsets', default=['pos', 'neg'])
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
