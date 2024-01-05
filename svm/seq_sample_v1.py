import argparse
import glob
import logging
import os

import numpy as np

from utils.utils import set_logging, db_gen_v1, plot_db_v1, make_dirs, find_key_idx, seq_pick_process, update_svm_label_file
from demos.demo_fft import fft_wrapper

def run(args):
    # if args.save_plot:
    #     make_dirs(args.dir_plot_save)
    if os.path.exists(args.path_out):
        os.remove(args.path_out)

    num_exist_files = len(glob.glob(os.path.join(args.dir_plot_save, '*')))
    idx_save = num_exist_files

    for subset in args.subsets:
        dir_in_s = os.path.join(args.dir_in, subset)
        paths_src = glob.glob(os.path.join(dir_in_s, '*'))
        for path_src in paths_src:
            logging.info((path_src, idx_save))
            for db in db_gen_v1(path_src):
                feats = np.array(db[args.key_choose.lower()]).astype(float)
                key_idx = find_key_idx(feats)
                # if subset == 'pos':
                #     assert key_idx > 0
                # if subset == 'neg' and key_idx < 0:
                #     continue
                if key_idx < 0:
                    continue
                seq_pick, _, _ = seq_pick_process(feats, key_idx, db=db, key_debug='addr')
                seq_pick_fft = fft_wrapper(seq_pick)
                update_svm_label_file(seq_pick, args.path_out, subset)
                if args.save_plot:
                    plot_db_v1(db, seq_pick_fft, 0.1, subset, args.dir_plot_save, idx_save)
                logging.info(idx_save)
                idx_save += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V1',
                        type=str)
    parser.add_argument('--path_out',
                        default='/home/manu/tmp/smartsd_v1',
                        type=str)
    parser.add_argument('--subsets', default=['pos', 'neg'])
    parser.add_argument('--key_choose', default='forward')
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
