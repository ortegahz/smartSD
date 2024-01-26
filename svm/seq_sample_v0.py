import argparse
import glob
import logging
import os

import numpy as np

from demos.demo_fft import fft_wrapper
from utils.utils import set_logging, db_gen, plot_db, make_dirs, \
    find_anchor_idx_up, update_svm_label_file, \
    LEN_SEQ, DEBUG_ALARM_INDICATOR_VAL


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
            if 'adc_forward' not in db.keys() or 'adc_backward' not in db.keys():
                continue
            seq_forward = np.array(db['adc_forward']).astype(float)
            seq_backward = np.array(db['adc_backward']).astype(float)
            anchor_idx = find_anchor_idx_up(seq_backward)
            if anchor_idx < 0:
                continue
            db['status'][anchor_idx - LEN_SEQ + 1] = DEBUG_ALARM_INDICATOR_VAL
            db['status'][anchor_idx] = DEBUG_ALARM_INDICATOR_VAL
            seq_forward_pick = seq_forward[anchor_idx - LEN_SEQ + 1:anchor_idx + 1]
            seq_backward_pick = seq_backward[anchor_idx - LEN_SEQ + 1:anchor_idx + 1]
            seq_pick = np.concatenate((seq_forward_pick, seq_backward_pick), axis=0)
            update_svm_label_file(seq_pick, args.path_out, subset)
            if args.save_plot:
                plot_db(db, None, 0.1, subset, args.dir_plot_save, idx_save)
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
