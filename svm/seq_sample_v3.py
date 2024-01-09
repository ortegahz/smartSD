import argparse
import glob
import logging
import os

import numpy as np

from utils.utils import set_logging, make_dirs, db_gen_v3, plot_db_v3, \
    find_anchor_idxes, seq_pick_process_future, update_svm_label_file, \
    DEBUG_ALARM_INDICATOR_VAL, LEN_SEQ_LOW

from demos.demo_fft import fft_wrapper


def update_and_plot(db, anchor_idx, seq_forward, seq_backward, subset, idx_save, args):
    db['state'][anchor_idx + 1:anchor_idx + LEN_SEQ_LOW + 1] = DEBUG_ALARM_INDICATOR_VAL / 8
    db['state'][anchor_idx] = DEBUG_ALARM_INDICATOR_VAL / 4
    seq_pick_forward, _, _ = seq_pick_process_future(seq_forward, anchor_idx)
    seq_pick_backward, _, _ = seq_pick_process_future(seq_backward, anchor_idx)
    seq_pick = np.concatenate((seq_pick_forward, seq_pick_backward), axis=0)
    # seq_pick = seq_pick_forward
    # seq_pick = fft_wrapper(seq_pick)
    update_svm_label_file(seq_pick, args.path_out, subset)
    plot_db_v3(db, 0.1, case=subset, dir_save=args.dir_plot_save, idx_save=idx_save)


def run(args):
    if args.dir_plot_save:
        make_dirs(args.dir_plot_save)
    if os.path.exists(args.path_out):
        os.remove(args.path_out)

    idx_save = 0
    for subset in args.subsets:
        dir_in_s = os.path.join(args.dir_in, subset)
        paths_src = glob.glob(os.path.join(dir_in_s, '*'))
        for path_src in paths_src:
            logging.info(path_src)
            db = db_gen_v3(path_src)
            seq_forward = np.array(db['forward']).astype(float)
            seq_backward = np.array(db['backward']).astype(float)
            anchor_idxes, anchor_idx_max = find_anchor_idxes(seq_backward)
            if len(anchor_idxes) < 1:
                continue
            for anchor_idx in anchor_idxes:
                update_and_plot(db, anchor_idx, seq_forward, seq_backward, subset, idx_save, args)
                idx_save += 1
            # update_and_plot(db, anchor_idx_max, seq_forward, seq_backward, subset, idx_save, args)
            # idx_save += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V3')
    parser.add_argument('--path_out', default='/home/manu/tmp/smartsd_v3')
    parser.add_argument('--subsets', default=['pos', 'neg'])
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/smartsd_plot_v3')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
