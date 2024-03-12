# for small labyrinth data

import argparse
import glob
import logging
import os

import numpy as np

from utils.utils import set_logging, make_dirs, db_gen_v3, plot_db_v3, \
    find_anchor_idxes_v4, seq_pick_process_last, update_svm_label_file, \
    DEBUG_ALARM_INDICATOR_VAL, SMALL_LABYRINTH_SEQ_LEN


def update_and_plot(db, anchor_idx, seq_forward, seq_backward, subset, idx_save, args):
    db['state'][anchor_idx - SMALL_LABYRINTH_SEQ_LEN + 1:anchor_idx + 1] = DEBUG_ALARM_INDICATOR_VAL / 8
    db['state'][anchor_idx] = DEBUG_ALARM_INDICATOR_VAL / 4
    seq_pick_forward, _, _ = seq_pick_process_last(seq_forward, anchor_idx)
    seq_pick_backward, _, _ = seq_pick_process_last(seq_backward, anchor_idx)
    seq_pick = np.concatenate((seq_pick_forward, seq_pick_backward), axis=0)
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
            aug_scale_base = 8.0
            aug_scale = aug_scale_base if subset == 'neg' else aug_scale_base + 0.5
            anchor_idxes = find_anchor_idxes_v4(seq_backward, aug_scale=aug_scale)
            if len(anchor_idxes) < 1:
                continue
            for anchor_idx in anchor_idxes:
                update_and_plot(db, anchor_idx, seq_forward, seq_backward, subset, idx_save, args)
                idx_save += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V4')
    parser.add_argument('--path_out', default='/home/manu/tmp/smartsd_v4')
    parser.add_argument('--subsets', default=['pos', 'neg'])
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/smartsd_plot_v4')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
