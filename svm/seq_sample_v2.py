import argparse
import glob
import logging
import os

import numpy as np

from utils.utils import set_logging, db_gen_v2, plot_db_v2, update_svm_label_file, find_anchor_idxes_up, LEN_SEQ, \
    DEBUG_ALARM_INDICATOR_VAL


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
            db = db_gen_v2(path_src)
            if 'forward' not in db.keys() or 'backward' not in db.keys():
                continue
            seq_forward = np.array(db['forward']).astype(float)
            seq_backward = np.array(db['backward']).astype(float)
            NUM_OF_NEG = 1
            anchor_idxes = find_anchor_idxes_up(seq_backward)
            if len(anchor_idxes) < 1:
                continue
            if 'pos' in subset or len(anchor_idxes) < NUM_OF_NEG:
                anchor_idxes_pick = anchor_idxes[:1]
            else:
                anchor_idxes_pick = anchor_idxes[:NUM_OF_NEG]
            for anchor_idx in anchor_idxes_pick:
                db['status'][anchor_idx - LEN_SEQ + 1] = DEBUG_ALARM_INDICATOR_VAL
                db['status'][anchor_idx] = DEBUG_ALARM_INDICATOR_VAL
                seq_forward_pick = seq_forward[anchor_idx - LEN_SEQ + 1:anchor_idx + 1]
                seq_backward_pick = seq_backward[anchor_idx - LEN_SEQ + 1:anchor_idx + 1]
                seq_pick = np.concatenate((seq_forward_pick, seq_backward_pick), axis=0)
                update_svm_label_file(seq_pick, args.path_out, subset)
                if args.save_plot:
                    plot_db_v2(db, None, 0.1, subset, args.dir_plot_save, idx_save)
                idx_save += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V2')
    parser.add_argument('--path_out', default='/home/manu/tmp/smartsd_v2')
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
