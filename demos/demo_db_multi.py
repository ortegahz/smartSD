import argparse
import glob
import logging
import os

from core.smoke_detector import SmokeDetector
from utils.utils import set_logging, db_gen, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS/pos')
    parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    parser.add_argument('--key_choose_forward', default='ADC_Forward')
    parser.add_argument('--key_choose_backward', default='ADC_Backward')
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/infer_results')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()
    make_dirs(args.dir_plot_save)
    paths_in = glob.glob(os.path.join(args.dir_in, '*'))
    for j, path_in in enumerate(paths_in):
        logging.info((j, len(paths_in), path_in))
        db = db_gen(path_in)
        for i in range(db['seq_len_max']):
            smoke_detector.update_db_v1(db, i, args.key_choose_forward,
                                        args.key_choose_backward, db_key=args.addrs_sensor[0])
            smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
            flag_save_plot = True if i == db['seq_len_max'] - 1 else False
            title_info = os.path.basename(path_in).split('.')[0]
            path_plot_save = os.path.join(args.dir_plot_save, title_info)
            smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.001, save_plot=flag_save_plot,
                                   path_save=path_plot_save, title_info=title_info)
        smoke_detector.clear_db()


if __name__ == '__main__':
    main()
