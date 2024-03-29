import argparse
import glob
import logging
import os
import sys

from core.smoke_detector import SmokeDetector
from utils.utils import set_logging, db_gen_v3, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/data_202403/小迷宫烟枪')
    # parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/data_202403/小迷宫烟枪弱')
    # parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/data_202403/小迷宫加湿器')
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/data_202403')
    parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/infer_results')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()
    dirs_case = glob.glob(os.path.join(args.dir_in, '*'))
    for dir_case in dirs_case:
        case_name = os.path.basename(dir_case)
        dir_plot_save_real = args.dir_plot_save + f'_{case_name}'
        make_dirs(dir_plot_save_real)
        paths_in = glob.glob(os.path.join(dir_case, '*'))
        for j, path_in in enumerate(paths_in):
            logging.info((j, len(paths_in), path_in))
            db = db_gen_v3(path_in)
            for i in range(db['seq_len']):
                smoke_detector.update_db_v3(db, i, db_key=args.addrs_sensor[0])
                # smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
                smoke_detector.infer_db_small_labyrinth(args.addrs_sensor, args.dir_root_libsvm)
                flag_save_plot = True if i == db['seq_len'] - 1 else False
                title_info = os.path.basename(path_in).split('.')[0]
                path_plot_save = os.path.join(dir_plot_save_real, title_info)
                cmd_exit = smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.001, save_plot=flag_save_plot,
                                                  path_save=path_plot_save, title_info=title_info)
                if cmd_exit:
                    sys.exit(0)
            smoke_detector.clear_db()


if __name__ == '__main__':
    main()
