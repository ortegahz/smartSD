import argparse
import logging

from smoke_detector import SmokeDetector
from utils import set_logging, db_gen_v3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in',
                        default='/media/manu/data/docs/smokes/data_202312/线香/2.txt')
    parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()
    db = db_gen_v3(args.path_in)
    for i in range(db['seq_len']):
        smoke_detector.update_db_v3(db, i, db_key=args.addrs_sensor[0])
        smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
        flag_save_plot = True if i == db['seq_len'] - 1 else False
        smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.1, save_plot=flag_save_plot)


if __name__ == '__main__':
    main()
