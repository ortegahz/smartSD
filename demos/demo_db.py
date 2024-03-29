import argparse
import logging

from core.smoke_detector import SmokeDetector
from utils.utils import set_logging, db_gen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in',
                        default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS/neg/Hot_SOTA_2022-08-09-3_1.xlsx')
    parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    parser.add_argument('--key_choose_forward', default='ADC_Forward')
    parser.add_argument('--key_choose_backward', default='ADC_Backward')
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()
    db = db_gen(args.path_in)
    for i in range(db['seq_len_max']):
        smoke_detector.update_db_v1(db, i, args.key_choose_forward,
                                    args.key_choose_backward, db_key=args.addrs_sensor[0])
        smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
        flag_save_plot = True if i == db['seq_len_max'] - 1 else False
        smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.1, save_plot=flag_save_plot, show=False)


if __name__ == '__main__':
    main()
