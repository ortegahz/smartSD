import argparse
import glob
import logging
import os
import shutil

from smoke_detector import SmokeDetector
from utils import set_logging, SENSOR_ID


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addrs_sensor', default=[f'1_{1}', f'1_{5}'])
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()

    while True:
        smoke_detector.update_db_ser()
        smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
        smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.5)


if __name__ == '__main__':
    main()
