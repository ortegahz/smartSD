import argparse
import glob
import logging
import os

from smoke_detector import SmokeDetector
from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/home/manu/nfs/data/smartsd_a',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()

    paths_txt = glob.glob(os.path.join(args.dir_in, '*.txt'))
    smoke_detector.update_db(paths_txt)
    smoke_detector.show_db()


if __name__ == '__main__':
    main()
