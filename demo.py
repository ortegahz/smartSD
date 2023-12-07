import argparse
import logging

from smoke_detector import SmokeDetector
from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR_TFS/neg',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()
    logging.info(smoke_detector)


if __name__ == '__main__':
    main()
