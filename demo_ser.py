import argparse
import glob
import logging
import os
import shutil

from smoke_detector import SmokeDetector
from utils import set_logging, SENSOR_ID


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()

    while True:
        smoke_detector.update_db_ser()
        # smoke_detector.infer_db(f'1_{SENSOR_ID}')
        smoke_detector.plot_db([f'1_{1}', f'1_{5}'], pause_time_s=0.5)


if __name__ == '__main__':
    main()
