import argparse
import glob
import logging
import os
import shutil

from smoke_detector import SmokeDetector
from utils import set_logging, SENSOR_ID


def paths_copy(paths, dir_out):
    for path in paths:
        file_name = os.path.basename(path)
        path_out = os.path.join(dir_out, file_name)
        shutil.copy(path, path_out)


def paths_move(paths, dir_out):
    for path in paths:
        file_name = os.path.basename(path)
        path_out = os.path.join(dir_out, file_name)
        shutil.move(path, path_out)


def cmp_path(path):
    file_name = os.path.basename(path).split('.')[0]
    group_id, sensor_id, date, time = file_name.split('-')
    hour, minute, second = time.split(':')
    date, hour, minute, second = int(date), int(hour), int(minute), int(second)
    second_total = hour * 60 * 60 + minute * 60 + second
    return date, second_total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/home/manu/nfs/data/smartsd_cur',
                        type=str)
    parser.add_argument('--dir_out',
                        default='/home/manu/nfs/data/smartsd_bak',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()

    os.system(f'cp {args.dir_out}/* {args.dir_in} -rvf')

    while True:
        paths_txt = glob.glob(os.path.join(args.dir_in, '*.txt'))
        paths_txt_sorted = sorted(paths_txt, key=cmp_path)
        smoke_detector.update_db(paths_txt_sorted)
        paths_txt_sorted_pick = paths_txt_sorted[:-10]
        paths_move(paths_txt_sorted_pick, args.dir_out)
        # paths_copy([paths_txt_sorted[-1]], args.dir_out)
        smoke_detector.print_db()
        smoke_detector.plot_db(f'1_{SENSOR_ID}', pause_time_s=2048)
        break


if __name__ == '__main__':
    main()
