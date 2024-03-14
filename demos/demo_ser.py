import argparse
import logging

from core.smoke_detector import SmokeDetector
from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baud_rate', default=115200)
    # parser.add_argument('--addrs_sensor', default=[f'1_{1}', f'1_{5}'])
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    # parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    # parser.add_argument('--dev_ser', default='/dev/ttyUSB0')
    # parser.add_argument('--save_dir', default='/home/manu/tmp')
    parser.add_argument('--dev_ser', default='COM3')
    # parser.add_argument('--dir_root_libsvm', default=r'C:\Users\admin\Desktop\demo\libsvm')
    # parser.add_argument('--save_dir', default=r'C:\Users\EOS1\Desktop\data')
    parser.add_argument('--dir_root_libsvm', default=r'C:\Users\zxthz\Desktop\demo\libsvm')
    parser.add_argument('--save_dir', default=r'C:\Users\zxthz\Desktop\data')
    parser.add_argument('--sample_idxes', default=[])
    # parser.add_argument('--sample_idxes', default=[0, 100])
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector(args.dev_ser, baud_rate=args.baud_rate)

    while True:
        # if smoke_detector.update_db_ser_particles() < 0:
        if smoke_detector.update_db_ser_multi_amp() < 0:
            continue
        smoke_detector.infer_db_small_labyrinth(args.addrs_sensor, args.dir_root_libsvm)
        # smoke_detector.infer_db_particles(args.addrs_sensor)
        cmd_sample = smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.5, show=True)
        if args.addrs_sensor[0] in smoke_detector.db.keys() and cmd_sample:
            smoke_detector.save_db(args.addrs_sensor, args.sample_idxes, save_dir=args.save_dir)
            break


if __name__ == '__main__':
    main()
