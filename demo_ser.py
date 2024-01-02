import argparse
import logging

from smoke_detector import SmokeDetector
from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addrs_sensor', default=[f'1_{1}', f'1_{5}'])
    parser.add_argument('--addrs_sensor', default=[f'1_{2}'])
    # parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    # parser.add_argument('--dev_ser', default='/dev/ttyUSB0')
    # parser.add_argument('--save_dir', default='/home/manu/tmp')
    parser.add_argument('--dir_root_libsvm', default=r'C:\Users\admin\Desktop\demo\libsvm')
    parser.add_argument('--dev_ser', default='COM3')
    parser.add_argument('--save_dir', default=r'C:\Users\EOS1\Desktop\data')
    # parser.add_argument('--sample_idxes', default=[])
    parser.add_argument('--sample_idxes', default=[0, 100])
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector(args.dev_ser)

    while True:
        smoke_detector.update_db_ser_multi_amp()
        smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
        smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.5)
        if len(args.sample_idxes) > 0 and \
                args.addrs_sensor[0] in smoke_detector.db.keys() and \
                smoke_detector.db[args.addrs_sensor[0]].get_seq_len() >= args.sample_idxes[-1]:
            smoke_detector.save_db(args.addrs_sensor, args.sample_idxes, save_dir=args.save_dir)
            break


if __name__ == '__main__':
    main()
