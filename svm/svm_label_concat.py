import argparse
import logging
import os

from utils.utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--paths_in', default=['/home/manu/tmp/smartsd_v0', '/home/manu/tmp/smartsd_v1', '/home/manu/tmp/smartsd_v2'])
    parser.add_argument('--paths_in', default=['/home/manu/tmp/smartsd_v4'])
    parser.add_argument('--path_out', default='/home/manu/tmp/smartsd')
    return parser.parse_args()


def run(args):
    if os.path.exists(args.path_out):
        os.remove(args.path_out)
    npos, nneg = 0, 0
    for path_in in args.paths_in:
        with open(path_in, 'r') as f:
            lines = f.readlines()
        with open(args.path_out, 'a') as f:
            for line in lines:
                label = line.strip().split()[0]
                if '+' in label:
                    npos += 1
                else:
                    nneg += 1
                f.write(line)
    logging.info(f'npos --> {npos} && nneg --> {nneg}')


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
