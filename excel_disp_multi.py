import argparse
import glob
import logging
import os

from utils import db_gen_v1, plot_db_v1, set_logging


def run(args):
    paths_in = glob.glob(os.path.join(args.dir_in, '*'))

    for i, path_in in enumerate(paths_in):
        logging.info((path_in, i, len(paths_in)))
        # db = db_gen(path_in)
        # plot_db(db, 1)
        dbs = db_gen_v1(path_in)
        # logging.info(dbs)
        plot_db_v1(dbs, pause_time_s=0.1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V1/pos',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
