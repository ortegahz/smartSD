import argparse
import glob
import logging
import os

from funs import plot_db, db_gen, set_logging


def run(args):
    paths_in = glob.glob(os.path.join(args.dir_in, '*.xlsx'))

    for path_in in paths_in:
        logging.info(path_in)
        db = db_gen(path_in)
        plot_db(db)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR/',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
