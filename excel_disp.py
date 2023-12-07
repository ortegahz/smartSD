import argparse
import logging

from utils import plot_db, db_gen, set_logging


def run(args):
    db = db_gen(args.path_in)
    plot_db(db, 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR_TFS/neg/Hot_SOCTA_2022-08-10-1_1.xlsx',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
