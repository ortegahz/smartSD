import argparse
import logging

from funs import plot_db, db_gen, set_logging


def run(args):
    db = db_gen(args.path_in)
    plot_db(db)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR/TF1_20220801092906_101001_data_export_002.xlsx',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
