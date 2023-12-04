import argparse
import glob
import logging
import os
import shutil

from funs import set_logging, make_dirs


def run(args):
    make_dirs(args.dir_out)

    paths_in = glob.glob(os.path.join(args.dir_in, '*.xlsx'))
    case_names_set = set()

    for path_in in paths_in:
        file_name = os.path.basename(path_in)
        file_name_base, _ = file_name.split('.')
        case_name = file_name_base.split('_')[0]
        case_name_sub = file_name_base.split('_')[1]
        logging.info(case_name)

        if case_name == 'Vapor':
            case_name = case_name + '_' + case_name_sub

        case_names_set.add(case_name)

    logging.info(case_names_set)

    for case_name in case_names_set:
        case_dir = os.path.join(args.dir_out, case_name)
        make_dirs(case_dir)

    for path_in in paths_in:
        file_name = os.path.basename(path_in)
        file_name_base, _ = file_name.split('.')
        case_name = file_name_base.split('_')[0]
        case_name_sub = file_name_base.split('_')[1]

        if case_name == 'Vapor':
            case_name = case_name + '_' + case_name_sub

        path_out = os.path.join(args.dir_out, case_name, file_name)
        logging.info(path_out)
        shutil.copy(path_in, path_out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR/',
                        type=str)
    parser.add_argument('--dir_out',
                        default='/media/manu/data/docs/nxp/AI烟感资料整合-第一批/SONAR_SORT',
                        type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
