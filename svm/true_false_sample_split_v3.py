import argparse
import glob
import logging
import os
import shutil

from utils.utils import make_dirs, set_logging


def run(args):
    dir_root_pos = os.path.join(args.dir_out, 'pos')
    dir_root_neg = os.path.join(args.dir_out, 'neg')

    make_dirs(args.dir_out)
    make_dirs(dir_root_pos)
    make_dirs(dir_root_neg)

    sub_dirs_case = glob.glob(os.path.join(args.dir_in, '*'))
    for sub_dir_case in sub_dirs_case:
        logging.info(sub_dir_case)
        case_name = os.path.basename(sub_dir_case)
        logging.info(case_name)
        paths_src = glob.glob(os.path.join(sub_dir_case, '*'))
        logging.info(paths_src)
        for path_src in paths_src:
            logging.info(path_src)
            assert path_src[-4:] == '.txt'
            file_name = os.path.basename(path_src)
            file_name_save = case_name + '_' + file_name
            if '线香' in case_name or '香烟' in case_name or '气溶胶' in case_name:
                path_dst = os.path.join(dir_root_pos, file_name_save)
            else:
                path_dst = os.path.join(dir_root_neg, file_name_save)
            shutil.copy(path_src, path_dst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/data_202312')
    parser.add_argument('--dir_out', default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V3')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
