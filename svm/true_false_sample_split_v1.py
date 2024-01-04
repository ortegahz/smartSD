import argparse
import glob
import logging
import os
import shutil

from utils import set_logging, make_dirs


def run(args):
    dir_root_pos = os.path.join(args.dir_out, 'pos')
    dir_root_neg = os.path.join(args.dir_out, 'neg')

    make_dirs(args.dir_out)
    make_dirs(dir_root_pos)
    make_dirs(dir_root_neg)

    sub_dirs_case = glob.glob(os.path.join(args.dir_in, '*'))
    for sub_dir_case in sub_dirs_case:
        if sub_dir_case[-4:] == '.pdf':
            continue
        logging.info(sub_dir_case)
        case_name = os.path.basename(sub_dir_case)
        logging.info(case_name)
        dir_case_data = os.path.join(sub_dir_case, '控制器数据')
        logging.info(dir_case_data)
        assert os.path.exists(dir_case_data)
        paths_src = glob.glob(os.path.join(dir_case_data, '*'))
        logging.info(paths_src)
        case_name = case_name.lower()
        for path_src in paths_src:
            logging.info(path_src)
            assert path_src[-4:] == '.xls'
            file_name = os.path.basename(path_src)
            file_name_save = case_name + '_' + file_name
            if 'tf' in case_name:
                path_dst = os.path.join(dir_root_pos, file_name_save)
            else:
                path_dst = os.path.join(dir_root_neg, file_name_save)
            shutil.copy(path_src, path_dst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/media/manu/data/docs/smokes/原数据')
    parser.add_argument('--dir_out', default='/media/manu/data/docs/smokes/AI烟感资料整合-第一批/SONAR_TFS_V1')
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
