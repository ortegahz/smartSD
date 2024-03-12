import argparse
import glob
import logging
import os

from core.smoke_detector import SmokeDetector
from utils.utils import set_logging, db_gen_v4, make_dirs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='/home/manu/tmp/particles_sorted_v1')
    parser.add_argument('--dir_root_libsvm', default='/home/manu/nfs/libsvm')
    # parser.add_argument('--key_choose_forward', default='ADC_Forward')
    # parser.add_argument('--key_choose_backward', default='ADC_Backward')
    parser.add_argument('--key_choose_forward', default='forward_red')
    parser.add_argument('--key_choose_backward', default='backward_red')
    parser.add_argument('--addrs_sensor', default=[f'1_{1}'])
    parser.add_argument('--dir_plot_save', default='/home/manu/tmp/infer_results_v1')
    parser.add_argument('--sample_pick', default=None)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    smoke_detector = SmokeDetector()
    make_dirs(args.dir_plot_save)
    paths_in = glob.glob(os.path.join(args.dir_in, '*'))
    for j, path_in in enumerate(paths_in):
        if args.sample_pick is not None and args.sample_pick not in path_in:
            continue
        logging.info((j, len(paths_in), path_in))
        # db = db_gen(path_in)
        db = db_gen_v4(path_in)
        # feat_backward = np.array(db[args.key_choose_backward.lower()]).astype('float')
        # if np.max(feat_backward) < ALARM_LOW_BASE_TH:
        #     continue
        for i in range(db['seq_len_max']):
            smoke_detector.update_db_v1(db, i, args.key_choose_forward,
                                        args.key_choose_backward, db_key=args.addrs_sensor[0])
            # smoke_detector.infer_db(args.addrs_sensor, args.dir_root_libsvm)
            smoke_detector.infer_db_naive(args.addrs_sensor)

        title_info = os.path.basename(path_in).split('.')[0]
        title_info = f'{j}[{len(paths_in)}]_' + title_info
        path_plot_save = os.path.join(args.dir_plot_save, title_info)
        smoke_detector.plot_db(args.addrs_sensor, pause_time_s=0.001, save_plot=True, path_save=path_plot_save,
                               title_info=title_info, show=False)
        smoke_detector.clear_db()


if __name__ == '__main__':
    main()
