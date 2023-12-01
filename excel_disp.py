import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import xlrd


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def run(args):
    obj_xlrd = xlrd.open_workbook(args.path_in)
    logging.info(obj_xlrd)

    sheet_names = obj_xlrd.sheet_names()
    logging.info(sheet_names)

    obj_sheet = obj_xlrd.sheet_by_name('Sheet4')
    logging.info(obj_sheet)

    nrows = obj_sheet.nrows
    ncols = obj_sheet.ncols
    logging.info((nrows, ncols))

    keys = obj_sheet.row_values(0)[:-1]
    keys.append('Alarm')
    logging.info(keys)

    db = dict()
    for i, key in enumerate(keys):
        db[key] = obj_sheet.col_values(i)[1:]

    idx_alarm = db['Alarm'].index('Alarm!')
    db['Alarm'] = [0] * idx_alarm + [255] * (len(db['Time']) - idx_alarm)

    plt.plot(np.array(db['Time']), np.array(db['Line']))
    plt.plot(np.array(db['Time']), np.array(db['Address']))
    plt.plot(np.array(db['Time']), np.array(db['Status']))
    plt.plot(np.array(db['Time']), np.array(db['Loop']))
    plt.plot(np.array(db['Time']), np.array(db['ADC_Forward']))
    plt.plot(np.array(db['Time']), np.array(db['ADC_Backward']))
    plt.plot(np.array(db['Time']), np.array(db['ADC_Heat']))
    plt.plot(np.array(db['Time']), np.array(db['Smoke_Forward']))
    plt.plot(np.array(db['Time']), np.array(db['Smoke_Backward']))
    plt.plot(np.array(db['Time']), np.array(db['Alarm']))
    plt.show()


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
