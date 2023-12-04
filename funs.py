import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xlrd


def make_dirs(dir_root):
    if os.path.exists(dir_root):
        shutil.rmtree(dir_root)
    os.makedirs(os.path.join(dir_root), exist_ok=True)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def db_gen(path_in):
    file_name = os.path.basename(path_in)
    obj_xlrd = xlrd.open_workbook(path_in)
    logging.info(obj_xlrd)

    sheet_names = obj_xlrd.sheet_names()
    logging.info(sheet_names)

    obj_sheet_pick = None
    for sheet_name in sheet_names:
        obj_sheet = obj_xlrd.sheet_by_name(sheet_name)
        logging.info(obj_sheet)
        nrows = obj_sheet.nrows
        ncols = obj_sheet.ncols
        logging.info((nrows, ncols))
        if nrows * ncols > 0:
            obj_sheet_pick = obj_sheet

    keys = obj_sheet_pick.row_values(0)
    logging.info(keys)

    db = dict()
    db['fname'] = file_name
    for i, key in enumerate(keys):
        db[key.lower()] = obj_sheet_pick.col_values(i)[1:]
        # logging.info(obj_sheet_pick.col_values(i)[1:])

    # idx_alarm = db['Alarm'].index('Alarm!')
    # logging.info(idx_alarm)
    # db['Alarm'] = [0] * idx_alarm + [255] * (len(db['Time']) - idx_alarm)

    return db


def plot_db(db, pause_time_s=1):
    time_idxs = range(len(db['Time'.lower()]))

    plt.ion()
    # plt.plot(np.array(time_idxs), np.array(db['Line'.lower()]))
    # plt.plot(np.array(time_idxs), np.array(db['Address'.lower()]))
    plt.plot(np.array(time_idxs), np.array(db['Status'.lower()]).astype(float), label='Status'.lower())
    # plt.plot(np.array(time_idxs), np.array(db['Loop'.lower()]))
    plt.plot(np.array(time_idxs), np.array(db['ADC_Forward'.lower()]).astype(float), label='ADC_Forward'.lower())
    plt.plot(np.array(time_idxs), np.array(db['ADC_Backward'.lower()]).astype(float), label='ADC_Backward'.lower())
    plt.plot(np.array(time_idxs), np.array(db['ADC_Heat'.lower()]).astype(float), label='ADC_Heat'.lower())
    plt.plot(np.array(time_idxs), np.array(db['Smoke_Forward'.lower()]).astype(float), label='Smoke_Forward'.lower())
    plt.plot(np.array(time_idxs), np.array(db['Smoke_Backward'.lower()]).astype(float), label='Smoke_Backward'.lower())
    # plt.plot(np.array(time_idxs), np.array(db['Alarm'.lower()]))  # miss in some xlsx
    plt.legend()
    plt.title(db['fname'])
    plt.show()
    plt.pause(pause_time_s)
    plt.clf()
