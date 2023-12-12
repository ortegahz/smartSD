import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xlrd

LEN_SEQ = 32
LEN_OVERLAP = 16
MAX_SEQ = 4096 * 64
SENSOR_ID = 1


def seq_pick_process(feats, key_idx, shift_rate_left=1 / 2, db=None):
    seq_len = LEN_SEQ
    idx_s = 0 if key_idx - seq_len * shift_rate_left < 0 else int(
        key_idx - seq_len * shift_rate_left)
    idx_e = len(feats) if idx_s + seq_len > len(feats) else int(idx_s + seq_len)
    if db is not None:
        db['Status'.lower()][int(key_idx)] = 100
        db['Status'.lower()][idx_s] = 255
        db['Status'.lower()][idx_e - 1] = 255
    seq_pick = feats[idx_s:idx_e]
    if len(seq_pick) < seq_len:
        pad = [seq_pick[-1]] * (seq_len - len(seq_pick))  # pad last seq val
        seq_pick = np.append(seq_pick, pad)
    return seq_pick, idx_s, idx_e


def find_key_idx(seq, th_val=10, th_cnt=10, th_mean=1.):
    cnt = 0
    key_idx = -1
    for i, val in enumerate(seq):
        cnt = cnt + 1 if val > th_val else 0
        if cnt > th_cnt:
            key_idx = i
            break
    if key_idx == -1:
        return key_idx
    mean = 0.
    while key_idx + int(LEN_SEQ / 2) <= len(seq):
        idx_end = key_idx + int(LEN_SEQ / 2)
        mean = np.mean(np.absolute(seq[key_idx - 1:idx_end - 1] - seq[key_idx:idx_end]))
        if mean > th_mean or seq[idx_end] > 220:
            break
        key_idx += int(LEN_SEQ / 2)
    # logging.info(f'mean --> {mean}')
    return key_idx


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
    # logging.info(obj_xlrd)

    sheet_names = obj_xlrd.sheet_names()
    # logging.info(sheet_names)

    obj_sheet_pick = None
    for sheet_name in sheet_names:
        obj_sheet = obj_xlrd.sheet_by_name(sheet_name)
        # logging.info(obj_sheet)
        nrows = obj_sheet.nrows
        ncols = obj_sheet.ncols
        # logging.info((nrows, ncols))
        if nrows * ncols > 0:
            obj_sheet_pick = obj_sheet

    keys = obj_sheet_pick.row_values(0)
    # logging.info(keys)

    db = dict()
    db['fname'] = file_name
    for i, key in enumerate(keys):
        db[key.lower()] = [0] * LEN_SEQ + obj_sheet_pick.col_values(i)[1:]
        # logging.info(obj_sheet_pick.col_values(i)[1:])

    # idx_alarm = db['Alarm'].index('Alarm!')
    # logging.info(idx_alarm)
    # db['Alarm'] = [0] * idx_alarm + [255] * (len(db['Time']) - idx_alarm)

    return db


def plot_db(db, pause_time_s=1, label='', dir_save='', idx_save=0):
    time_idxs = range(len(db['Time'.lower()]))

    plt.ion()
    # plt.plot(np.array(time_idxs), np.array(db['Line'.lower()]))
    # plt.plot(np.array(time_idxs), np.array(db['Address'.lower()]))
    plt.plot(np.array(time_idxs), np.array(db['Status'.lower()]).astype(float), label='Status'.lower())
    # plt.plot(np.array(time_idxs), np.array(db['Loop'.lower()]))
    plt.plot(np.array(time_idxs), np.array(db['ADC_Forward'.lower()]).astype(float), label='ADC_Forward'.lower())
    plt.plot(np.array(time_idxs), np.array(db['ADC_Backward'.lower()]).astype(float), label='ADC_Backward'.lower())
    # plt.plot(np.array(time_idxs), np.array(db['ADC_Heat'.lower()]).astype(float), label='ADC_Heat'.lower())
    plt.plot(np.array(time_idxs), np.array(db['Smoke_Forward'.lower()]).astype(float), label='Smoke_Forward'.lower())
    plt.plot(np.array(time_idxs), np.array(db['Smoke_Backward'.lower()]).astype(float), label='Smoke_Backward'.lower())
    # plt.plot(np.array(time_idxs), np.array(db['Alarm'.lower()]))  # miss in some xlsx
    plt.legend()
    plt.title(db['fname'] + f' <{label}>')
    plt.show()
    if not dir_save == '':
        fn_base, _ = os.path.basename(db['fname']).split('.')
        plt.savefig(os.path.join(dir_save, f'{idx_save}_' + fn_base))
    plt.pause(pause_time_s)
    plt.clf()
