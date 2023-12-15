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


def update_svm_label_file(seq_pick, path_out, subset='neg'):
    idx_feat = 0
    with open(path_out, 'a') as f:
        label = '+1' if 'pos' in subset else '-1'
        f.write(label + ' ')
        for feat in seq_pick:
            f.write(f'{idx_feat + 1}:{feat} ')
            idx_feat += 1
    with open(path_out, 'a') as f:
        f.write('\n')


def seq_pick_process(feats, key_idx, shift_rate_left=1 / 2, db=None, key_debug='Status'):
    seq_len = LEN_SEQ
    idx_s = 0 if key_idx - seq_len * shift_rate_left < 0 else int(
        key_idx - seq_len * shift_rate_left)
    idx_e = len(feats) if idx_s + seq_len > len(feats) else int(idx_s + seq_len)
    if db is not None:
        db[key_debug.lower()][int(key_idx)] = 100
        db[key_debug.lower()][idx_s] = 255
        db[key_debug.lower()][idx_e - 1] = 255
    seq_pick = feats[idx_s:idx_e]
    if len(seq_pick) < seq_len:
        pad = [seq_pick[-1]] * (seq_len - len(seq_pick))  # pad last seq val
        seq_pick = np.append(seq_pick, pad)
    assert len(seq_pick) == LEN_SEQ
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


def db_gen_v1(path_in):
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

    key_addr = '部件地址'
    assert key_addr in keys
    idx_addr = keys.index(key_addr)
    raw_data_addr = obj_sheet_pick.col_values(idx_addr)[1:]
    # logging.info(raw_data_addr)
    addr_min, addr_max = min(raw_data_addr), max(raw_data_addr)
    addr_set = set(raw_data_addr)

    dbs = list()
    for addr in addr_set:
        db = dict()
        db['fname'] = file_name
        raw_data_addr_array = np.array(obj_sheet_pick.col_values(idx_addr)[1:])
        data_mask = raw_data_addr_array == addr
        for i, key in enumerate(keys):
            db_key = key
            if key == '部件地址':
                db_key = 'addr'
            if key == '状态':
                db_key = 'state'
            if key == '暗电流':
                db_key = 'dark'
            if key == '前向电流':
                db_key = 'forward'
            if key == '后向电流':
                db_key = 'backward'
            if key == '温度':
                db_key = 'temperature'
            if key == '时间戳':
                db_key = 'timestamp'
                continue
            raw_data_array = np.array(obj_sheet_pick.col_values(i)[1:]).astype(float)
            if key == key_addr:
                db[db_key.lower()] = np.concatenate((np.array([addr] * LEN_SEQ), raw_data_array[data_mask]), axis=0)
            else:
                db[db_key.lower()] = np.concatenate((np.array([0] * LEN_SEQ), raw_data_array[data_mask]), axis=0)
        dbs.append(db)

    return dbs


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

def plot_dbs_v1(dbs, pause_time_s=1, case='', dir_save='', idx_save=0):
    plt.ion()
    for db in dbs:
        time_idxs = range(len(db['addr'.lower()]))
        plt.title(db['fname'] + f' <{case}>')
        for key in db.keys():
            label = key
            if key == 'fname' or key == 'timestamp' or key == 'co' or key == 'dark':
                continue
            plt.plot(np.array(time_idxs), np.array(db[key]).astype(float), label=key)
            plt.legend()
        plt.show()
        if not dir_save == '':
            fn_base, _ = os.path.basename(db['fname']).split('.')
            plt.savefig(os.path.join(dir_save, f'{idx_save}_' + fn_base))
        plt.pause(pause_time_s)
        plt.clf()

def plot_db_v1(db, pause_time_s=1, case='', dir_save='', idx_save=0):
    plt.ion()
    time_idxs = range(len(db['addr'.lower()]))
    plt.title(db['fname'] + f' <{case}>')
    for key in db.keys():
        if key == 'fname' or key == 'timestamp' or key == 'co' or key == 'dark':
            continue
        plt.plot(np.array(time_idxs), np.array(db[key]).astype(float), label=key)
        plt.legend()
    plt.show()
    if not dir_save == '':
        fn_base, _ = os.path.basename(db['fname']).split('.')
        plt.savefig(os.path.join(dir_save, f'{idx_save}_' + fn_base))
    plt.pause(pause_time_s)
    plt.clf()


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
