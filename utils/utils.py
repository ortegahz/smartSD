import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import xlrd

GUARANTEE_BACK_TH = 25000
SENSE_LOW_BACK_TH = 20000
LEN_SEQ_LOW = 16

LEN_SEQ_NAIVE_BG = 8
LEN_SEQ_NAIVE = 8
ALARM_NAIVE_TH = 16
ALARM_NAIVE_BG_LR = 1e-4

LEN_SEQ = 16
LEN_OVERLAP = 16
MAX_SEQ = 4096 * 64
SENSOR_ID = 5
MIN_SER_CHAR_NUM = 40
ALARM_CNT_TH = 16
ALARM_CNT_TH_SVM = 10
ALARM_CNT_GUARANTEE_TH = 16
ALARM_NEG_SCORE_WEIGHT = 3
ALARM_GUARANTEE_SHORT_TH = 35000
ALARM_LOW_DIFF_TH = 800
ALARM_LOW_TH = 256
ALARM_LOW_BASE_TH = 32
ALARM_LOW_SVM_WIN_LEN = 64
ALARM_LOW_CNT_TH_SVM = 5
ALARM_LOW_CNT_DECAY = 0.1
ALARM_LOW_NEG_SCORE_WEIGHT = 2
ALARM_LOW_SMOOTH_TH = 1000
DEBUG_ALARM_INDICATOR_VAL = 2 ** 16
ALARM_LOW_ANCHOR_STEP = 2


# SVM_LEN_SEQ_FUTURE = 16


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
        db[key_debug.lower()][int(key_idx - 1)] = 100
        db[key_debug.lower()][idx_s] = 255
        db[key_debug.lower()][idx_e - 1] = 255
    seq_pick = feats[idx_s:idx_e]
    if len(seq_pick) < seq_len:
        pad = [seq_pick[-1]] * (seq_len - len(seq_pick))  # pad last seq val
        seq_pick = np.append(seq_pick, pad)
    assert len(seq_pick) == LEN_SEQ
    return seq_pick, idx_s, idx_e


def seq_pick_process_future(seq, anchor_idx):
    idx_s = anchor_idx + 1
    idx_e = len(seq) if idx_s + LEN_SEQ_LOW > len(seq) else int(idx_s + LEN_SEQ_LOW)
    seq_pick = seq[idx_s:idx_e]
    if len(seq_pick) < LEN_SEQ_LOW:
        pad = [seq_pick[-1]] * (LEN_SEQ_LOW - len(seq_pick))  # pad last seq val
        seq_pick = np.append(seq_pick, pad)
    assert len(seq_pick) == LEN_SEQ_LOW
    return seq_pick, idx_s, idx_e


def find_anchor_idx_up(seq, th_sum=32, th_left=32, th_right=250):
    anchor_idx = -1
    if len(seq) < LEN_SEQ:
        return anchor_idx
    for i in range(LEN_SEQ - 1, len(seq)):
        seq_pick = seq[i - LEN_SEQ + 1:i + 1]
        seq_diff = np.diff(seq_pick)
        seq_diff_valid = seq_diff[seq_diff > 0.]
        seq_diff_valid_sum = np.sum(seq_diff_valid) if len(seq_diff_valid) > 0 else 0
        seq_pick_idx_max = np.argmax(seq_pick)
        if seq_diff_valid_sum > th_sum and th_left < seq_pick[0] < seq_pick[-1] < th_right and \
                seq_pick_idx_max == LEN_SEQ - 1:
            anchor_idx = i
            break
    return anchor_idx


def find_anchor_idxes_up(seq, th_sum=32, th_left=32, th_right=250):
    anchor_idxes = list()
    if len(seq) < LEN_SEQ:
        return anchor_idxes
    # for i in range(LEN_SEQ - 1, len(seq)):
    i = LEN_SEQ - 1
    while i < len(seq):
        seq_pick = seq[i - LEN_SEQ + 1:i + 1]
        seq_diff = np.diff(seq_pick)
        seq_diff_valid = seq_diff[seq_diff > 0.]
        seq_diff_valid_sum = np.sum(seq_diff_valid) if len(seq_diff_valid) > 0 else 0
        seq_pick_idx_max = np.argmax(seq_pick)
        if seq_diff_valid_sum > th_sum and th_left < seq_pick[0] < seq_pick[-1] < th_right and \
                seq_pick_idx_max == LEN_SEQ - 1:
            anchor_idxes.append(i)
            i += LEN_SEQ
        else:
            i += 1
    return anchor_idxes


def find_anchor_idxes(seq, last_val_th=0, anchor_val_th=256):
    anchor_idxes = list()
    anchor_idx, anchor_val, cnt = -1, -1, 0
    for i, val in enumerate(seq):
        if cnt >= LEN_SEQ_LOW and val > last_val_th and seq[anchor_idx] > anchor_val_th:
            anchor_idxes.append(anchor_idx)
            anchor_idx, anchor_val, cnt = -1, -1, 0
        elif cnt >= LEN_SEQ_LOW:
            anchor_idx, anchor_val, cnt = -1, -1, 0
        elif val >= anchor_val * ALARM_LOW_ANCHOR_STEP:
            anchor_idx, anchor_val, cnt = i, val, 0
        else:
            cnt += 1
    anchor_val_max, anchor_idx_max = -1, -1
    for anchor_idx in anchor_idxes:
        anchor_val = seq[anchor_idx]
        if anchor_val > anchor_val_max:
            anchor_val_max, anchor_idx_max = anchor_val, anchor_idx
    return anchor_idxes, anchor_idx_max


def find_key_idx(seq, th_val=10, th_cnt=10, th_delta=10):
    cnt = 0
    key_idx = -1
    # filter one
    for i, val in enumerate(seq):
        cnt = cnt + 1 if val > th_val else 0
        if cnt > th_cnt:
            key_idx = i
            logging.info(f'filter one found key_idx -> {key_idx}')
            break
    if key_idx == -1:
        return key_idx
    # filter two
    flag_valid = False
    while key_idx + int(LEN_SEQ / 2) <= len(seq):
        # logging.info(f'key_idx -> {key_idx}')
        idx_end = key_idx + int(LEN_SEQ / 2)
        # idx_start = key_idx - int(LEN_SEQ / 2)
        mean = np.mean(np.absolute(seq[key_idx - 1:idx_end - 1] - seq[key_idx:idx_end]))
        # if mean > th_mean or seq[idx_end-1] > 220:
        logging.info(f'seq[idx_end - 1] - seq[key_idx] -> {seq[idx_end - 1] - seq[key_idx]}')
        if seq[idx_end - 1] - seq[key_idx] > th_delta:
            flag_valid = True
            logging.info(f'filter two found key_idx -> {key_idx}')
            break
        key_idx += int(LEN_SEQ / 2)
    # logging.info(f'mean --> {mean}')
    return key_idx if flag_valid else -1


def make_dirs(dir_root, reset=True):
    if os.path.exists(dir_root) and reset:
        shutil.rmtree(dir_root)
    os.makedirs(os.path.join(dir_root), exist_ok=True)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def db_gen_v4(path_in):
    file_name = os.path.basename(path_in)
    db = dict()
    seq_len = 0
    db['fname'] = file_name
    keys = ('voc', 'co', 'temper', 'humid', 'pm010', 'pm025', 'pm100', 'forward_red', 'forward_blue', 'backward_red')
    for key in keys:
        db[key] = list()

    with open(path_in, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if len(line) < 128:
            continue
        # logging.info(line)
        line_lst = line.strip().split(',')
        if not len(line_lst) == 14:
            continue
        line_lst_pick = line_lst[1:]
        # logging.info(line_lst_pick)
        voc, _, co, _, temper, humid, pm010, pm025, pm100, _, _, _, smoke = line_lst_pick
        cur_data_lst = [voc, co, temper, humid, pm010, pm025, pm100, smoke]
        # logging.info(cur_data_lst)
        smoke_lst = smoke.split()
        # logging.info(smoke_lst)
        forward_red, forward_blue, backward_red = smoke_lst[2], smoke_lst[3], smoke_lst[4]
        cur_data_lst = [voc, co, temper, humid, pm010, pm025, pm100, forward_red, forward_blue, backward_red]
        # logging.info(cur_data_lst)
        for i, key in enumerate(keys):
            db[key].append(cur_data_lst[i])  # must be same order
        seq_len += 1
    db['seq_len_max'] = seq_len
    return db

def db_gen_v3(path_in):
    keys = ['forward', 'backward']
    db = dict()
    db['fname'] = path_in
    for key in keys:
        db[key] = list()
    with open(path_in, 'r') as f:
        lines = f.readlines()
    lines_valid = lines[1:]
    # logging.info(lines_valid)
    for line in lines_valid:
        line_lst = line.split()
        # logging.info(line_lst)
        db['forward'].append(float(line_lst[0]))
        db['backward'].append(float(line_lst[1]))
    db['state'] = np.zeros_like(np.array(db['forward']).astype('float'))
    db['seq_len'] = len(lines_valid)
    return db


def db_gen_v2(path_in):
    keys = ['time', 'address', 'status', 'temperature', 'forward', 'backward']
    db = dict()
    db['fname'] = path_in
    for key in keys:
        db[key] = list()
    with open(path_in, 'r') as f:
        lines = f.readlines()
    lines_valid = lines[1:]
    # logging.info(lines_valid)
    for line in lines_valid:
        line_lst = line.split()
        # logging.info(line_lst)
        db['address'].append(int(line_lst[2][:-1], base=16))
        db['status'].append(int(line_lst[3], base=16))
        db['forward'].append(int(line_lst[-1], base=16))
        db['backward'].append(int(line_lst[-2], base=16))
        db['temperature'].append(int(line_lst[-3], base=16))
    addr_set = set(db['address'])
    assert len(addr_set) == 1
    seq_len_max = 0
    for key in db.keys():
        if key == 'time' or key == 'fname':
            continue
        if key == 'address':
            db[key] = np.concatenate((np.array([db['address'][0]] * LEN_SEQ), np.array(db[key]).astype('float')),
                                     axis=0)
        else:
            db[key] = np.concatenate((np.array([0] * LEN_SEQ), np.array(db[key]).astype('float')), axis=0)
        seq_len_max = len(db[key.lower()]) if len(db[key.lower()]) > seq_len_max else seq_len_max
    db['seq_len_max'] = seq_len_max
    return db


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
        seq_len_max = 0
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
            seq_len_max = len(db[db_key.lower()]) if len(db[db_key.lower()]) > seq_len_max else seq_len_max
        db['seq_len_max'] = seq_len_max
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
    seq_len_max = 0
    for i, key in enumerate(keys):
        db[key.lower()] = [0] * LEN_SEQ + obj_sheet_pick.col_values(i)[1:]
        seq_len_max = len(db[key.lower()]) if len(db[key.lower()]) > seq_len_max else seq_len_max
        # logging.info(obj_sheet_pick.col_values(i)[1:])

    # idx_alarm = db['Alarm'].index('Alarm!')
    # logging.info(idx_alarm)
    # db['Alarm'] = [0] * idx_alarm + [255] * (len(db['Time']) - idx_alarm)
    db['seq_len_max'] = seq_len_max
    return db


def plot_db_v3(db, pause_time_s=1, case='', dir_save='', idx_save=0):
    plt.ion()
    time_idxs = range(len(db['forward']))
    plt.title(db['fname'] + f' <{case}>')
    for key in db.keys():
        if key == 'fname' or key == 'seq_len':
            continue
        plt.plot(np.array(time_idxs), np.array(db[key]).astype(float), label=key)
        plt.ylim(0, 2 ** 16)
        plt.legend()
    plt.show()
    if dir_save:
        fn_base, _ = os.path.basename(db['fname']).split('.')
        plt.savefig(os.path.join(dir_save, f'{idx_save}_' + fn_base))
    plt.pause(pause_time_s)
    plt.clf()


def plot_db_v2(db, seq_fft=None, pause_time_s=1, case='', dir_save='', idx_save=0):
    plt.ion()
    time_idxs = range(len(db['address']))
    if seq_fft is not None:
        freq_idxs = range(len(seq_fft))
        plt.plot(np.array(freq_idxs), np.array(seq_fft).astype(float), label='freq'.lower())
    plt.title(db['fname'] + f' <{case}>')
    for key in db.keys():
        if key == 'fname' or key == 'time' or key == 'seq_len_max':
            continue
        plt.plot(np.array(time_idxs), np.array(db[key]).astype(float), label=key)
        plt.legend()
    plt.show()
    if not dir_save == '':
        fn_base, _ = os.path.basename(db['fname']).split('.')
        plt.savefig(os.path.join(dir_save, f'{idx_save}_' + fn_base))
    plt.pause(pause_time_s)
    plt.clf()


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


def plot_db_v1(db, seq_fft=None, pause_time_s=1, case='', dir_save='', idx_save=0):
    plt.ion()
    time_idxs = range(len(db['addr'.lower()]))
    if seq_fft is not None:
        freq_idxs = range(len(seq_fft))
        plt.plot(np.array(freq_idxs), np.array(seq_fft).astype(float), label='freq'.lower())
    plt.title(db['fname'] + f' <{case}>')
    for key in db.keys():
        if key == 'fname' or key == 'timestamp' or key == 'co' or key == 'dark' or key == 'seq_len_max':
            continue
        plt.plot(np.array(time_idxs), np.array(db[key]).astype(float), label=key)
        plt.legend()
    plt.show()
    if not dir_save == '':
        fn_base, _ = os.path.basename(db['fname']).split('.')
        plt.savefig(os.path.join(dir_save, f'{idx_save}_' + fn_base))
    plt.pause(pause_time_s)
    plt.clf()


def plot_db(db, seq_fft=None, pause_time_s=1, label='', dir_save='', idx_save=0):
    time_idxs = range(len(db['Time'.lower()]))
    plt.ion()
    if seq_fft is not None:
        freq_idxs = range(len(seq_fft))
        plt.plot(np.array(freq_idxs), np.array(seq_fft).astype(float), label='freq'.lower())
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
