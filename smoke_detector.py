import logging
import os
import time
from subprocess import *

import matplotlib.pyplot as plt
import numpy as np
import serial

from utils import LEN_SEQ, MAX_SEQ, SENSOR_ID, MIN_SER_CHAR_NUM, find_key_idx, seq_pick_process, update_svm_label_file


class SensorDB:
    def __init__(self, second_total):
        self.seq_state = list()
        self.seq_forward = list()
        self.seq_backward = list()
        self.last_second_total = second_total
        self.cur_state_idx = 0

    def update(self, second_total, cur_forward, cur_backward, cur_state):
        self.last_second_total = second_total
        self.seq_state.extend(cur_state)
        self.seq_forward.extend(cur_forward)
        self.seq_backward.extend(cur_backward)

    def balance(self):
        self.seq_state = self.seq_state[-MAX_SEQ:]
        self.seq_forward = self.seq_forward[-MAX_SEQ:]
        self.seq_backward = self.seq_backward[-MAX_SEQ:]

    def get_seq_len(self):
        assert len(self.seq_forward) == len(self.seq_backward) and len(self.seq_state) == len(self.seq_backward)
        return len(self.seq_forward)

    def print_db(self):
        logging.info(('seq_state', self.seq_state))
        logging.info(('seq_forward', self.seq_forward))
        logging.info(('seq_backward', self.seq_backward))
        # logging.info(('seq_forward_reversed', list(reversed(self.seq_forward))))
        # logging.info(('seq_backward_reversed', list(reversed(self.seq_backward))))
        logging.info(('seq len', len(self.seq_forward), len(self.seq_backward)))


class SmokeDetector:
    def __init__(self):
        self.db = dict()
        self.interval = 3
        self.ser = serial.Serial('/dev/ttyUSB0', 115200)
        self.ser_buff = ''
        self.ser.flushInput()

    @staticmethod
    def svm_infer(seq, path_label='/home/manu/tmp/rtsvm', dir_libsvm='/home/manu/nfs/libsvm'):
        svmscale_exe = os.path.join(dir_libsvm, 'svm-scale')
        svmpredict_exe = os.path.join(dir_libsvm, 'svm-predict')
        range_file = os.path.join(dir_libsvm, 'tools', 'smartsd.range')
        model_file = os.path.join(dir_libsvm, 'tools', 'smartsd.model')
        test_pathname = path_label
        scaled_test_file = path_label + '.scale'
        predict_test_file = path_label + '.predict'
        os.remove(path_label)
        update_svm_label_file(seq, path_label)
        cmd = '{0} -l 0 -u 1 -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()
        cmd = '{0} -b 1 "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
        Popen(cmd, shell=True).communicate()
        with open(predict_test_file) as f:
            lines = f.readlines()
        return int(lines[1].split(' ')[0])

    def infer_db(self, key):
        if key not in self.db.keys():
            return
        sensor_db = self.db[key]
        if sensor_db.get_seq_len() <= LEN_SEQ:
            return
        while not sensor_db.cur_state_idx == sensor_db.get_seq_len():
            seq_forward = sensor_db.seq_forward[sensor_db.cur_state_idx:]
            # seq_state = sensor_db.seq_state[sensor_db.cur_state_idx:]
            seq_forward = np.array(seq_forward).astype(float)
            # seq_state = np.array(seq_state).astype(float)
            key_idx = find_key_idx(seq_forward)
            if key_idx < 0:
                sensor_db.cur_state_idx = sensor_db.get_seq_len()
                break
            seq_pick, idx_s, idx_e = seq_pick_process(seq_forward, key_idx)
            res = self.svm_infer(seq_pick)
            if res > 0:
                _seq_state = np.array(sensor_db.seq_state)
                _seq_state[idx_s + sensor_db.cur_state_idx: idx_e + sensor_db.cur_state_idx] = 70
                sensor_db.seq_state = list(_seq_state)
            # sensor_db.seq_state[idx_s + sensor_db.cur_state_idx] = 50  # start position
            # sensor_db.seq_state[key_idx + sensor_db.cur_state_idx] = 100  # key position
            sensor_db.cur_state_idx = idx_e + sensor_db.cur_state_idx

    @staticmethod
    def value_preprocess(val, th=255, scale=1 / 32):
        val *= scale
        val = th if val > th else val
        return val

    def update_db_ser(self):
        cnt = self.ser.inWaiting()
        if cnt > 0:
            recv = self.ser.read(self.ser.in_waiting).decode()
            self.ser_buff += recv
            if self.ser_buff[-1] == '\n':
                buff_lst = self.ser_buff.split('\n')
                buff_lst_valid = [x for x in buff_lst if len(x) > MIN_SER_CHAR_NUM - 1]
                self.ser_buff = ''
                logging.info(buff_lst_valid)
                for seq_valid in buff_lst_valid:
                    seq_valid_lst = seq_valid.strip().split()
                    if seq_valid_lst[4] == '08':  # frame data
                        logging.info(seq_valid_lst)
                        # addr, forward, backward = seq_valid_lst[-4], seq_valid_lst[-6], seq_valid_lst[-2]
                        addr, val_forward, val_backward = int(seq_valid_lst[-4], 16), int(seq_valid_lst[-6], 16), int(
                            seq_valid_lst[-2], 16)
                        logging.info((addr, val_forward, val_backward))
                        db_key = '1' + '_' + str(addr)
                        second_total = int(time.time())
                        if db_key not in self.db.keys():
                            self.db[db_key] = SensorDB(second_total)
                        elif second_total - self.db[db_key].last_second_total > MAX_SEQ:
                            self.db.pop(db_key)
                            self.db[db_key] = SensorDB(second_total)
                        elif second_total - self.db[db_key].last_second_total <= 0:
                            continue
                        elif second_total - self.db[db_key].last_second_total > self.interval:
                            seq_pad = [0] * (second_total - self.db[db_key].last_second_total - self.interval)
                            self.db[db_key].update(second_total, seq_pad, seq_pad, seq_pad)
                        # val_forward, val_backward = self.value_preprocess(val_forward), self.value_preprocess(
                        #     val_backward)
                        self.db[db_key].update(second_total, [val_forward], [val_backward], [0])
                        self.db[db_key].balance()

    def update_db(self, paths_txt_sorted):
        # paths_txt_sorted = sorted(paths_txt, key=self._cmp)
        for path_txt_sorted in paths_txt_sorted:
            logging.info(path_txt_sorted)
            file_name = os.path.basename(path_txt_sorted).split('.')[0]
            group_id, sensor_id, date, time = file_name.split('-')
            group_id, date = group_id[-1], int(date)
            if not date == 20231208 or not group_id == '1' or not sensor_id == f'{SENSOR_ID}':
                continue
            # hour, minute, second = time.split(':')
            # hour, minute, second = int(hour), int(minute), int(second)
            # second_total = hour * 60 * 60 + minute * 60 + second
            # logging.info((group_id, sensor_id, date, hour, minute, second, second_total))
            db_key = group_id + '_' + sensor_id
            with open(path_txt_sorted) as f:
                lines = f.readlines()
            for line in lines:
                # logging.info(line)
                val_forward, val_backward = int(line.split(' ')[-3], 16), int(line.split(' ')[-1], 16)
                date, time = line.split(' ')[-8], line.split(' ')[-7]
                # year, month, day = date.split('-')
                hour, minute, second = time.split(':')
                hour, minute, second = int(hour), int(minute), int(second)
                second_total = hour * 60 * 60 + minute * 60 + second
                # logging.info((hour, minute, second, second_total, val_forward, val_backward))
                if db_key not in self.db.keys():
                    self.db[db_key] = SensorDB(second_total)
                elif second_total - self.db[db_key].last_second_total > MAX_SEQ:
                    self.db.pop(db_key)
                    self.db[db_key] = SensorDB(second_total)
                elif second_total - self.db[db_key].last_second_total <= 0:
                    continue
                elif second_total - self.db[db_key].last_second_total > self.interval:
                    seq_pad = [0] * (second_total - self.db[db_key].last_second_total - self.interval)
                    self.db[db_key].update(second_total, seq_pad, seq_pad, seq_pad)
                val_forward, val_backward = self.value_preprocess(val_forward), self.value_preprocess(val_backward)
                self.db[db_key].update(second_total, [val_forward], [val_backward], [0])
                self.db[db_key].balance()

    def print_db(self):
        for key in self.db.keys():
            logging.info(key)
            self.db[key].print_db()

    def plot_db(self, keys, pause_time_s=1):
        for key in keys:
            if key not in self.db.keys():
                return
        plt.ion()
        for i, key in enumerate(keys):
            time_idxs = range(self.db[key].get_seq_len())
            plt.subplot(len(keys), 1, i + 1)
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_forward).astype(float), label='seq_forward'.lower())
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_backward).astype(float),
                     label='seq_backward'.lower())
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_state).astype(float), label='seq_state'.lower())
            plt.ylim(0, 255)
            plt.legend()
            plt.title(key)
        plt.show()
        plt.pause(pause_time_s)
        plt.clf()
