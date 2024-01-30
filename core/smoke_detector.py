import logging
import os
import sys
import time
from subprocess import *

import matplotlib.pyplot as plt
import numpy as np
import serial

from demos.demo_fft import fft_wrapper
from utils.utils import ALARM_CNT_TH_SVM, ALARM_LOW_CNT_DECAY, ALARM_LOW_TH, ALARM_CNT_GUARANTEE_TH, GUARANTEE_BACK_TH, \
    LEN_SEQ, LEN_SEQ_LOW, ALARM_LOW_CNT_TH_SVM, MAX_SEQ, SENSOR_ID, ALARM_GUARANTEE_SHORT_TH, MIN_SER_CHAR_NUM, \
    DEBUG_ALARM_INDICATOR_VAL, ALARM_LOW_ANCHOR_STEP, ALARM_LOW_BASE_TH, ALARM_NEG_SCORE_WEIGHT, \
    ALARM_LOW_NEG_SCORE_WEIGHT, update_svm_label_file


class SensorDB:
    def __init__(self, second_total=-1):
        self.seq_low_sens_score = list()
        self.seq_forward_amp = list()
        self.seq_backward_amp = list()
        self.seq_state_time = list()
        self.seq_state_freq = list()
        self.seq_state = list()
        self.seq_forward = list()
        self.seq_backward = list()
        self.last_second_total = second_total
        self.cur_state_idx = 0
        self.cnt_alarm = 0
        self.cnt_alarm_svm = 0
        self.cnt_alarm_guarantee = 0
        self.alarm_logic_low_anchor_idx = 0
        self.alarm_logic_low_probation_scores = list()
        self.alarm_record = list()
        self.alarm_record_last_pos = -1
        self.anchor_val = 0

    def update(self, cur_forward, cur_backward, cur_state, cur_state_t, cur_state_f, cur_forward_amp, cur_backward_amp,
               cur_low_sens_score,
               second_total=-1):
        self.last_second_total = second_total
        self.seq_state_time.extend(cur_state_t)
        self.seq_state_freq.extend(cur_state_f)
        self.seq_state.extend(cur_state)
        self.seq_forward.extend(cur_forward)
        self.seq_backward.extend(cur_backward)
        self.seq_forward_amp.extend(cur_forward_amp)
        self.seq_backward_amp.extend(cur_backward_amp)
        self.seq_low_sens_score.extend(cur_low_sens_score)

    def balance(self):
        self.seq_state_time = self.seq_state_time[-MAX_SEQ:]
        self.seq_state_freq = self.seq_state_freq[-MAX_SEQ:]
        self.seq_state = self.seq_state[-MAX_SEQ:]
        self.seq_forward = self.seq_forward[-MAX_SEQ:]
        self.seq_backward = self.seq_backward[-MAX_SEQ:]
        self.seq_forward_amp = self.seq_forward_amp[-MAX_SEQ:]
        self.seq_backward_amp = self.seq_backward_amp[-MAX_SEQ:]
        self.seq_low_sens_score = self.seq_low_sens_score[-MAX_SEQ:]

    def get_seq_len(self):
        assert len(self.seq_forward) == len(self.seq_backward) and len(self.seq_state) == len(self.seq_backward)
        return len(self.seq_forward)

    def print_db(self):
        logging.info(('seq_state', self.seq_state))
        logging.info(('seq_forward', self.seq_forward))
        logging.info(('seq_backward', self.seq_backward))
        logging.info(('seq len', len(self.seq_forward), len(self.seq_backward)))


class SmokeDetector:
    def __init__(self, dev_ser=None):
        self.db = dict()
        self.interval = 3
        if dev_ser:
            self.ser = serial.Serial(dev_ser, 115200)
            self.ser_buff = ''
            self.ser.flushInput()

    @staticmethod
    def svm_infer_freq(seq, path_label='./rtsvm_freq', dir_libsvm='/home/manu/nfs/libsvm'):
        is_win32 = (sys.platform == 'win32')
        if is_win32:
            svmscale_exe = os.path.join(dir_libsvm, 'windows', 'svm-scale.exe')
            svmpredict_exe = os.path.join(dir_libsvm, 'windows', 'svm-predict.exe')
        else:
            svmscale_exe = os.path.join(dir_libsvm, 'svm-scale')
            svmpredict_exe = os.path.join(dir_libsvm, 'svm-predict')
        range_file = os.path.join(dir_libsvm, 'tools', 'smartsd_freq.range')
        model_file = os.path.join(dir_libsvm, 'tools', 'smartsd_freq.model')
        test_pathname = path_label
        scaled_test_file = path_label + '.scale'
        predict_test_file = path_label + '.predict'
        if os.path.exists(path_label):
            os.remove(path_label)
        update_svm_label_file(seq, path_label)
        cmd = '{0} -l 0 -u 1 -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()
        cmd = '{0} -b 0 "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
        Popen(cmd, shell=True).communicate()
        with open(predict_test_file) as f:
            lines = f.readlines()
        # return int(lines[1].split(' ')[0])
        return int(lines[0].strip())

    @staticmethod
    def svm_infer_time(seq, path_label='./rtsvm_time', dir_libsvm='/home/manu/nfs/libsvm'):
        is_win32 = (sys.platform == 'win32')
        if is_win32:
            svmscale_exe = os.path.join(dir_libsvm, 'windows', 'svm-scale.exe')
            svmpredict_exe = os.path.join(dir_libsvm, 'windows', 'svm-predict.exe')
        else:
            svmscale_exe = os.path.join(dir_libsvm, 'svm-scale')
            svmpredict_exe = os.path.join(dir_libsvm, 'svm-predict')
        range_file = os.path.join(dir_libsvm, 'tools', 'smartsd_time.range')
        model_file = os.path.join(dir_libsvm, 'tools', 'smartsd_time.model')
        test_pathname = path_label
        scaled_test_file = path_label + '.scale'
        predict_test_file = path_label + '.predict'
        if os.path.exists(path_label):
            os.remove(path_label)
        update_svm_label_file(seq, path_label)
        cmd = '{0} -l 0 -u 1 -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()
        cmd = '{0} -b 0 "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
        Popen(cmd, shell=True).communicate()
        with open(predict_test_file) as f:
            lines = f.readlines()
        # return int(lines[1].split(' ')[0])
        return int(lines[0].strip())

    @staticmethod
    def svm_infer(seq, suffix='', path_label='./rtsvm', dir_libsvm='/home/manu/nfs/libsvm'):
        is_win32 = (sys.platform == 'win32')
        if is_win32:
            svmscale_exe = os.path.join(dir_libsvm, 'windows', 'svm-scale.exe')
            svmpredict_exe = os.path.join(dir_libsvm, 'windows', 'svm-predict.exe')
        else:
            svmscale_exe = os.path.join(dir_libsvm, 'svm-scale')
            svmpredict_exe = os.path.join(dir_libsvm, 'svm-predict')
        # range_file = os.path.join(dir_libsvm, 'tools', 'smartsd_time.range')
        # model_file = os.path.join(dir_libsvm, 'tools', 'smartsd_time.model')
        range_file = os.path.join(dir_libsvm, 'tools', 'smartsd' + suffix + '.range')
        model_file = os.path.join(dir_libsvm, 'tools', 'smartsd' + suffix + '.model')
        test_pathname = path_label
        scaled_test_file = path_label + '.scale'
        predict_test_file = path_label + '.predict'
        if os.path.exists(path_label):
            os.remove(path_label)
        update_svm_label_file(seq, path_label)
        cmd = '{0} -l 0 -u 1 -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
        Popen(cmd, shell=True, stdout=PIPE).communicate()
        cmd = '{0} -b 0 "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
        Popen(cmd, shell=True).communicate()
        with open(predict_test_file) as f:
            lines = f.readlines()
        # return int(lines[1].split(' ')[0])
        return int(lines[0].strip())

    def _low_sensitivity_logic(self, sensor_db, dir_root_svm):
        # logging.info('running low sensitivity logic ...')
        # sensor_db.cur_state_idx = sensor_db.get_seq_len() - LEN_SEQ + 1  # skip ambiguous signal
        # step one evaluate anchor
        # seq_forward_pre = np.array(sensor_db.seq_forward[-2 * LEN_SEQ_LOW:-LEN_SEQ_LOW]).astype(float)
        seq_forward = np.array(sensor_db.seq_forward[-LEN_SEQ_LOW:]).astype(float)
        seq_backward = np.array(sensor_db.seq_backward[-LEN_SEQ_LOW:]).astype(float)
        idx_backward_max = np.argmax(seq_backward)
        flag_valid = True
        # if seq_backward[-1] < ALARM_LOW_TH / 2:
        #     flag_valid = False
        # sensor_db.alarm_logic_low_anchor_idx = sensor_db.get_seq_len() - 1 \
        #     if idx_backward_max == LEN_SEQ_LOW - 1 else sensor_db.alarm_logic_low_anchor_idx
        if idx_backward_max == len(seq_backward) - 1 and seq_backward[
            -1] > sensor_db.anchor_val * ALARM_LOW_ANCHOR_STEP:
            sensor_db.anchor_val = seq_backward[-1]
            sensor_db.seq_state[sensor_db.alarm_logic_low_anchor_idx] = DEBUG_ALARM_INDICATOR_VAL / 10 \
                if sensor_db.get_seq_len() - 1 - sensor_db.alarm_logic_low_anchor_idx < LEN_SEQ_LOW \
                else DEBUG_ALARM_INDICATOR_VAL / 5
            sensor_db.alarm_logic_low_anchor_idx = sensor_db.get_seq_len() - 1
            sensor_db.alarm_logic_low_probation_scores = list()  # reset
        # sensor_db.alarm_logic_low_anchor_idx = 0 \
        #     if sensor_db.alarm_logic_low_anchor_idx < sensor_db.get_seq_len() - LEN_SEQ_LOW - ALARM_LOW_SVM_WIN_LEN \
        #     else sensor_db.alarm_logic_low_anchor_idx
        sensor_db.seq_state[sensor_db.alarm_logic_low_anchor_idx] = DEBUG_ALARM_INDICATOR_VAL / 5
        if sensor_db.alarm_logic_low_anchor_idx == 0 or \
                sensor_db.get_seq_len() - sensor_db.alarm_logic_low_anchor_idx < LEN_SEQ_LOW:
            flag_valid = False
        if not flag_valid:
            sensor_db.cnt_alarm_svm = sensor_db.cnt_alarm_svm - ALARM_LOW_CNT_DECAY \
                if sensor_db.cnt_alarm_svm > 0 else sensor_db.cnt_alarm_svm
            return
        seq_pick = np.concatenate((seq_forward, seq_backward), axis=0)
        # seq_pick = seq_forward
        # seq_pick = fft_wrapper(seq_pick)
        res = self.svm_infer(seq_pick, suffix='_low', dir_libsvm=dir_root_svm)
        # rectification
        seq_diff = np.diff(seq_forward)
        seq_diff_valid = seq_diff[seq_diff > 0.]
        seq_diff_valid_mean = np.mean(seq_diff_valid) if len(seq_diff_valid) > 0 else 0
        # seq_diff_valid_mean = seq_diff_valid_mean / sensor_db.seq_forward[sensor_db.alarm_logic_low_anchor_idx]
        # res = res if seq_diff_valid_mean < ALARM_LOW_SMOOTH_TH else -1.
        # if seq_diff_valid_mean > ALARM_LOW_SMOOTH_TH and res > 0:
        #     sensor_db.seq_state[-LEN_SEQ_LOW] = -1 * DEBUG_ALARM_INDICATOR_VAL / 4
        #     res = 0
        sensor_db.seq_state_freq[-LEN_SEQ_LOW] = res * DEBUG_ALARM_INDICATOR_VAL / 4
        # sensor_db.seq_state[sensor_db.cur_state_idx] = res * 128 if res > 0 else 0
        res = res * ALARM_LOW_NEG_SCORE_WEIGHT if res < 0 else res
        sensor_db.cnt_alarm_svm = sensor_db.cnt_alarm_svm + res
        # sensor_db.seq_state[key_idx + sensor_db.cur_state_idx] = sensor_db.cnt_alarm_svm * 10
        logging.info(('low sens case svm calc info', res, sensor_db.cnt_alarm_svm))
        # if alarm_pos > sensor_db.alarm_record_last_pos + 1:
        #     sensor_db.alarm_record.append([alarm_pos, alarm_diff_abs_mean, seq_diff_valid_mean])
        alarm_pos = sensor_db.get_seq_len() - LEN_SEQ_LOW
        alarm_diff_abs_mean = np.mean(np.absolute(seq_forward - seq_backward))
        if res > 0:
            sensor_db.alarm_record = [[alarm_pos, alarm_diff_abs_mean, seq_diff_valid_mean]]
        sensor_db.alarm_record_last_pos = alarm_pos
        if sensor_db.cnt_alarm_svm > ALARM_LOW_CNT_TH_SVM:
            sensor_db.seq_state_freq[-LEN_SEQ_LOW] = DEBUG_ALARM_INDICATOR_VAL
            # alarm_pos = sensor_db.get_seq_len() - LEN_SEQ_LOW
            # alarm_diff_abs_mean = np.mean(np.absolute(seq_forward - seq_backward))
            # if alarm_pos > sensor_db.alarm_record_last_pos + 1:
            #     sensor_db.alarm_record.append([alarm_pos, alarm_diff_abs_mean])
            # sensor_db.alarm_record_last_pos = alarm_pos
        # sensor_db.cur_state_idx = sensor_db.alarm_logic_low_anchor_idx - 2 * LEN_SEQ  # seq for svm models
        # sensor_db.cur_state_idx = sensor_db.cur_state_idx if sensor_db.cur_state_idx > 0 else 0
        # idx_valid_s = int(-LEN_SEQ_LOW / 2)
        # seq_backward_diff = np.diff(seq_backward)
        # seq_backward_diff = np.diff(seq_forward)
        # seq_backward_diff_valid = np.absolute(seq_backward_diff[idx_valid_s:])
        # seq_backward_diff_valid_mean = np.mean(seq_backward_diff_valid)
        # logging.info((seq_forward[idx_valid_s:], seq_backward_diff_valid, seq_backward_diff_valid_mean))
        # particle_size_eval = np.absolute((seq_forward[-1] - seq_backward[-1]) / seq_backward[-1])
        # if particle_size_eval < 0.3 and idx_backward_max == LEN_SEQ_LOW - 1:
        #     sensor_db.seq_low_sens_score[-1] = 1. - particle_size_eval
        # seq_low_sens_score_mean = np.mean(
        #     np.array(sensor_db.seq_low_sens_score[-LEN_SEQ_LOW:]).astype(float))
        # logging.info(('lsl', key, sensor_db.get_seq_len() - 1, seq_backward[-1],
        #               particle_size_eval, sensor_db.seq_low_sens_score[-1], seq_low_sens_score_mean))
        # seq_diff_pre = np.diff(seq_forward_pre[-LEN_SEQ_LOW:int(-LEN_SEQ_LOW / 2)])
        # seq_diff_pre_valid = seq_diff_pre[seq_diff_pre < 0.]
        # seq_diff_pre_valid_mean = np.mean(seq_diff_pre_valid) if len(seq_diff_pre_valid) > 0 else 0
        # --------------------------------------------------------------------------------------------------------------
        # manual logic
        # --------------------------------------------------------------------------------------------------------------
        # seq_diff_pre_valid_mean = 0.
        # idx_valid_s = int(-LEN_SEQ_LOW / 4 * 3)
        # seq_diff = np.diff(seq_forward[idx_valid_s:])
        # seq_diff_valid = seq_diff[seq_diff > 0.]
        # seq_diff_valid_mean = np.mean(seq_diff_valid) if len(seq_diff_valid) > 0 else 0
        # alarm_low_diff_th_auto = sensor_db.seq_forward[sensor_db.alarm_logic_low_anchor_idx] / 20.
        # alarm_low_diff_th_auto = alarm_low_diff_th_auto \
        #     if alarm_low_diff_th_auto > ALARM_LOW_DIFF_TH else ALARM_LOW_DIFF_TH
        # seq_diff_valid_mean_total = seq_diff_valid_mean - seq_diff_pre_valid_mean
        # logging.info((seq_forward[idx_valid_s:], seq_diff,
        #               seq_diff_pre_valid_mean, seq_diff_valid_mean,
        #               seq_diff_valid_mean_total, alarm_low_diff_th_auto))
        # for i in range(-idx_valid_s):
        #     sensor_db.seq_state[-i] = DEBUG_ALARM_INDICATOR_VAL / 20
        # for i in range(sensor_db.cur_state_idx, sensor_db.alarm_logic_low_anchor_idx):
        #     # sensor_db.seq_state_freq[i] = DEBUG_ALARM_INDICATOR_VAL / 20
        #     sensor_db.seq_state_freq[i] = sensor_db.seq_state_freq[i] \
        #         if sensor_db.seq_state_freq[i] == DEBUG_ALARM_INDICATOR_VAL else DEBUG_ALARM_INDICATOR_VAL / 20
        # if seq_diff_valid_mean_total < alarm_low_diff_th_auto:
        #     sensor_db.seq_state[-1] = DEBUG_ALARM_INDICATOR_VAL

    def _high_sensitivity_logic(self, sensor_db, dir_root_svm, key=f'1_{1}'):
        if sensor_db.get_seq_len() < LEN_SEQ or sensor_db.seq_forward[-1] < ALARM_LOW_BASE_TH:
            return
        seq_forward = np.array(sensor_db.seq_forward[-LEN_SEQ:]).astype(float)
        seq_forward[seq_forward > ALARM_LOW_TH] = ALARM_LOW_TH
        seq_backword = np.array(sensor_db.seq_backward[-LEN_SEQ:]).astype(float)
        seq_backword[seq_backword > ALARM_LOW_TH] = ALARM_LOW_TH
        seq_pick = np.concatenate((seq_forward, seq_backword), axis=0)
        # score = self.svm_infer(seq_pick, suffix='_high', dir_libsvm=dir_root_svm)
        score = self.svm_infer(seq_pick, suffix='', dir_libsvm=dir_root_svm)
        sensor_db.seq_state_time[-1] = score * DEBUG_ALARM_INDICATOR_VAL / 4
        score = score * ALARM_NEG_SCORE_WEIGHT if score < 0 else score
        sensor_db.cnt_alarm_svm = sensor_db.cnt_alarm_svm + score
        if sensor_db.cnt_alarm_svm > ALARM_CNT_TH_SVM:
            sensor_db.seq_state_time[-1] = DEBUG_ALARM_INDICATOR_VAL

    @staticmethod
    def _alarm_guarantee_logic(sensor_db):
        sensor_db.cnt_alarm_guarantee = \
            sensor_db.cnt_alarm_guarantee + 1 if sensor_db.seq_backward[-1] > GUARANTEE_BACK_TH else 0
        if sensor_db.cnt_alarm_guarantee > ALARM_CNT_GUARANTEE_TH:
            logging.info(('guarantee alarm !', sensor_db.cnt_alarm_guarantee))
            sensor_db.seq_state_time[-1] = DEBUG_ALARM_INDICATOR_VAL
        if sensor_db.seq_backward[-1] > ALARM_GUARANTEE_SHORT_TH:
            sensor_db.seq_state_time[-1] = DEBUG_ALARM_INDICATOR_VAL

    def infer_db(self, keys, dir_root_svm):
        for key in keys:
            if key not in self.db.keys():
                return
        for key in keys:
            sensor_db = self.db[key]
            # logging.info((key, sensor_db.cur_state_idx, sensor_db.get_seq_len()))
            # if sensor_db.get_seq_len() <= LEN_SEQ:
            #     # logging.info('sensor_db.get_seq_len() <= LEN_SEQ')
            #     continue
            # while not sensor_db.cur_state_idx == sensor_db.get_seq_len():
            # while sensor_db.cur_state_idx + LEN_SEQ <= sensor_db.get_seq_len():
            self._alarm_guarantee_logic(sensor_db)
            seq_forward = np.array(sensor_db.seq_forward[-LEN_SEQ:]).astype(float)
            seq_forward_max = np.max(seq_forward)
            if seq_forward_max < ALARM_LOW_TH - 1:  # TODO: ALARM_LOW_TH
                self._high_sensitivity_logic(sensor_db, dir_root_svm, key)
            else:
                self._low_sensitivity_logic(sensor_db, dir_root_svm)
            # self._low_sensitivity_logic(sensor_db, dir_root_svm)

    @staticmethod
    def value_preprocess(val, th=255, scale=1 / 32):
        val *= scale
        val = th if val > th else val
        return val

    def update_db_v3(self, db, idx, key_forward='forward', key_backward='backward', db_key='1_1'):
        feat_forward = np.array(db[key_forward.lower()]).astype('float')[idx]
        feat_backward = np.array(db[key_backward.lower()]).astype('float')[idx]
        if db_key not in self.db.keys():
            self.db[db_key] = SensorDB()
        self.db[db_key].update([feat_forward], [feat_backward], [0], [0], [0], [0], [0], [0])
        self.db[db_key].balance()

    def update_db_v1(self, db, idx, key_forward, key_backward, db_key='1_1'):
        feat_forward = np.array(db[key_forward.lower()]).astype('float')[idx]
        feat_backward = np.array(db[key_backward.lower()]).astype('float')[idx]
        if db_key not in self.db.keys():
            self.db[db_key] = SensorDB()
        self.db[db_key].update([feat_forward], [feat_backward], [0], [0], [0], [0], [0], [0])
        self.db[db_key].balance()

    def update_db_ser_multi_amp(self):
        magnifications = (141.6488675, 68.47658, 37.79155058, 18.84206, 9.421031, 4.7105153, 2.355258, 1.177629,
                          0.588814, 0.294407, 0.1472036, 0.073602, 0.036801)
        cnt = self.ser.inWaiting()
        if cnt <= 0:
            return
        recv = self.ser.read(self.ser.in_waiting).decode()
        self.ser_buff += recv
        if not self.ser_buff[-1] == '\n':
            return
        buff_lst = self.ser_buff.split('\n')
        buff_lst_valid = [x for x in buff_lst if len(x) > MIN_SER_CHAR_NUM - 1]
        self.ser_buff = ''
        # logging.info(buff_lst_valid)
        for seq_valid in buff_lst_valid:
            seq_valid_lst = seq_valid.strip().split()
            if seq_valid_lst[4] == '08':  # frame data
                # logging.info(seq_valid_lst)
                # addr, forward, backward = seq_valid_lst[-4], seq_valid_lst[-6], seq_valid_lst[-2]
                addr, val_forward, val_backward, amp_forward, amp_backward = int(seq_valid_lst[-4], 16), int(
                    seq_valid_lst[-6], 16), int(seq_valid_lst[-2], 16), (int(seq_valid_lst[-1], 16) & 0xf0) >> 4, int(
                    seq_valid_lst[-1], 16) & 0x0f
                val_forward = val_forward / magnifications[amp_forward] * magnifications[1]
                val_backward = val_backward / magnifications[amp_backward] * magnifications[0]
                logging.info((addr, val_forward, val_backward, amp_forward, amp_backward))
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
                    self.db[db_key].update(seq_pad, seq_pad, seq_pad, seq_pad, seq_pad, seq_pad, seq_pad, seq_pad,
                                           second_total)
                # val_forward, val_backward = self.value_preprocess(val_forward), self.value_preprocess(
                #     val_backward)
                self.db[db_key].update([val_forward], [val_backward], [0], [0], [0], [amp_forward],
                                       [amp_backward], [0],
                                       second_total)
                self.db[db_key].balance()

    def update_db_ser(self):
        cnt = self.ser.inWaiting()
        if cnt > 0:
            recv = self.ser.read(self.ser.in_waiting).decode()
            self.ser_buff += recv
            if self.ser_buff[-1] == '\n':
                buff_lst = self.ser_buff.split('\n')
                buff_lst_valid = [x for x in buff_lst if len(x) > MIN_SER_CHAR_NUM - 1]
                self.ser_buff = ''
                # logging.info(buff_lst_valid)
                for seq_valid in buff_lst_valid:
                    seq_valid_lst = seq_valid.strip().split()
                    if seq_valid_lst[4] == '08':  # frame data
                        # logging.info(seq_valid_lst)
                        # addr, forward, backward = seq_valid_lst[-4], seq_valid_lst[-6], seq_valid_lst[-2]
                        addr, val_forward, val_backward = int(seq_valid_lst[-4], 16), int(seq_valid_lst[-6], 16), int(
                            seq_valid_lst[-2], 16)
                        # logging.info((addr, val_forward, val_backward))
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
                            self.db[db_key].update(seq_pad, seq_pad, seq_pad, seq_pad, seq_pad, second_total)
                        # val_forward, val_backward = self.value_preprocess(val_forward), self.value_preprocess(
                        #     val_backward)
                        self.db[db_key].update([val_forward], [val_backward], [0], [0], [0], second_total)
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

    def clear_db(self):
        self.db.clear()

    def plot_db(self, keys, pause_time_s=1, save_plot=False, path_save='/home/manu/tmp/plt.png', title_info=''):
        for key in keys:
            if key not in self.db.keys():
                return None
        plt.ion()
        for i, key in enumerate(keys):
            time_idxs = range(self.db[key].get_seq_len())
            plt.subplot(len(keys), 1, i + 1)
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_forward).astype(float), label='seq_forward')
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_backward).astype(float), label='seq_backward')
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_state).astype(float), label='seq_state')
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_state_time).astype(float), label='seq_state_time')
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_state_freq).astype(float), label='seq_state_freq')
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_forward_amp).astype(float), label='seq_forward_amp')
            plt.plot(np.array(time_idxs), np.array(self.db[key].seq_backward_amp).astype(float),
                     label='seq_backward_amp')
            # plt.plot(np.array(time_idxs), np.array(self.db[key].seq_low_sens_score).astype(float) * 5000.,
            #          label='seq_low_sens_score')
            # plt.ylim(-255, 255)
            # plt.ylim(-255, 2 ** 16)
            # plt.yticks(np.arange(0, 2 ** 16, 5000))
            plt.yticks(np.arange(0, DEBUG_ALARM_INDICATOR_VAL, DEBUG_ALARM_INDICATOR_VAL / 10))
            plt.legend()
            plt.grid()
            plt.title(title_info)
            for _record in self.db[key].alarm_record:
                pos_x, _diff_fb, _diff_f = _record
                plt.text(pos_x, int(DEBUG_ALARM_INDICATOR_VAL / 2),
                         f'{_diff_fb:.0f}\n{_diff_f:.2f}')
        button_rst = plt.waitforbuttonpress(0.001)
        # logging.info(('button_rst -> ', button_rst))
        plt.show()
        plt.pause(pause_time_s)
        if save_plot:
            plt.savefig(path_save)
        plt.clf()
        return button_rst

    def save_db(self, keys, save_idxes=(0, 100), save_dir='/home/manu/tmp'):
        for key in keys:
            if key not in self.db.keys():
                return
        for key in keys:
            save_path = os.path.join(save_dir, key + '.txt')
            seq_len = self.db[key].get_seq_len()
            with open(save_path, 'w') as f:
                f.write('forward backward \n')
                for i in range(seq_len):
                    val_forward = np.array(self.db[key].seq_forward[i]).astype('float')
                    val_backward = np.array(self.db[key].seq_backward[i]).astype('float')
                    f.write(f'{val_forward} {val_backward} \n')
