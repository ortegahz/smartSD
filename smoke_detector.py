import logging
import os

from utils import LEN_SEQ

class SensorDB:
    def __init__(self, second_total):
        self.seq_forward = list()
        self.seq_backward = list()
        self.last_second_total = second_total

    def update(self, second_total, cur_forward, cur_backward):
        self.last_second_total = second_total
        self.seq_forward.extend(cur_forward)
        self.seq_backward.extend(cur_backward)

    def balance(self):
        self.seq_forward = self.seq_forward[-LEN_SEQ:]
        self.seq_backward = self.seq_backward[-LEN_SEQ:]

    def show(self):
        logging.info(('seq_forward', self.seq_forward))
        logging.info(('seq_backward', self.seq_backward))


class SmokeDetector:
    def __init__(self):
        self.db = dict()
        self.interval = 3

    def infer(self):
        pass

    def update_db(self, paths_txt):
        paths_txt_sorted = sorted(paths_txt, key=self._cmp)
        for path_txt_sorted in paths_txt_sorted:
            logging.info(path_txt_sorted)
            file_name = os.path.basename(path_txt_sorted).split('.')[0]
            group_id, sensor_id, date, time = file_name.split('-')
            group_id, date = group_id[-1], int(date)
            if not date == 20231207 or not group_id == '1' or not sensor_id == '1':
                continue
            # hour, minute, second = time.split(':')
            # hour, minute, second = int(hour), int(minute), int(second)
            # second_total = hour * 60 * 60 + minute * 60 + second
            # logging.info((group_id, sensor_id, date, hour, minute, second, second_total))
            db_key = group_id + '_' + sensor_id
            with open(path_txt_sorted) as f:
                lines = f.readlines()
            for line in lines:
                logging.info(line)
                val_forward, val_backward = int(line.split(' ')[-3], 16), int(line.split(' ')[-1], 16)
                date, time = line.split(' ')[-8], line.split(' ')[-7]
                # year, month, day = date.split('-')
                hour, minute, second = time.split(':')
                hour, minute, second = int(hour), int(minute), int(second)
                second_total = hour * 60 * 60 + minute * 60 + second
                logging.info((hour, minute, second, second_total, val_forward, val_backward))
                if db_key not in self.db.keys():
                    self.db[db_key] = SensorDB(second_total)
                elif second_total - self.db[db_key].last_second_total > LEN_SEQ:
                    self.db.pop(db_key)
                    self.db[db_key] = SensorDB(second_total)
                elif second_total - self.db[db_key].last_second_total > self.interval:
                    seq_pad = [0] * (second_total - self.db[db_key].last_second_total - self.interval)
                    self.db[db_key].update(second_total, seq_pad, seq_pad)
                self.db[db_key].update(second_total, [val_forward], [val_backward])
                self.db[db_key].balance()
                # self.show_db()

    def show_db(self):
        for key in self.db.keys():
            logging.info(key)
            self.db[key].show()

    @staticmethod
    def _cmp(item):
        file_name = os.path.basename(item).split('.')[0]
        group_id, sensor_id, date, time = file_name.split('-')
        hour, minute, second = time.split(':')
        date, hour, minute, second = int(date), int(hour), int(minute), int(second)
        second_total = hour * 60 * 60 + minute * 60 + second
        return date, second_total
