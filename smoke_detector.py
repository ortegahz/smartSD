import logging
import os


class SensorDB:
    def __init__(self):
        self.time_cur = [20231207, 0, 0, 0]  # date hour minute second
        self.seq_forward = []
        self.seq_backward = []

    def update(self, cur_forward, cur_backward):
        self.seq_forward.append(cur_forward)
        self.seq_backward.append(cur_backward)

    def show(self):
        logging.info(('seq_forward', self.seq_forward))
        logging.info(('seq_backward', self.seq_backward))


class SmokeDetector:
    def __init__(self):
        self.db = dict()

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
            hour, minute, second = time.split(':')
            hour, minute, second = int(hour), int(minute), int(second)
            second_total = hour * 60 * 60 + minute * 60 + second
            logging.info((group_id, sensor_id, date, hour, minute, second, second_total))
            db_key = group_id + '_' + sensor_id
            if db_key not in self.db.keys():
                self.db[db_key] = SensorDB()
            self.db[db_key].update(0, 0)

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
