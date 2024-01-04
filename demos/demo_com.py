import argparse
import logging

import serial

from utils import set_logging


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def run():
    ser = serial.Serial('/dev/ttyUSB0', 115200)
    ser.flushInput()
    while True:
        cnt = ser.inWaiting()
        if cnt > 0:
            recv = ser.read(ser.in_waiting).decode()
            logging.info(recv)


def main():
    set_logging()
    args = parse_args()
    logging.info(args)
    run()


if __name__ == '__main__':
    main()
