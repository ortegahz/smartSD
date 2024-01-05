import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

from utils.utils import set_logging


def fft_wrapper(time_seq):
    n = len(time_seq)
    res = np.abs(fft(x=time_seq, n=n)) / n * 2
    res = res[:int(n / 2)]
    return res


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', default=512)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_args()
    logging.info(args)

    fs = args.sample_rate
    f1 = 50
    f2 = 100
    t = np.linspace(0, 1, fs)
    y = 2 * np.sin(2 * np.pi * f1 * t) + 5 * np.sin(2 * np.pi * f2 * t)

    f_x = np.arange(int(fs / 2))
    f_y = fft_wrapper(y)

    plt.plot(f_x, f_y)
    plt.show()


if __name__ == '__main__':
    main()
