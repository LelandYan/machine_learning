# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/2/27 15:22'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == '__main__':
    precision = np.loadtxt("MF_precisions.txt")
    auc = np.loadtxt("MF_Aupr_values.txt")
    recalls = np.loadtxt("MF_recalls.txt")
    plt.plot(precision,label="precisions")
    plt.plot(recalls,label="recalls")
    plt.plot(auc,label='auc')
    plt.legend()
    plt.show()
