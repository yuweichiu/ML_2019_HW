# -*- coding: utf-8 -*-
"""

Created on : 2019/6/22
@author: Ivan Chiu
"""

import sys
import numpy as np
import pandas as pd
import csv

raw_data = np.genfromtxt("train_en.csv", delimiter=',') ## train.csv
data = raw_data[1:,3:]
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 0

month_to_data = {}  ## Dictionary (key:month , value:data)

for month in range(12):
    sample = np.empty(shape = (18 , 480))
    for day in range(20):
        for hour in range(24):
            sample[:,day * 24 + hour] = data[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
    month_to_data[month] = sample