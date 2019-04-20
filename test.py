# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import data_process
import numpy as np
import pandas as pd
import time


current = time.time()

# random initialized
sigma = 0.2
nu0 = 0.2
kappa = 0.5
rho = 0
theta = 0.2

s = 280.41
t = 1
r = 0.024

h = stochastic.Heston(nu0, kappa, theta, sigma, rho, s, r, t)

# Calibration test, slow

"""
df = pd.read_csv('mf796-hw5-computed.csv', usecols=[1,2,7])
n = 44
tm, km, cm = [np.array(df.iloc[:,i]).reshape(-1) for i in range(3)]
"""

n, tm, km, cm = data_process.data_import('3/1/2019')    # 2998 rows

h.calibrate(n, tm, km, cm)    # ? min

print('t', time.time() - current)

