import stochastic
import numpy as np
import pandas as pd
import time


current = time.time()

sigma = 0.2
nu0 = 0.2
kappa = 0.5
rho = 0
theta = 0.2

s = 267.15
t = 1
r = 0.015-0.0177

# sigma, kappa, theta, xi, rho, s, r, t, alpha=2, q=0.0
h = stochastic.Heston(nu0, kappa, theta, sigma, rho, s, r, t)

# fft test
# payoff, strikes = h.fft()
# print(payoff[len(payoff) // 2]) # 20.6219

# Simulation test
"""
su = 0
for i in range(25000):
    su += max(h.simulate() - 282, 0) * np.exp(-0.015) # 25.41
print(su / 25000)
"""

# payoff test, work

# print(h.payoff(282))

# Calibration test, does not work


df = pd.read_csv('mf796-hw5-computed.csv', usecols=[1,2,7])
tm, km, cm = [np.array(df.iloc[:,i]).reshape(-1) for i in range(3)]

h.calibrate(44, tm, km, cm) # 26 min

print('t', time.time() - current)

