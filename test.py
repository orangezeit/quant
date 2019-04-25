# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import data_process
import time


current = time.time()

# random initialized
# sigma = 0.2, nu0 = 0.2, kappa = 0.5, rho = 0, theta = 0.2
# h = stochastic.Heston(nu0, kappa, theta, sigma, rho, s, r)





# Calibration test, really fast

# h.calibrate(n, tm, km, cm, b)    # ? min




print('time', time.time() - current)

