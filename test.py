# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def no_arbitrage_test(num=16):

    """
        test the no-arbitrage assumption of the SABR model
        return none, show plots of risk-neutral density
        extracted from the B method
    """

    df = pd.read_csv('parameters_price.csv')
    h = 2

    rn = np.random.choice(df.shape[0], num)

    for n in rn:

        alpha, beta, rho, sigma, _, s = df.iloc[n, 1:7]
        sabr = stochastic.SABR(sigma, alpha, beta, rho, s, 0.024, 0.25)

        sim_st = [sabr.simulate() for _ in range(10000)]
        strikes = np.linspace(s - 3 * sigma, s + 3 * sigma, num=100)
        density = np.zeros(100)

        for i, k in enumerate(strikes):

            sim_test = sim_st.copy()
            sim_test -= (k - h)
            payoff_right = np.mean(sim_test[sim_test > 0])

            sim_test -= h
            payoff_middle = np.mean(sim_test[sim_test > 0])

            sim_test -= h
            payoff_left = np.mean(sim_test[sim_test > 0])

            density[i] = (payoff_left + payoff_right - 2 * payoff_middle) / h ** 2

        plt.plot(strikes, density)
        plt.show()


def TDMA_solver(a, b, c, d, mode='div', n=4):
    if mode == 'mult':
        x = np.zeros(n)
        for i in range(-1, n-1):

            if i == -1:
                row = np.array([b[0], c[0]])
            elif i == n - 2:
                row = np.array([a[-1], b[-1]])
            else:
                row = np.array([a[i], b[i+1], c[i+1]])
            x[i+1] = row @ d[max(i, 0):min(i+3, n)]
        return x
    else:
        bc, dc = map(np.array, (b, d))

        for i in range(n - 1):
            bc[i + 1] -= a[i] / bc[i] * c[i]
            dc[i + 1] -= a[i] / bc[i] * dc[i]

        # xc = bc
        dc[-1] /= bc[-1]

        for i in range(n-2, -1, -1):
            dc[i] = (dc[i] - c[i] * dc[i+1]) / bc[i]

        return dc


if __name__ == '__main__':
    start = time.time()
    t = 144/252
    cev = stochastic.CEV(277.33, 0.0247, 0.1118, t, 1)
    #ans1 = cev.pde(t / 1000, 1, 400, 285, 290, 'EE')
    #ans2 = cev.pde(t / 1000, 0.5, 350, 285, 290, 'EI')
    ans3 = cev.pde(t / 1000, 0.1, 350, 285, 290, 'CN')
    print(ans3) # [3.2732151] [3.2540502] [3.26332504]  // 3.1764 // 3.1589 // 3.1483 // 3.1374 (286s) // 3.1335
    print(time.time() - start)
    """
    A = np.array([[10,2,0,0],[3,10,4,0],[0,1,7,5],[0,0,3,4]],dtype=float)

    a = np.array([3, 1, 3], dtype=float)
    b = np.array([10, 10, 7, 4], dtype=float)
    c = np.array([2, 4, 5], dtype=float)
    d = np.array([3, 4, 5, 6], dtype=float)

    print(TDMA_solver(a,b,c,d, 'mult'))
    print(A @ d)
    print(a)
    print(b)
    print(c)
    print(d)
    """
