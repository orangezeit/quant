# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
