# Author: Yunfei Luo
# Date: Mar 19, 2020
# Version: 0.0.2


import numpy as np
import pandas as pd
import time

import stochastic
import portfolio


def test_est():

    a = np.random.randint(10, 20)
    b = np.random.randint(2 * a, 3 * a)
    m1 = ('equal_weights', 'exponential_weights', 'linear-weights')
    m2 = ('equal_weights', 'exponential_weights', 'ledoit_wolf', 'oas')
    test_set = pd.DataFrame(np.random.normal(size=(b, a)))
    for x in m1:
        for y in m2:
            rs = portfolio.estimate(test_set, x, y)
            print(rs)


def test_markowitz(n1=100, n2=1000, n3=1000):

    cases = np.zeros(4, dtype=np.int)

    for _ in range(n1):

        # simulated data set
        a = np.random.randint(10, 20)
        b = np.random.randint(2 * a, 3 * a)

        test_set = pd.DataFrame(np.random.normal(size=(b, a)))
        rs, covs = portfolio.estimate(test_set)

        '''
            singleton1: no short constraints
            singleton2: short constraints, (-inf, inf)
                compare singleton1 with singleton2
                measure the efficiency of optimization algorithm (scipy.optimize.minimize)
            singleton3: short constraints, (0, 1) -> non-negative
        '''

        singleton1 = portfolio.Markowitz(covs, rs)
        singleton2 = portfolio.Markowitz(covs, rs, bounds=((None, None),))
        singleton3 = portfolio.Markowitz(covs, rs, bounds=((0, 1),))

        ''' Without short constraints '''

        w1 = singleton1.allocate('GMV')
        w2 = singleton2.allocate('GMV')

        if (np.absolute(w1 - w2) > 1e-8).any():
            print(np.absolute(w1 - w2).max())
            cases[0] += 1

        port_cov1 = w1 @ covs @ w1

        for _ in range(n2):
            random_weights = np.random.normal(size=a)
            random_weights /= random_weights.sum()
            random_cov = random_weights @ covs @ random_weights

            # check if variance is minimum among other portfolios
            if random_cov < port_cov1:
                cases[1] += 1

        ''' With short constraints '''

        w3 = singleton3.allocate('GMV')
        port_cov3 = w3 @ covs @ w3
        w4 = singleton3.allocate('MSR')
        port_sharpe3 = (w4 @ rs) / (w4 @ covs @ w4)

        for _ in range(n3):

            random_weights = np.absolute(np.random.normal(size=a))
            random_weights /= random_weights.sum()
            random_cov = random_weights @ covs @ random_weights
            random_sharpe = random_weights @ rs / random_cov

            # check if variance is minimum among other portfolios
            if random_cov < port_cov3:
                cases[2] += 1
            # check if sharpe is maximum among other portfolios
            if random_sharpe > port_sharpe3:
                cases[3] += 1
                print(random_sharpe, port_sharpe3)

        ''' Portfolio with expected returns '''

        er = np.random.uniform(0.0, 0.2)

        w5 = singleton1.allocate('opt-ret', er)
        w6 = singleton2.allocate('opt-ret', er)

        if (np.absolute(w5 - w6) > 1e-8).any():
            print(np.absolute(w5 - w6).max())
            cases[0] += 1

    print(f'Case 1 Error: Exceed maximum tolerance in optimization. {cases[0]} / {n1 * 2} failed.')
    print(f'Case 2 Error: GMV without short constraints is incorrect. {cases[1]} / {n1 * n2} failed.')
    print(f'Case 3 Error: GMV with short constraints is incorrect. {cases[2]} / {n1 * n3} failed.')
    print(f'Case 4 Error: MSR with short constraints is incorrect. {cases[3]} / {n1 * n3} failed.')


def test_rp(n=100):

    pass


def test_stochastic_simulate_ir(times=10000):

    # ou, cir
    ou = stochastic.OrnsteinUhlenbeck(0.02, 0.05, 1, 0.1, 0.2)
    res1 = np.fromiter((ou.simulate() for _ in range(times)), dtype=np.float).mean()
    print(res1)  # 0.037
    print()
    cir = stochastic.CoxIntergellRoss(0.02, 0.05, 1, 0.1, 0.2)
    res = np.fromiter((cir.simulate() for _ in range(times)), dtype=np.float).mean()
    print(res)   # 0.037


def test_stochastic_simulate_stock(model=0):

    if model == 0:
        cev = stochastic.CEV(292.45, 0.0236, 0.12, 84 / 252, 1.0)
    elif model == 1:
        cev = stochastic.CEV(120.0, 0.04, 0.3, 0.5, 1.0)

    options = ['call', 'put', 'call-spread', 'put-spread', 'call-binary', 'put-binary']  #
    exercises = ['vanilla']  # 'lookback-fixed', 'lookback-float' 'Asian-fixed', 'Asian-float'
    styles = ['American']    # 'European'

    for option in options:
        for exercise in exercises:
            if option[-1] == 'd':
                if exercise[-1] == 't':
                    continue
                strikes = (310.0, 315.0)
            else:
                strikes = (310.0,)
            for style in styles:
                res = cev.simulate(n=1000, pack=(option, exercise, style, *strikes, 10000, None, None))
                print(option, exercise, style, res)


def test_cev_pde(case=2):

    """ test the special case (BS-Model), check if pde solution converges to formula solution """

    if case == 0:
        t = 144 / 252
        cev = stochastic.CEV(277.33, 0.0247, 0.1118, t, 1.0)
        k1, k2 = 285.0, 290.0
    elif case == 1:
        t = 1.0
        cev = stochastic.CEV(100.0, 0.1, 0.3, t, 1.0)
        k1, k2 = 95.0, 105.0
    elif case == 2:
        t = 84 / 252
        cev = stochastic.CEV(292.45, 0.0236, 0.12, 84 / 252, 1.0)
        k1, k2 = 310.0, 315.0

    d = {1: cev.pde(t / 1000, 0.01, ('call', 'European', k1)),
         2: cev.pde(t / 1000, 0.01, ('put', 'European', k1)),
         3: cev.pde(t / 1000, 0.01, ('call-spread', 'European', k1, k2)),
         4: cev.pde(t / 1000, 0.01, ('put-spread', 'European', k1, k2)),
         5: cev.pde(t / 1000, 0.01, ('call-binary', 'European', k1)),
         6: cev.pde(t / 1000, 0.01, ('put-binary', 'European', k1)),
         7: cev.pde(t / 1000, 0.01, ('call', 'American', k1)),
         8: cev.pde(t / 1000, 0.01, ('put', 'American', k1)),
         9: cev.pde(t / 1000, 0.01, ('call-spread', 'American', k1, k2)),
         10: cev.pde(t / 1000, 0.01, ('put-spread', 'American', k1, k2)),
         11: cev.pde(t / 1000, 0.01, ('call-binary', 'American', k1)),
         12: cev.pde(t / 1000, 0.01, ('put-binary', 'American', k1))}

    for v in d.values():
        print(v)
    """
    7.68560738555255 11.361290729770904 7.6786823587981825 11.911016320988761
    1.8823261132827405 2.64853753885148 3.108537197878851 4.929993852051919

    19.47436149386628 5.433098144275096 19.474361504654876 5.936348291067235
    5.184298869815485 2.6868106922175428 8.046269671302197 4.801753497973709
    """


def test_bs_formula():

    bs = stochastic.BlackScholes(277.33, 0.0247, 0.1118, 144/252)
    c1 = bs.european_vanilla_option_formula(285.0, 'call', 'value')
    c2 = bs.european_vanilla_option_formula(290.0, 'call', 'value')
    p1 = bs.european_vanilla_option_formula(285.0, 'put', 'value')
    p2 = bs.european_vanilla_option_formula(290.0, 'put', 'value')

    cb1 = bs.european_vanilla_option_formula(280.0, 'call', 'value')
    b1 = bs.european_barrier_option_formula(280.0, 300.0, 0.0, 'up-out-call', 'value') # 0.0070

    print(c1, c1-c2, p1, p2-p1)
    print(cb1, b1)  # 7.6856073822393824 1.8823250330486587 11.361290726706613 3.0475992361525073


def test_heston_calibrate():

    ht = stochastic.Heston(267.15, 0.015, 0.08, 0.7, 0.1, 0.2, -0.4, 0.5, 2.0, 0.0177)
    df = pd.read_csv(r'data\opt-data.csv')
    tm = df['expT'].values
    km = df['K'].values
    cm = (df['call_bid'].values + df['call_ask'].values) / 2
    print(tm, km, cm)
    res = ht.calibrate(tm, km, cm)
    print(res)  # [ 0.06110104,  0.52289908,  0.07485918,  2.68277196, -0.55149318]


def test_cev_calibrate():

    t = 144 / 252
    cev = stochastic.CEV(277.33, 0.0247, 0.1118, t, 1.0)
    df = pd.read_csv(r'data\opt-data.csv')

    # expiry, strike, call price, put price
    tm = df['expT'].values
    km = df['K'].values
    cm = (df['call_bid'].values + df['call_ask'].values) / 2
    pm = (df['put_bid'].values + df['put_ask'].values) / 2

    res = cev.calibrate(tm, km, cm, pm)
    print(res)


if __name__ == '__main__':

    start = time.time()
    # test_est()
    # test_markowitz()
    # test_stochastic_simulate_ir()
    # test_stochastic_simulate_stock(0)
    # test_cev_pde(2)
    # test_bs_formula()
    # test_cev_calibrate()
    # a = cp.exp(1)

    print('t', time.time() - start)
