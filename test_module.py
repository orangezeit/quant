# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import portfolio

import numpy as np
import pandas as pd
import time


def test_markowitz(n1=100, n2=10, n3=10000):

    cases = np.zeros(4, dtype=np.int)

    for _ in range(n1):

        a = np.random.randint(10, 11)
        b = np.random.randint(2 * a, 3 * a)
        # m1, m2 = ('equal_weights', 'exponential_weights'), (
        # 'equal_weights', 'exponential_weights', 'ledoit_wolf', 'oas')

        test_set = pd.DataFrame(np.random.normal(size=(b, a)))
        rs, covs = portfolio.estimate(test_set)

        singleton1 = portfolio.Markowitz(covs, rs, -2)
        singleton2 = portfolio.Markowitz(covs, rs, -2, bounds=((None, None),))
        singleton3 = portfolio.Markowitz(covs, rs, -2, bounds=((0, 1),))

        w1 = singleton1.allocate(-2)
        w2 = singleton2.allocate(-2)

        if (np.absolute(w1 - w2) > 1e-8).any():
            print(np.absolute(w1 - w2).max())
            cases[0] += 1

        port_cov1 = w1 @ covs @ w1

        for _ in range(n2):
            random_weights = np.random.normal(size=a)
            random_weights /= random_weights.sum()
            random_cov = random_weights @ covs @ random_weights
            if random_cov < port_cov1:
                cases[1] += 1

        w3 = singleton3.allocate(-2)
        port_cov3 = w3 @ covs @ w3
        w4 = singleton3.allocate(-1)
        port_sharpe3 = (w4 @ rs) / (w4 @ covs @ w4)

        for _ in range(n3):

            random_weights = np.absolute(np.random.normal(size=a))
            random_weights /= random_weights.sum()
            random_cov = random_weights @ covs @ random_weights
            random_sharpe = random_weights @ rs / random_cov

            if random_cov < port_cov3:
                cases[2] += 1
            if random_sharpe > port_sharpe3:
                cases[3] += 1
                print(random_sharpe, port_sharpe3)

        c = np.random.uniform(0.0, 0.2)

        w5 = singleton1.allocate(1, c)
        w6 = singleton2.allocate(1, c)

        if (np.absolute(w5 - w6) > 1e-8).any():
            print(np.absolute(w5 - w6).max())
            cases[0] += 1

    print(f'Case 1 Error: Exceed maximum tolerance in optimization. {cases[0]} / {n1 * 2} failed.')
    print(f'Case 2 Error: GMV without short constraints is incorrect. {cases[1]} / {n1 * n2} failed.')
    print(f'Case 3 Error: GMV with short constraints is incorrect. {cases[2]} / {n1 * n3} failed.')
    print(f'Case 4 Error: MSR with short constraints is incorrect. {cases[3]} / {n1 * n3} failed.')


def test_rp(n=100):

    pass


def test_stochastic_simulate_ir(n=100):

    # ou, cir
    ou = stochastic.OrnsteinUhlenbeck(0.02, 0.05, 1, 0.1, 0.2)
    for _ in range(10):
        res1 = np.fromiter((ou.simulate() for _ in range(1000)), dtype=np.float).mean()
        print(res1) # 0.037
    print()
    cir = stochastic.CoxIntergellRoss(0.02, 0.05, 1, 0.1, 0.2)
    for _ in range(10):
        res = np.fromiter((cir.simulate() for _ in range(1000)), dtype=np.float).mean()
        print(res) # 0.037


def test_stochastic_simulate_stock():

    t = 144 / 252
    cev = stochastic.CEV(277.33, 0.0247, 0.1118, t, 1)
    # res = np.array([max(cev.simulate() - 285, 0.0) for _ in range(1000)]).mean() # 7.52 7.41 7.59 7.39 7.54
    # res2 = np.array([max(285 - cev.simulate(), 0.0) for _ in range(1000)]).mean() # 11.65
    """
    res1 = cev.simulate(package=('call', 'European', 285.0, 1000))
    res2 = cev.simulate(package=('put', 'European', 285.0, 1000))
    res3 = cev.simulate(package=('call', 'American', 285.0, 1000))
    res4 = cev.simulate(package=('put', 'American', 285.0, 1000))
    """

    res1 = cev.simulate(package=('call', 'Asian-fixed', 285.0, 1000))
    res2 = cev.simulate(package=('put', 'Asian-fixed', 285.0, 1000))
    res3 = cev.simulate(package=('call', 'Asian-float', 285.0, 1000))
    res4 = cev.simulate(package=('put', 'Asian-float', 285.0, 1000))

    """
    res5 = cev.simulate(package=('call-spread', 285.0, 290.0, 1000))
    res6 = cev.simulate(package=('put-spread', 285.0, 290.0, 1000))
    res7 = cev.simulate(package=('call-spread', 285.0, 290.0, 1000), exercise='a')
    res8 = cev.simulate(package=('put-spread', 285.0, 290.0, 1000), exercise='a')
    """

    print(res1, res2, res3, res4)


def test_cev_pde(case=0):

    """ test the special case (BS-Model), check if pde solution converges to formula solution """

    if case == 0:
        t = 144 / 252
        cev = stochastic.CEV(277.33, 0.0247, 0.1118, t, 1.0)
        k1, k2 = 285.0, 290.0
    elif case == 1:
        t = 1.0
        cev = stochastic.CEV(100.0, 0.1, 0.3, t, 1)
        k1, k2 = 95.0, 105.0
    """
    ans1 = cev.pde(t / 1000, 0.01, ('call', 'European', k1))  # 7.6792 (20s)
    ans2 = cev.pde(t / 1000, 0.01, ('put', 'European', k1))  # 11.3607 (21s)

    ans5 = cev.pde(t / 1000, 0.01, ('call-spread', 'European', k1, k2))  # 1.8824 (20s)
    ans6 = cev.pde(t / 1000, 0.01, ('put-spread', 'European', k1, k2))  # 2.64845 (21s)

    ans3 = cev.pde(t / 1000, 0.01, ('call', 'American', k1))  # 7.6792
    ans4 = cev.pde(t / 1000, 0.01, ('put', 'American', k1))  # 11.9104 (21s)

    ans7 = cev.pde(t / 1000, 0.01, ('call-spread', 'American', k1, k2))  # 3.10866 (20s)
    ans8 = cev.pde(t / 1000, 0.01, ('put-spread', 'American', k1, k2))  # 4.92999 (21s)

    print(ans1, ans2, ans3, ans4)
    print(ans5, ans6, ans7, ans8)
    """
    ans6 = cev.pde(t / 1000, 0.01, ('put-spread', 'European', k1, k2))
    print(ans6)

    """
    7.68560738555255 11.361290729770904 7.6786823587981825 11.911016320988761
    1.8823261132827405 2.64853753885148 3.108537197878851 4.929993852051919
    
    19.47436149386628 5.433098144275096 19.474361504654876 5.936348291067235
    5.184298869815485 2.6868106922175428 8.046269671302197 4.801753497973709
    """


def test_bs_formula():

    bs = stochastic.BlackScholes(277.33, 0.0247, 0.1118, 144/252)
    c1 = bs.european_option_formula(285.0, 'call', 'value')
    c2 = bs.european_option_formula(290.0, 'call', 'value')
    p1 = bs.european_option_formula(285.0, 'put', 'value')
    p2 = bs.european_option_formula(290.0, 'put', 'value')
    print(c1, c1-c2, p1, p2-p1)
    # 7.6856073822393824 1.8823250330486587 11.361290726706613 3.0475992361525073

def test_heston_calibrate():
    ht = stochastic.Heston(267.15, 0.015, 0.08, 0.7, 0.1, 0.2, -0.4, 0.5, 2.0, 0.0177)
    df = pd.read_csv(r'data\opt-data.csv')
    tm = df['expT'].values
    km = df['K'].values
    cm = (df['call_bid'].values + df['call_ask'].values) / 2
    print(tm, km, cm)
    res = ht.calibrate(tm, km, cm)
    print(res)  #print(obj([ 0.06110104,  0.52289908,  0.07485918,  2.68277196, -0.55149318]))


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
    # test_markowitz(1)
    # test_stochastic_simulate()
    test_cev_pde()
    # test_bs_formula()
    # test_cev_calibrate()

    print('t', time.time() - start)

