# Creator: Yunfei Luo
# Date: Apr 19, 2019  10:07 PM

import stochastic
import portfolio

import numpy as np
import time

if __name__ == '__main__':
    start = time.time()
    t = 144/252
    cev = stochastic.CEV(277.33, 0.0247, 0.1118, t, 1)

    ans3 = cev.pde(t / 1000, 0.01, 350, 285, 290, 'CN')
    print(ans3) # [3.2732151] [3.2540502] [3.26332504]  // 3.1764 // 3.1589 // 3.1483 // 3.1374 (67s) // 3.1335
    print(time.time() - start)  #3.000975
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
    # test code
    """
    r = np.array([0.10, 0.09, 0.16])
    r2 = np.array([-0.001237, 0.004848, -0.003694, 0.007403, -0.000610])
    covs = np.array([[0.09, -0.03, 0.084], [-0.03, 0.04, 0.012], [0.084, 0.012, 0.16]])
    covs2 = np.array([[0.002027, -0.000362,  0.000099, -0.000220, -0.000305],
                      [-0.000362, 0.002421, 0.000297, 0.000090, 0.000151],
                      [0.000099, 0.000297, 0.002420, 0.000020, 0.000113],
                      [-0.000220,  0.000090,  0.000020,  0.002302, 0.000047],
                      [-0.000305,  0.000151,  0.000113,  0.000047,  0.002877]])

    #mka = Markowitz(covs, returns, 0.2, None, None)
    #mk = Markowitz(covs, returns, 0.2, None, (0, 0))
    mk_test = Markowitz(covs, r, 1, bounds=((-np.inf, np.inf),), target=0.00376)
    mk_test2 = Markowitz(covs, r, 1, target=0.00376)

    w = np.array([35/50, 10/50, 5/50])
    c = np.array([[7.344, 2.015, 3.309], [2.015, 4.410, 1.202], [3.309, 1.202, 3.497]]) / 100

    p = np.array([[1, 0, 0], [-1, 1, 0]])
    q = np.array([2.5/100, 2/100])
    omega = np.array([[0.01, 0], [0, 0.015]]) ** 2

    bl = BlackLitterman(c, w, p, q, omega)
    print(bl.optimal_weights())
    """

    # [201.22, 244.88, 16.83, 259.53, 370.59, 369.38, 221.56, 1060, 100.92, 332.47]

    test_set = pd.DataFrame([[np.random.uniform(-1, 1) for _ in range(10)] for _ in range(30)])

    res = RiskParity(test_set).hrp()
    print(res)
