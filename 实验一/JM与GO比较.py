import numpy
from scipy.optimize import *
import matplotlib.pyplot as plt
NTDS = [9, 12, 11,  4,  7,  2,   5,   8,  5,  7,  1,  6,
        1,  9,  4,  1,  3,  3,   6,   1, 11, 33,  7, 91,
        2,  1, 87, 47, 12,  9, 135, 258, 16, 35]
NTDS1 = [0]
for interval in NTDS:
    NTDS1.append(NTDS1[-1] + interval)
def JMLikelihood2(x, N0):
    n = len(x)
    sumX = sum(x)
    l = sum(1 / (N0 - i) for i in range(n))
    r = n / (N0 - 1 / sumX * sum(i * x[i] for i in range(n)))
    return l - r
def JMLikelihood1GetPhi(x, N0):
    n = len(x)
    sumX = sum(x)
    return n / (N0 * sumX - sum(i * x[i] for i in range(n)))
def JMMTBF(N0, phi, i):
    return 1 / (phi * (N0 - (i - 1)))
def GOLikelihood2(S, b):
    N = len(S)
    SN = S[-1]
    e_bSN = numpy.exp(-b * SN)
    l = N / b
    r = sum(S) + N / (1 - e_bSN) * SN * e_bSN
    return l - r
def GOLikelihood1GetA(S, b):
    N = len(S)
    SN = S[-1]
    return N / (1 - numpy.exp(-b * SN))
def GOMTBF(a, b, t):
    return 1 / (a * b * numpy.exp(-b * t))
def Main():
    ansJmN0 = root_scalar(lambda x: JMLikelihood2(NTDS[:31], x), x0=30.2)
    N0 = ansJmN0.root
    phi = JMLikelihood1GetPhi(NTDS[:31], ansJmN0.root)
    print(f"{N0=}, {phi=}")

    ansGoB = root_scalar(lambda x: GOLikelihood2(NTDS1[1:32], x), x0=0.005)
    b = ansGoB.root
    a = GOLikelihood1GetA(NTDS1[1:32], ansGoB.root)
    print(f"{a=}, {b=}")

    JMMTBFarr = [JMMTBF(N0, phi, i) for i in range(1, 32)]
    GOMTBFarr = [GOMTBF(a, b, NTDS1[i]) for i in range(1, 32)]
    print("i JMMTBF GOMTBF")
    for i in range(1, 32):
        print(f"{i} {JMMTBFarr[i - 1]:.4f} {GOMTBFarr[i - 1]:.4f}")

    error = 0
    for i in range(1, 32):
        e = abs(NTDS[i - 1] - JMMTBFarr[i - 1]) / NTDS[i - 1]
        print(f"{i} {e:.4f}")
        error += e
    error /= 31
    print(f"JM {error=:%}")

    error = 0
    for i in range(1, 32):
        error += abs(NTDS[i - 1] - GOMTBFarr[i - 1]) / NTDS[i - 1]
    error /= 31
    print(f"GO {error=:%}")

    x = [i for i in range(1, 32)]
    plt.plot(x, NTDS[:31], label='NTDS')
    plt.plot(x, JMMTBFarr, label='JM')
    plt.plot(x, GOMTBFarr, label='GO', color = 'green')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('失效时间间隔')
    plt.xlabel('失效序号')
    plt.ylabel('时间间隔')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Main()