import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
# 数据
NTDS = [9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6,
        1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91,
        2, 1, 87, 47, 12, 9, 135, 258, 16, 35]
NTDS1 = [0]
for interval in NTDS:
    NTDS1.append(NTDS1[-1] + interval)
NTDS1_subset = NTDS1[1:32]
# GO模型
def GOLikelihood2(S, b):
    N = len(S)
    SN = S[-1]
    e_bSN = np.exp(-b * SN)
    l = N / b
    r = sum(S) + N / (1 - e_bSN) * SN * e_bSN
    return l - r
def GOLikelihood1GetA(S, b):
    N = len(S)
    SN = S[-1]
    return N / (1 - np.exp(-b * SN))
def GOMTBF(a, b, t):
    return 1 / (a * b * np.exp(-b * t))
# 参数估计
ansGoB = root_scalar(lambda x: GOLikelihood2(NTDS1_subset, x), x0=0.005)
b = ansGoB.root
a = GOLikelihood1GetA(NTDS1_subset, b)
print(f"GO参数: {a=}, {b=}")
# MTBF计算
GOMTBFarr = [GOMTBF(a, b, t) for t in NTDS1_subset]
# 误差计算
error = sum(abs(NTDS[i - 1] - GOMTBFarr[i - 1]) / NTDS[i - 1] for i in range(1, 32)) / 31
print(f"GO平均误差={error:.4%}")
# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.plot([i for i in range(1, 32)], NTDS[:31], label='NTDS')
plt.plot([i for i in range(1, 32)], GOMTBFarr, label='GO', color='green')
plt.title('失效时间间隔 - GO 模型')
plt.xlabel('失效序号')
plt.ylabel('时间间隔')
plt.legend()
plt.show()