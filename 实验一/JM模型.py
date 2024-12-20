from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
# 数据
NTDS = [9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6,
        1, 9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91,
        2, 1, 87, 47, 12, 9, 135, 258, 16, 35]
NTDS_subset = NTDS[:31]
# 模型实现
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
# 参数估计
ansJmN0 = root_scalar(lambda x: JMLikelihood2(NTDS_subset, x), x0=30.2)
N0 = ansJmN0.root
phi = JMLikelihood1GetPhi(NTDS_subset, N0)
print(f"参数估计: {N0=}, {phi=}")
# MTBF计算
JMMTBFarr = [JMMTBF(N0, phi, i) for i in range(1, 32)]
# 误差计算
error = sum(abs(NTDS_subset[i - 1] - JMMTBFarr[i - 1]) / NTDS_subset[i - 1] for i in range(1, 32)) / 31
print(f"JM平均误差={error:.4%}")
# 绘图
# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
plt.plot([i for i in range(1, 32)], NTDS_subset, label='NTDS')
plt.plot([i for i in range(1, 32)], JMMTBFarr, label='JM')
plt.title('失效时间间隔 - JM 模型')
plt.xlabel('失效序号')
plt.ylabel('时间间隔')
plt.legend()
plt.show()