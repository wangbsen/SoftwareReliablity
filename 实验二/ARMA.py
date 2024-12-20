import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from itertools import product

# 示例数据：假设这是软件失效的时间点（单位可以是天、小时等）
failure_times = [500, 800, 1000, 1100, 1210, 1320, 1390, 1500, 1630, 1700, 1890, 1960,
                 2010, 2100, 2150, 2230, 2350, 2470, 2500, 3000, 3050, 3110, 3170, 3230,
                 3290, 3320, 3350, 3430, 3480, 3495, 3540, 3560, 3720, 3750, 3795, 3810,
                 3830, 3855, 3876, 3896, 3908, 3920, 3950, 3975, 3982]

# 分割数据为训练集和测试集
test_size = 5
train_failure_times = failure_times[:-test_size]
test_failure_times = failure_times[-test_size:]

# 将训练集的失效时间转换为间隔时间序列
train_time_intervals = np.diff(train_failure_times, prepend=0)

# 定义参数范围
p_values = range(0, 3)  # 自回归项数
d_values = range(0, 2)  # 差分次数
q_values = range(0, 3)  # 移动平均项数

# 初始化变量存储最佳参数和对应的AIC/BIC值
best_aic = float("inf")
best_bic = float("inf")
best_param_aic = None
best_param_bic = None

# 遍历所有参数组合
for p, d, q in product(p_values, d_values, q_values):
    try:
        model = sm.tsa.ARIMA(train_time_intervals, order=(p, d, q))
        results = model.fit()

        # 更新最佳AIC参数组合
        if results.aic < best_aic:
            best_aic = results.aic
            best_param_aic = (p, d, q)

        # 更新最佳BIC参数组合
        if results.bic < best_bic:
            best_bic = results.bic
            best_param_bic = (p, d, q)

    except Exception as e:
        continue  # 如果模型无法收敛，跳过该组合

print(f"Best ARIMA parameters by AIC: {best_param_aic} with AIC: {best_aic}")
print(f"Best ARIMA parameters by BIC: {best_param_bic} with BIC: {best_bic}")

# 构建并拟合最终的ARIMA模型（这里以AIC为例）
final_model = sm.tsa.ARIMA(train_time_intervals, order=best_param_aic)
results = final_model.fit()

# 打印模型摘要
print(results.summary())

# 绘制诊断图以检查残差是否为白噪声
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# 预测未来n个时间步长的失效间隔
future_steps = len(test_failure_times) - 1  # 预测接下来的几个失效间隔
forecast_intervals = results.forecast(steps=future_steps)

# 将预测的时间间隔转换回失效时间点
predicted_failure_times = np.cumsum(np.concatenate((np.array([train_failure_times[-1]]), forecast_intervals)))

# 确保预测的时间点数量与测试集中的时间点数量一致
predicted_failure_times = predicted_failure_times[-test_size:]

# 创建失效序号数组，从总长度减去测试集大小开始编号
failure_indices = np.arange(len(failure_times) - test_size + 1, len(failure_times) + 1)

# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 绘制测试数据与预测数据的对比图
plt.figure(figsize=(10, 6))

# 绘制测试数据部分
plt.plot(failure_indices, test_failure_times, label='实际失效时间', marker='o')

# 绘制预测数据部分
plt.plot(failure_indices, predicted_failure_times, label='预测失效时间', marker='x', linestyle='--')

plt.title('测试数据与预测数据对比')
plt.xlabel('失效序号')
plt.ylabel('失效时间')
plt.legend()
plt.grid(True)
plt.show()

# 计算预测误差
errors = np.abs(np.array(test_failure_times) - predicted_failure_times)
print("预测误差:", errors)