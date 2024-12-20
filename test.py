import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import ttest_rel

# 示例失效时间点数据（实际应用中应使用真实的数据）
failure_times = [500, 800, 1000, 1100, 1210, 1320, 1390, 1500, 1630, 1700, 1890, 1960,
                 2010, 2100, 2150, 2230, 2350, 2470, 2500, 3000, 3050, 3110, 3170, 3230,
                 3290, 3320, 3350, 3430, 3480, 3495, 3540, 3560, 3720, 3750, 3795, 3810,
                 3830, 3855, 3876, 3896, 3908, 3920, 3950, 3975, 3982]

# 分割数据为训练集和测试集
test_size = 5
train_failure_times = failure_times[:-test_size]
test_failure_times = failure_times[-test_size:]

# 定义延迟参数d (用于BP神经网络)
d = 5

# 数据预处理：归一化处理（用于BP神经网络）
scaler_bp = MinMaxScaler(feature_range=(0, 1))
scaled_failure_times_bp = scaler_bp.fit_transform(np.array(failure_times).reshape(-1, 1)).flatten()

# 准备BP神经网络训练数据集
def create_dataset(data, d):
    X, y = [], []
    for i in range(len(data) - d):
        X.append(data[i:i+d])
        y.append(data[i+d])
    return np.array(X), np.array(y)

X_bp, y_bp = create_dataset(scaled_failure_times_bp, d)

# 明确指定训练集和测试集的大小（用于BP神经网络）
train_size_bp = len(train_failure_times) - d  # 使用前len(train_failure_times)-d个样本作为训练集
test_size_bp = len(X_bp) - train_size_bp  # 剩下的作为测试集

X_train_bp, X_test_bp = X_bp[:train_size_bp], X_bp[train_size_bp:]
y_train_bp, y_test_bp = y_bp[:train_size_bp], y_bp[train_size_bp:]

# 构建BP神经网络模型
model_bp = Sequential()
model_bp.add(Input(shape=(d,)))  # 使用Input层指定输入形状
model_bp.add(Dense(20, activation='sigmoid'))  # 隐含层
model_bp.add(Dense(1, activation='linear'))   # 输出层

# 编译模型
model_bp.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
history_bp = model_bp.fit(X_train_bp, y_train_bp, epochs=1000, batch_size=1, verbose=0)

# 测试BP神经网络模型
predictions_bp_scaled = model_bp.predict(X_test_bp)

# 反归一化处理，将BP神经网络预测结果转换回原始尺度
predictions_bp_rescaled = scaler_bp.inverse_transform(predictions_bp_scaled)

# 将训练集的失效时间转换为间隔时间序列（用于ARIMA）
train_time_intervals = np.diff(train_failure_times, prepend=0)

# 定义参数范围（用于ARIMA）
p_values = range(0, 3)  # 自回归项数
d_values = range(0, 2)  # 差分次数
q_values = range(0, 3)  # 移动平均项数

# 初始化变量存储最佳参数和对应的AIC/BIC值（用于ARIMA）
best_aic = float("inf")
best_bic = float("inf")
best_param_aic = None
best_param_bic = None

# 遍历所有参数组合（用于ARIMA）
for p, d, q in product(p_values, d_values, q_values):
    try:
        model_arima = ARIMA(train_time_intervals, order=(p, d, q))
        results_arima = model_arima.fit()

        # 更新最佳AIC参数组合
        if results_arima.aic < best_aic:
            best_aic = results_arima.aic
            best_param_aic = (p, d, q)

        # 更新最佳BIC参数组合
        if results_arima.bic < best_bic:
            best_bic = results_arima.bic
            best_param_bic = (p, d, q)

    except Exception as e:
        continue  # 如果模型无法收敛，跳过该组合

print(f"Best ARIMA parameters by AIC: {best_param_aic} with AIC: {best_aic}")
print(f"Best ARIMA parameters by BIC: {best_param_bic} with BIC: {best_bic}")

# 构建并拟合最终的ARIMA模型（这里以AIC为例）
final_model_arima = ARIMA(train_time_intervals, order=best_param_aic)
results_arima = final_model_arima.fit()

# 预测未来n个时间步长的失效间隔（用于ARIMA）
future_steps = test_size - 1  # 预测接下来的几个失效间隔
forecast_intervals_arima = results_arima.forecast(steps=future_steps)

# 将预测的时间间隔转换回失效时间点（用于ARIMA）
predicted_failure_times_arima = np.cumsum(np.concatenate((np.array([train_failure_times[-1]]), forecast_intervals_arima)))

# 确保预测的时间点数量与测试集中的时间点数量一致（用于ARIMA）
predicted_failure_times_arima = predicted_failure_times_arima[-test_size:]

# 创建失效序号数组，从总长度减去测试集大小开始编号
failure_indices_test = np.arange(len(failure_times) - test_size + 1, len(failure_times) + 1)

# 计算预测误差
errors_bp = np.abs(np.array(test_failure_times) - predictions_bp_rescaled.flatten())
errors_arima = np.abs(np.array(test_failure_times) - predicted_failure_times_arima)

# 计算MSE和MAE
mse_bp = mean_squared_error(test_failure_times, predictions_bp_rescaled.flatten())
mae_bp = mean_absolute_error(test_failure_times, predictions_bp_rescaled.flatten())

mse_arima = mean_squared_error(test_failure_times, predicted_failure_times_arima)
mae_arima = mean_absolute_error(test_failure_times, predicted_failure_times_arima)

print("BP神经网络预测误差:")
print(f"Mean Squared Error (MSE): {mse_bp:.4f}")
print(f"Mean Absolute Error (MAE): {mae_bp:.4f}")

print("\nARIMA模型预测误差:")
print(f"Mean Squared Error (MSE): {mse_arima:.4f}")
print(f"Mean Absolute Error (MAE): {mae_arima:.4f}")

# 配对t检验
t_statistic, p_value = ttest_rel(errors_bp, errors_arima)

print(f"\n配对t检验结果: T-statistic={t_statistic}, P-value={p_value}")
if p_value < 0.05:
    print("存在显著性差异")
else:
    print("不存在显著性差异")

# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 绘制对比图
plt.figure(figsize=(14, 7))

# 绘制原始时间序列数据（训练部分）
plt.plot(range(1, len(train_failure_times) + 1), train_failure_times, label='历史失效时间', marker='o')

# 绘制测试数据
plt.plot(failure_indices_test, test_failure_times, label='测试失效时间', marker='o', color='green')

# 绘制BP神经网络预测的数据
plt.plot(failure_indices_test, predictions_bp_rescaled.flatten(), label='BP神经网络预测', linestyle='--', marker='x', color='blue')

# 绘制ARIMA模型预测的数据
plt.plot(failure_indices_test, predicted_failure_times_arima, label='ARIMA模型预测', linestyle='-.', marker='+', color='red')

# 标记训练集和测试集的分界线
plt.axvline(x=len(train_failure_times) + 0.5, color='purple', linestyle='--', label='训练集和测试集的分界线')

plt.xlabel('失效序号')
plt.ylabel('失效时间')
plt.title('真实失效时间与两种模型预测结果对比')
plt.legend()
plt.grid(True)
plt.show()