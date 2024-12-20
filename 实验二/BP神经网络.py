import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 示例失效时间点数据（实际应用中应使用真实的数据）
failure_times = [500,800,1000,1100,1210,1320,1390,1500,1630,1700,1890,1960,
                 2010,2100,2150,2230,2350,2470,2500,3000,3050,3110,3170,3230,
                 3290,3320,3350,3430,3480,3495,3540,3560,3720,3750,3795,3810,
                 3830,3855,3876,3896,3908,3920,3950,3975,3982]

# 定义延迟参数d
d = 5

# 数据预处理：归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_failure_times = scaler.fit_transform(np.array(failure_times).reshape(-1, 1)).flatten()

# 准备训练数据集
def create_dataset(data, d):
    X, y = [], []
    for i in range(len(data) - d):
        X.append(data[i:i+d])
        y.append(data[i+d])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_failure_times, d)

# 明确指定训练集和测试集的大小
train_size = 35  # 使用前35个样本作为训练集
test_size = len(X) - train_size  # 剩下的作为测试集

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 打印所有训练样本
print("Training samples (first 35 samples):")
for i in range(train_size):
    # 反归一化处理，以便打印时显示原始值
    input_values = scaler.inverse_transform(X_train[i].reshape(-1, 1)).flatten()
    output_value = scaler.inverse_transform(y_train[i].reshape(-1, 1)).flatten()[0]
    print(f'Sample {i+1}: Input={input_values}, Output={output_value}')

# 构建BP神经网络模型
model = Sequential()
model.add(Input(shape=(d,)))  # 使用Input层指定输入形状
model.add(Dense(20, activation='sigmoid')) # 隐含层
model.add(Dense(1, activation='linear'))   # 输出层

# 打印模型结构
model.summary()

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=1)

# 测试模型
predictions = model.predict(X_test)

# 反归一化处理，将预测结果转换回原始尺度
predictions_rescaled = scaler.inverse_transform(predictions)

# 打印预测结果与真实值对比
for i, pred in enumerate(predictions_rescaled):
    actual = failure_times[train_size + d + i]
    print(f'Predicted: {pred[0]:.2f}, Actual: {actual:.2f}')

# 评估模型性能
mse = mean_squared_error(y_test, predictions.flatten())
mae = mean_absolute_error(y_test, predictions.flatten())

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')

# 对整个时间序列进行预测
all_predictions_scaled = model.predict(X)
all_predictions_rescaled = scaler.inverse_transform(all_predictions_scaled)

# 绘制图表以比较预测值与实际值
plt.figure(figsize=(14, 7))
# 设置中文显示字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 绘制原始时间序列数据
plt.plot(range(d, len(failure_times)), failure_times[d:], label='真实失效时间', marker='o')

# 绘制预测的时间序列数据
plt.plot(range(d, len(failure_times)), all_predictions_rescaled, label='预测失效时间', linestyle='--', marker='x')

# 标记训练集和测试集的分界线
plt.axvline(x=train_size + d - 0.5, color='red', linestyle='--', label='训练集和测试集的分界线')

plt.xlabel('失效序号')
plt.ylabel('失效时间')
plt.title('真实失效时间和预测失效时间对比')
plt.legend()
plt.grid(True)
plt.show()