import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Model
import matplotlib.pyplot as plt

# 1. 加载CSV数据
data = pd.read_csv(r'D:\A毕业2025ffcb\lstm_biao\zongB\LSTM_1.csv')  # 使用绝对路径
 # 使用绝对路径
# 数据在LSTM_1.csv文件中

# 2. 查看数据的前几行
print(data.head())

# 3. 假设第一列是栅格ID，剩余列是每个月的NDVI数据
# 将栅格ID列去除，只保留时序数据
ndvi_data = data.drop(columns=['UID'])

# 4. 转换为numpy数组以供LSTM使用
ndvi_data = ndvi_data.values

# 5. 将数据转换为适合LSTM输入的三维格式：(样本数, 时间步长, 特征数)
X = np.reshape(ndvi_data, (ndvi_data.shape[0], ndvi_data.shape[1], 1))

# 6. 查看数据的形状
print(X.shape)  # 输出：(栅格数, 120, 1) 其中120是时间步数（12个月*10年）


# 步骤3：构建 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))  # 50个LSTM单元
model.add(Dense(1))  # 输出层（输出目标变量或时序特征）

# 8. 编译模型，使用Adam优化器和均方误差损失函数
model.compile(optimizer='adam', loss='mean_squared_error')

# 9. 训练LSTM模型（仅作为示范，不进行实际训练）
# 在这里，目标是生成时序特征，所以我们并不关心输出，只是训练网络以学习时序特征
model.fit(X, ndvi_data[:, -1], epochs=20, batch_size=32)

# 10. 获取LSTM层的时序特征
# 为了获取时序特征，我们可以通过以下方式从LSTM层输出特征
lstm_features = model.layers[0].output  # LSTM层的输出
feature_extractor = Model(inputs=model.input, outputs=lstm_features)

# 获取每个栅格的时序特征
predicted_features = feature_extractor.predict(X)

# 查看输出的特征形状
print(predicted_features.shape)  # 输出：(栅格数, 50) 50是LSTM提取的特征维度

output_features = pd.DataFrame(predicted_features, columns=[f'Feature_{i+1}' for i in range(predicted_features.shape[1])])
output_features['UID'] = data['UID']  # 保留栅格ID列

# 将时序特征保存到 CSV 文件
output_features.to_csv('predicted_features.csv', index=False)

print("时序特征已保存到 'predicted_features.csv' 文件")

