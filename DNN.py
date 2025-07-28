from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

def train_dnn_model(dnn_data_path, lstm_data_path, labels_data_path, output_path, model_save_path="dnn_model.h5"):
    # 1. 加载数据
    dnn_data = pd.read_csv(dnn_data_path)  # DNN特征数据
    lstm_data = pd.read_csv(lstm_data_path)  # LSTM特征数据
    labels_data = pd.read_csv(labels_data_path)  # 标签数据

    # 2. 合并栅格特征数据和标签数据
    final_data = pd.merge(dnn_data, labels_data[['UID', 'ZRBH_1']], on="UID", how="inner")

    # 3. 合并DNN数据和LSTM数据，按UID合并
    final_data = pd.merge(final_data, lstm_data, on="UID", how="inner")

    # 4. 提取特征和标签
    X = final_data.drop(columns=["UID", "ZRBH_1"])  # 特征数据，去除UID和标签列
    y = final_data["ZRBH_1"]  # 标签数据（0 或 1）

    # 5. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. 划分训练集和测试集（80%用于训练，20%用于测试）
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 7. 定义DNN模型，增加更多层数和神经元
    dnn_model = Sequential([
        Dense(256, activation='relu', input_dim=X_train.shape[1]),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    # 8. 编译DNN模型，优化器使用Adam
    dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 9. 定义早停回调，避免过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 10. 训练模型
    dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # 11. 在测试集上评估模型性能
    test_loss, test_acc = dnn_model.evaluate(X_test, y_test)
    print(f"测试集上的损失：{test_loss}, 测试集上的准确率：{test_acc}")

    # 12. 预测所有栅格的结果（包括已经是保护地的栅格）
    dnn_predictions_all = dnn_model.predict(X_scaled)

    # 将预测结果与栅格UID结合
    final_predictions = pd.DataFrame({
        "UID": final_data["UID"],  # 所有栅格的UID
        "Predicted_Protection_Probability": dnn_predictions_all.flatten()  # 所有栅格的预测概率
    })

    # 13. 保存预测结果（包含保护地的概率以及最终预测值）
    final_predictions.to_csv(output_path, index=False)

    # 14. 保存训练好的模型
    dnn_model.save(model_save_path)
    print(f"模型已保存至 {model_save_path}")

    # 返回训练好的模型和训练数据X_train
    return dnn_model, X_train



if __name__ == "__main__":
    dnn_data_path = "D:/A毕业2025ffcb/lstm_biao/zongB/GATbiao.csv"
    lstm_data_path = "D:/A毕业2025ffcb/lstm_biao/zongB/LSTMjieguo.csv"
    labels_data_path = "D:/A毕业2025ffcb/阻力面/保护地标签.csv"
    output_path = "dnn_yuce_optimized.csv"
    model_save_path = "dnn_model.h5"  # 保存模型的路径

    # 调用训练函数并保存模型
    dnn_model, X_train = train_dnn_model(dnn_data_path, lstm_data_path, labels_data_path, output_path, model_save_path)