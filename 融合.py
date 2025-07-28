import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# 1. 加载数据
labels_data = pd.read_csv("D:/A毕业2025ffcb/阻力面/保护地标签.csv")  # 标签数据
dnn_predictions = pd.read_csv("B:/毕业代码/毕业文件/dnn_yuce_optimized.csv")  # DNN模型的预测结果
gat_predictions = pd.read_csv("B:/毕业代码/毕业文件/GATjieguo.csv")  # GAT模型的预测结果

# 2. 合并数据，按UID对齐x
final_data = pd.merge(dnn_predictions, labels_data[['UID', 'ZRBH_1']], on="UID", how="inner")
final_data = pd.merge(final_data, gat_predictions[['UID', 'GAT_Predicted_Value']], on="UID", how="inner")

# 3. 提取DNN和GAT模型的预测概率
dnn_probabilities = final_data['DNN_Predicted_Protection_Probability'].values  # DNN的预测结果
gat_probabilities = final_data['GAT_Predicted_Value'].values  # GAT的预测结果

# 4. 提取标签数据
y_true = final_data["ZRBH_1"].values  # 真实标签（0 或 1）

# 5. 创建新的特征矩阵，将DNN和GAT的预测结果作为特征
X_stack = np.column_stack((dnn_probabilities, gat_probabilities))

# 6. 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_stack_scaled = scaler.fit_transform(X_stack)

# 7. 使用SMOTE进行过采样，平衡数据集
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_stack_scaled, y_true)

# 8. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# 9. 定义XGBoost模型
xgb_model = XGBClassifier(scale_pos_weight=1, random_state=42)

# 10. 超参数调优：使用GridSearchCV优化XGBoost的超参数
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

# 11. 输出最佳超参数
print(f"最佳超参数: {grid_search.best_params_}")

# 使用最佳超参数的模型
best_xgb_model = grid_search.best_estimator_

# 12. 在测试集上评估优化后的模型
best_predictions = best_xgb_model.predict(X_test)

# 13. 计算准确度、精度、召回率和F1分数
accuracy = accuracy_score(y_test, best_predictions)
precision = precision_score(y_test, best_predictions)
recall = recall_score(y_test, best_predictions)
f1 = f1_score(y_test, best_predictions)

print(f"XGBoost优化模型的准确度: {accuracy}")
print(f"XGBoost优化模型的精度: {precision}")
print(f"XGBoost优化模型的召回率: {recall}")
print(f"XGBoost优化模型的F1分数: {f1}")

# 14. 保存优化后的预测结果
final_predictions = best_xgb_model.predict_proba(X_stack_scaled)[:, 1]  # 获取最终的预测概率
final_predictions_df = pd.DataFrame({
    'UID': final_data['UID'],  # 假设UID在两个文件中都存在
    'Predicted_Protection_Probability': final_predictions  # 最终优化后的预测概率
})

final_predictions_df.to_csv('final_predictions_xgb_optimized.csv', index=False)

print("优化后的预测结果已保存！")


