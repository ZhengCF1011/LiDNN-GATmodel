import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch.optim as optim
import torch.nn.init as init

# Step 1: 加载稀疏邻接矩阵
adj_matrix_data = np.load('D:/A毕业2025ffcb/阻力面/调整后的加权邻接矩阵（替换0）.npz')

# 提取 CSR 稀疏矩阵的组件
indices = adj_matrix_data['indices']
indptr = adj_matrix_data['indptr']
data = adj_matrix_data['data']
shape = adj_matrix_data['shape']

# Step 2: 加载栅格特征数据
feature_df = pd.read_csv('D:/A毕业2025ffcb/lstm_biao/zongB/GATbiao.csv')
features = feature_df.iloc[:, 1:].values  # 去除第一列UID
scaler = StandardScaler()
features = scaler.fit_transform(features)  # 特征标准化

# Step 3: 加载标签数据
labels_df = pd.read_csv('D:/A毕业2025ffcb/阻力面/保护地标签.csv')
labels = labels_df['ZRBH_1'].values  # 获取保护地标签

# Step 4: 构建图数据
edge_index = []
for row in range(shape[0]):
    start_idx = indptr[row]
    end_idx = indptr[row + 1] if row + 1 < len(indptr) else len(indices)
    for col_idx in indices[start_idx:end_idx]:
        edge_index.append([row, col_idx])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(data, dtype=torch.float)  # 边的权重

# Step 5: 强化邻接区域的特征权重，增强邻接栅格的影响力
for i in range(len(labels)):
    if labels[i] == 1:  # 如果是保护地
        # 获取与保护地栅格相邻的栅格索引
        adj_index = edge_index[1][edge_index[0] == i]  # 获取与保护地相邻的目标节点（栅格）

        # 获取当前节点i的所有边，并针对每个与i相邻的节点调整权重
        for idx in adj_index:
            # 创建布尔掩码来找出与栅格i相关的边，且目标节点为idx
            mask = (edge_index[0] == i) & (edge_index[1] == idx)

            # 更新对应边的权重，增强邻接区域的影响
            edge_weight[mask] *= 2  # 增加权重，确保邻接区域的影响

# Step 6: 构建数据对象
x = torch.tensor(features, dtype=torch.float)  # 栅格特征
y = torch.tensor(labels, dtype=torch.float)  # 标签
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


# Step 7: 定义GAT模型，增加深度和头数
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        # 增加图卷积层的深度和每层的头数
        self.gat1 = GATConv(in_channels, 128, heads=8, dropout=0.6)  # 第一层GAT
        self.gat2 = GATConv(128 * 8, 64, heads=8, dropout=0.6)  # 第二层GAT
        self.gat3 = GATConv(64 * 8, 32, heads=8, dropout=0.6)  # 第三层GAT
        self.gat4 = GATConv(32 * 8, out_channels, heads=1, dropout=0.6)  # 第四层GAT

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.gat1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat3(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.gat4(x, edge_index, edge_attr)
        return x


# 权重初始化
def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.xavier_uniform_(param)
        elif 'bias' in name:
            init.zeros_(param)


# Step 8: 定义训练过程
model = GATModel(in_channels=features.shape[1], out_channels=1)  # 输入维度为特征数量，输出为1（二分类）

# 初始化模型权重
initialize_weights(model)

# 使用Adam优化器，调整学习率
optimizer = optim.Adam(model.parameters(), lr=0.005)  # 调整学习率
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)  # 前向传播
    loss = criterion(out.squeeze(), data.y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss.item()


# Step 9: 模型训练
for epoch in range(100):
    loss = train()
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

# Step 10: 模型评估与预测
model.eval()
with torch.no_grad():
    out = model(data)  # 进行预测
    probabilities = torch.sigmoid(out).squeeze().numpy()  # 使用sigmoid将logits转换为概率值

# Step 11: 输出结果调整，给每个结果加上0.5
probabilities_adjusted = probabilities   # 给每个预测结果加上0.5

# Step 12: 保存预测结果为表格
output_df = pd.DataFrame({
    'UID': feature_df['UID'],  # 假设UID在原始特征文件的第一列
    'Predicted_Value': probabilities_adjusted  # 使用调整后的预测值
})

# 保存为CSV文件
output_df.to_csv('predictions_adjusted.csv', index=False)
print("调整后的预测结果已保存为 predictions_adjusted.csv")
