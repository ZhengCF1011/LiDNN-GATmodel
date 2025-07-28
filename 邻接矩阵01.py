import pandas as pd
from dbfread import DBF
from scipy.sparse import lil_matrix, save_npz

# 1. 读取邻接矩阵（DBF 文件）
dbf_file_path = r'D:\A毕业2025ffcb\阻力面\linjiejuzhen1.dbf'  # 替换为实际的文件路径
table = DBF(dbf_file_path, load=True)

# 转换为 pandas DataFrame
adj_matrix_df = pd.DataFrame(iter(table))

# 2. 读取阻力面数据（DBF 文件）
resistance_dbf_file_path = r'D:\A毕业2025ffcb\阻力面\zulimian.dbf'  # 替换为实际的文件路径
resistance_table = DBF(resistance_dbf_file_path, load=True)

# 转换为 pandas DataFrame
resistance_df = pd.DataFrame(iter(resistance_table))

# 查看邻接矩阵和阻力面数据的前几行
print(adj_matrix_df.head())
print(resistance_df.head())
print(adj_matrix_df.tail(10))

# 3. 创建 UID 到阻力值的字典映射
resistance_dict = dict(zip(resistance_df['UID'], resistance_df['zulizhi']))

# 4. 获取栅格数量（栅格的 UID 总数）
num_grids = len(resistance_dict)

# 创建一个空的稀疏矩阵（LIL 格式）
adj_matrix_lil = lil_matrix((num_grids, num_grids))

# 5. 遍历邻接矩阵的每个连接（UID 和 JOIN_FID）
for _, row in adj_matrix_df.iterrows():
    uid_i = row['uid']  # 当前栅格
    uid_j = row['JOIN_FID']  # 相邻栅格

    # 获取对应栅格的阻力值
    resistance_i = resistance_dict.get(uid_i, 1)  # 使用实际阻力值
    resistance_j = resistance_dict.get(uid_j, 1)  # 使用实际阻力值

    # 计算连接强度，防止除零
    epsilon = 1e-6
    if resistance_i == 0 or resistance_j == 0:
        # 如果阻力面值为 0，加上 epsilon
        resistance_value = 1 / (resistance_i + resistance_j + epsilon)
    else:
        # 如果阻力面值不为 0，直接计算
        resistance_value = 1 / (resistance_i + resistance_j)

    # 更新加权邻接矩阵的值
    adj_matrix_lil[uid_i - 1, uid_j - 1] = resistance_value  # 使用 -1 调整索引（因为 UID 从 1 开始）
    adj_matrix_lil[uid_j - 1, uid_i - 1] = resistance_value  # 无向图，更新对称位置

# 6. 将 LIL 格式转换为 CSR 格式（稀疏矩阵）
adj_matrix_csr = adj_matrix_lil.tocsr()

# 7. 将所有1000000值改为0
def replace_1000000_with_0(csr_matrix):
    # 获取所有非零元素的行和列
    rows, cols = csr_matrix.nonzero()

    # 创建一个新的稀疏矩阵来存储更新后的值
    updated_matrix = lil_matrix(csr_matrix.shape)  # 创建一个新的稀疏矩阵
    for r, c in zip(rows, cols):
        if csr_matrix[r, c] == 1000000.0:
            updated_matrix[r, c] = 0  # 替换为0
        else:
            updated_matrix[r, c] = csr_matrix[r, c]  # 保留原值
    return updated_matrix

# 替换所有1000000为0
adjusted_matrix = replace_1000000_with_0(adj_matrix_csr)

# 8. 将调整后的矩阵转换为 CSR 格式（以便保存）
adjusted_matrix_csr = adjusted_matrix.tocsr()

# 9. 保存调整后的矩阵为稀疏矩阵 .npz 文件
save_npz(r'D:\A毕业2025ffcb\阻力面\调整后的加权邻接矩阵（替换0）.npz', adjusted_matrix_csr)
print("调整后的加权邻接矩阵已成功保存！")
