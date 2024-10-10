import numpy as np

# 并查集类
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

# 矩阵处理函数，保持原矩阵形状，子矩阵对角排布
def process_matrix_to_diagonal(matrix):
    rows, cols = matrix.shape
    uf = UnionFind(cols)
    
    # 遍历矩阵的每一行，找到可以合并的列
    for r in range(rows):
        non_zero_cols = np.where(matrix[r, :] != 0)[0]  # 找到非零列的索引
        if len(non_zero_cols) > 1:
            # 合并非零列
            for i in range(1, len(non_zero_cols)):
                uf.union(non_zero_cols[0], non_zero_cols[i])
    
    # 根据并查集结果，将列分组
    unique_groups = {}
    for c in range(cols):
        root = uf.find(c)
        if root not in unique_groups:
            unique_groups[root] = []
        unique_groups[root].append(c)

    # 准备新的矩阵，初始为全零矩阵，保持与原矩阵同样形状
    new_matrix = np.zeros_like(matrix)

    # 对每个分组（子矩阵）处理并对角排布
    start_row = 0
    start_col = 0

    for group in unique_groups.values():
        sub_matrix = matrix[:, group]  # 提取子矩阵
        non_zero_rows = np.any(sub_matrix != 0, axis=1)  # 找到非全零行
        sub_matrix = sub_matrix[non_zero_rows, :]  # 去除全零行
        
        # 如果子矩阵不为空，进行对角线排列
        if sub_matrix.size > 0:
            rows_sub, cols_sub = sub_matrix.shape
            # 将子矩阵放到主对角线
            end_row = start_row + rows_sub
            end_col = start_col + cols_sub
            new_matrix[start_row:end_row, start_col:end_col] = sub_matrix
            # 更新起始位置，确保对角线排布
            start_row = end_row
            start_col = end_col

    return new_matrix

# 示例矩阵
matrix = np.array(
[[0, 1, 0, 1],
 [0, 0, 0, 1],
 [4, 0, 2, 0],
 [5, 0, 0, 0]])

# 处理矩阵
new_matrix = process_matrix_to_diagonal(matrix)
print("对角排列后的矩阵:")
print(new_matrix)