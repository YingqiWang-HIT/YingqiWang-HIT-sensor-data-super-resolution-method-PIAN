import pandas as pd
import numpy as np

def random_zero_first_column(file_path, output_path, a):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 获取第一列
    first_column = df.iloc[:, 0]

    # 计算需要置零的元素数量
    num_zeros = int(len(first_column) * (a / 100))

    # 随机选择需要置零的索引
    zero_indices = np.random.choice(first_column.index, num_zeros, replace=False)

    # 将选中的索引对应的值置为0
    df.loc[zero_indices, df.columns[0]] = 0

    # 将修改后的数据保存为新的Excel文件
    df.to_excel(output_path, index=False)

    print(f"{a}% 的第一列数据已随机置为0，并保存到新的文件：{output_path}")


a = 10  # 比如要将30%的数据置为0
# 调用函数，设置百分比 a 为 30，例如
file_path = '.\data_set\FY-3A 飞轮E转速 降采样（×40）.xlsx'  # 输入Excel文件路径
output_path = f'.\data_set\FY-3A 飞轮E转速 降采样（×40, 缺失{a}%的数据）.xlsx'  # 输出Excel文件路径

random_zero_first_column(file_path, output_path, a)
