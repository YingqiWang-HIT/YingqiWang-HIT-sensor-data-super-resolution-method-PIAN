# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline
import os

# 读取Excel数据
def load_data(file_path):
    """
    从Excel文件中加载数据
    :param file_path: Excel文件路径
    :param column_name: 要读取的列名
    :return: 数据数组
    """
    data = pd.read_excel(file_path)
    data_1D = np.squeeze(data)
    return data_1D.values

# 线性插值函数
def cubic_spline_interpolation(data, upscale_factor):
    """
    使用三次样条插值提升数据分辨率
    :param data: 原始数据
    :param upscale_factor: 分辨率提升倍数
    :return: 插值后的高分辨率数据
    """
    x_original = np.arange(len(data))  # 原始数据的索引
    x_high_res = np.linspace(0, len(data) - 1, len(data) * upscale_factor)  # 新的索引
    cs = CubicSpline(x_original, data, bc_type='natural')  # 创建三次样条插值对象
    interpolated_data = cs(x_high_res)  # 进行插值
    return interpolated_data

# 数据集划分
def split_data(data, test_size=0.2):
    """
    将数据集划分为训练集和测试集
    :param data: 原始数据
    :param test_size: 测试集比例
    :return: 训练集和测试集
    """
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    return train_data, test_data

# 模型测试函数
def evaluate_model(test_data, interpolated_data):
    """
    比较插值结果和测试数据，计算误差
    :param test_data: 测试数据（真实高分辨率数据）
    :param interpolated_data: 插值后的数据
    :return: 均方误差
    """
    mse = np.mean((test_data[:len(interpolated_data)] - interpolated_data) ** 2)  # 计算均方误差
    return mse


# 可视化结果
def plot_results(original_data, high_res_data, interpolated_data, upscale_factor, output_dir, SRP):
    """
    绘制原始数据、高分辨率数据和插值数据的对比图
    :param original_data: 原始数据 (低分辨率数据)
    :param high_res_data: 高分辨率数据
    :param interpolated_data: 插值后的数据
    :param upscale_factor: 分辨率提升倍数
    """
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.plot(np.arange(len(original_data)), original_data, 'o-', label='Original Data', markersize=5)

    # 绘制高分辨率数据
    x_high_res = np.linspace(0, len(original_data) - 1, len(high_res_data))
    plt.plot(x_high_res, high_res_data, 'x-', label='High Resolution Data', markersize=5)

    # 绘制插值数据
    x_interpolated = np.linspace(0, len(original_data) - 1, len(interpolated_data))
    plt.plot(x_interpolated, interpolated_data, '-', label=f'Interpolated Data (x{upscale_factor})')

    plt.legend()
    plt.title('Data Resolution Enhancement using Linear Interpolation')
    plt.xlabel('Sample Index')
    plt.ylabel('Data Value')
    plt.show()

    output_excel_path = os.path.join(output_dir, f'Cubic_Spline_Interpolation_Reslut_{SRP}.xlsx')
    # 将预测和标签保存到Excel
    df = pd.DataFrame({
        'Real': high_res_data,
        'Reconstruction': interpolated_data
    })
    df.to_excel(output_excel_path, index=False)
    print(f"Data saved to: {output_excel_path}")


# 主程序入口
if __name__ == "__main__":
    # 参数设置
    low_file_path = './data_set/FY-3A 飞轮E转速 降采样（×40, 缺失5%的数据）.xlsx'   # Excel文件路径
    high_file_path = './data_set/FY-3A 飞轮E转速.xlsx'   # Excel文件路径
    upscale_factor = 40  # 分辨率提升倍数
    output_excel_path = './Results'
    Split = 0.2

    # 加载数据
    data_low = load_data(low_file_path)
    data_high = load_data(high_file_path)
    data_low_Length = len(data_low) - int(Split * len(data_low))
    data_high_length = len(data_high) - int(Split * len(data_high))
    data_low_real = data_low[data_low_Length:]
    data_high_real = data_high[data_high_length:]
    # 对训练数据进行线性插值提升分辨率
    interpolated_data = cubic_spline_interpolation(data_low_real, upscale_factor)

    # 计算测试集误差
    mse = evaluate_model(data_high_real, interpolated_data)
    print(f'Mean Squared Error on Test Set: {mse}')

    # 可视化原始数据、插值数据与真实高分辨率数据的对比
    plot_results(data_low_real, data_high_real, interpolated_data, upscale_factor, output_dir = output_excel_path, SRP = upscale_factor)