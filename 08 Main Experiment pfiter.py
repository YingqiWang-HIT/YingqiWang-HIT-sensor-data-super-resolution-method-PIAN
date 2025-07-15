# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from dtaidistance import dtw

class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_noise, observation_noise):
        self.num_particles = num_particles  # 粒子数量
        self.state_dim = state_dim  # 状态的维度
        self.process_noise = process_noise  # 过程噪声
        self.observation_noise = observation_noise  # 观测噪声

        # 初始化粒子 (均匀分布)
        self.particles = torch.randn((self.num_particles, self.state_dim))
        # 初始化权重 (均匀分布)
        self.weights = torch.ones(self.num_particles) / self.num_particles

    def predict(self):
        """
        预测步骤：根据过程模型更新每个粒子的状态
        """
        noise = torch.randn(self.num_particles, self.state_dim) * self.process_noise
        self.particles += noise  # 更新粒子状态，假设过程模型为 x_t = x_t-1 + noise

    def update(self, observation):
        """
        更新步骤：根据观测值更新粒子的权重
        observation: 观测值 (torch.Tensor)
        """
        # 计算观测值与粒子的差异，计算似然
        observation = observation.unsqueeze(0).repeat(self.num_particles, 1)
        distances = torch.norm(self.particles - observation, dim=1)


        # 使用高斯分布似然函数更新权重
        epsilon = 1e-6  # 防止分母为0的一个小常数
        likelihood = torch.exp(-0.5 * (distances / (self.observation_noise + epsilon)) ** 2)
        self.weights = likelihood / (likelihood.sum() + epsilon)  # 确保权重和为1，避免极小值

        # 使用高斯分布似然函数更新权重
        # likelihood = torch.exp(-0.5 * (distances / self.observation_noise) ** 2)
        # self.weights = likelihood / likelihood.sum()

    def resample(self):
        """
        重采样步骤：根据权重重新采样粒子
        """
        # 确保权重为非负数并归一化
        self.weights = torch.clamp(self.weights, min=1e-10)
        self.weights /= (self.weights.sum() + 1e-10)  # 确保权重和为1

        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices]

    def estimate(self):
        """
        估计当前的状态 (粒子加权平均)
        """
        return torch.sum(self.particles * self.weights.unsqueeze(1), dim=0)


def read_data_from_excel(file_path_1, file_path_2):
    """
    从 Excel 文件中读取两列时间序列数据
    返回：时间序列 1 和 时间序列 2
    """
    data_1 = pd.read_excel(file_path_1)
    data_2 = pd.read_excel(file_path_2)
    series_1 = torch.tensor(data_1.iloc[:, 1].values, dtype=torch.float32)  # 第一列
    series_2 = torch.tensor(data_2.iloc[:, 1].values, dtype=torch.float32)  # 第二列
    origin_data = torch.tensor(data_2.iloc[:, 0].values, dtype=torch.float32)
    return series_1, series_2, origin_data


def run_particle_filter(file_path_1, file_path_2, output_dir, SRP, num_particles=100, process_noise=0.5, observation_noise=0.1):
    # 从 Excel 文件中读取时间序列
    series_1, series_2, origin_data = read_data_from_excel(file_path_1, file_path_2)

    # 创建粒子滤波器(通过调整state_dim调整输出的维度)
    pf = ParticleFilter(num_particles=num_particles, state_dim=1,process_noise=process_noise,
                        observation_noise=observation_noise)

    estimates = []
    # 遍历时间序列数据
    for t in range(len(series_1)):
        pf.predict()  # 预测步骤
        observation = torch.tensor([series_1[t], series_2[t]])  # 使用 series_1 和 series_2 的观测值进行更新
        pf.update(observation)  # 更新粒子权重
        pf.resample()  # 重采样
        estimate = pf.estimate()  # 估计当前状态
        estimates.append(estimate.numpy())  # 记录估计值

    estimates = np.array(estimates)

    output_excel_path = os.path.join(output_dir, f'PF_Reslut_{SRP}.xlsx')
    df = pd.DataFrame({
        'Real': origin_data,
        'series_1': series_1,
        'series_2': series_2,
        'Reconstruction': estimates[:,0]
    })

    df.to_excel(output_excel_path, index=False)
    print(f"Data saved to: {output_excel_path}")

    # 绘制结果
    plt.figure(figsize=(12, 6))
    # plt.subplot(1, 1, 1)
    plt.plot(origin_data.numpy(), label='True Series 1', color='blue')
    plt.plot(estimates[:, 0], label='Estimated Series 1', color='red', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Particle Filter Estimation for Series 1')
    plt.legend()


    plt.tight_layout()
    plt.show()

    # 假设第一列是真实值，第二列是预测值
    y_true = origin_data  # 第一列：真实值
    y_pred = estimates[:,0]  # 第二列：预测值

    # 计算 MSE
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse}")

    # 计算 RMSE
    rmse = sqrt(mse)
    print(f"RMSE: {rmse}")

    # 计算 DTW
    dtw_distance = dtw.distance(y_true, y_pred)
    print(f"DTW Distance: {dtw_distance}")


# 示例调用
file_path_1 = 'Results/CNN_1DcovCNN_MAP_Reslut_2_real_NCDE.xlsx'  # 替换为你的 Excel 文件路径
file_path_2 = './Results/CNN_1DcovCNN_MAP_Reslut_2_real.xlsx'  # 替换为你的 Excel 文件路径
output_excel_path = './Results'
SRP = 2

run_particle_filter(file_path_2, file_path_1, output_dir = output_excel_path, SRP = SRP,num_particles=50000, process_noise=0.15, observation_noise=0.10)
