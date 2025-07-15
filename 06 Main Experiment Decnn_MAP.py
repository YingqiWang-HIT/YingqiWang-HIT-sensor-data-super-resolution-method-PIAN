# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 1. 数据准备 (Data Preparation)

def prepare_data(low_res_path, high_res_path, SRP, window_size, stride, batch_size, test_split=0.2):
    """
    从Excel文件中读取低分辨率和高分辨率数据，并划分训练集和测试集。

    参数:
    - low_res_path: 低分辨率数据的Excel文件路径
    - high_res_path: 高分辨率数据的Excel文件路径
    - SRP: 超分辨因子
    - window_size: 代表窗长度，也就是样本的大小
    - stride: 代表窗的滑动位置，也就是某个样本的第一个位置
    - test_split: 测试集所占比例 (默认 20%)

    返回:
    - train_loader: 训练集的数据加载器
    - test_loader: 测试集的数据加载器
    """

    # 读取数据
    low_res_data = pd.read_excel(low_res_path)
    high_res_data = pd.read_excel(high_res_path)

    # 划分样本
    Sample_lows = []
    Sample_highs = []
    for i in range(0, len(low_res_data) - window_size//SRP + 1, stride//SRP):
        Sample_low = low_res_data[i:i + window_size//SRP]
        Sample_lows.append(Sample_low)
    for i in range(0, len(high_res_data) - window_size + 1, stride):
        Sample_high = high_res_data[i:i + window_size]
        Sample_highs.append(Sample_high)
    Sample_lows = np.array(Sample_lows)
    Sample_highs = np.array(Sample_highs)
    Sample_lows = Sample_lows.reshape(Sample_lows.shape[0], 1, Sample_lows.shape[1])
    Sample_highs = Sample_highs.reshape(Sample_highs.shape[0], 1, Sample_highs.shape[1])
    Sample_lows = torch.tensor(Sample_lows, dtype=torch.float32).to(device)
    Sample_highs = torch.tensor(Sample_highs, dtype=torch.float32).to(device)

    # 确保低分辨率和高分辨率样本数量相同
    assert Sample_lows.shape[0] == Sample_highs.shape[0], "Low resolution and high resolution data sizes do not match."

    # 计算训练集和测试集的大小
    dataset_size = len(Sample_highs)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    # 创建数据集
    Sample_lows_train = Sample_lows[:train_size, :]
    Sample_highs_train = Sample_highs[:train_size, :]
    train_dataset = TensorDataset(Sample_lows_train, Sample_highs_train)
    Sample_lows_test = Sample_lows[train_size:, :]
    Sample_highs_test = Sample_highs[train_size:, :]
    test_dataset = TensorDataset(Sample_lows_test, Sample_highs_test)
    # 划分数据集

    # 创建数据加载器zai zaide shengde shen
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)#随机种子，让模型的训练从不同的位置开始
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 2. 模型构建 (Model Construction)

class Cnn_CovCnn_Model(nn.Module):
    """
    CNN+DeCNN模型，CNN用于特征提取，DeCNN用于分辨率提升。

    参数:
    - input_channels: 输入通道数
    - output_channels: 输出通道数
    - kernel_size: 卷积核大小
    - feature_channels: 特征提取的通道数
    - stride: 反卷积的步幅，用于控制分辨率提升倍数
    """

    def __init__(self, input_channels, output_channels, kernel_size, feature_channels, stride_1,stride_2,stride_3):
        super(Cnn_CovCnn_Model, self).__init__()
        # Cnn部分：特征提取
        self.conv1 = nn.Conv1d(input_channels, 8, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(8, 64, kernel_size, padding=kernel_size // 2)
        # self.conv3 = nn.Conv1d(16, 32, kernel_size, padding=kernel_size // 2)
        # self.conv4 = nn.Conv1d(32, 64, kernel_size, padding=kernel_size // 2)
        # self.conv5 = nn.Conv1d(64, 128, kernel_size, padding=kernel_size // 2)
        # self.conv6 = nn.Conv1d(128, 256, kernel_size, padding=kernel_size // 2)
        self.conv7 = nn.Conv1d(64, 128, kernel_size, padding=kernel_size // 2)
        self.deconv_1 = nn.ConvTranspose1d(128, 128, kernel_size, stride=stride_1,
                                         padding=(kernel_size - 1) // 2, output_padding=stride_1 - 1)
        self.conv8 = nn.Conv1d(128, 256, kernel_size, padding=kernel_size // 2)
        # self.conv9 = nn.Conv1d(1024, 2048, kernel_size, padding=kernel_size // 2)
        # self.conv10 = nn.Conv1d(2048, 5096, kernel_size, padding=kernel_size // 2)
        # self.conv11 = nn.Conv1d(5096, 5096, kernel_size, padding=kernel_size // 2)
        # self.conv12 = nn.Conv1d(5096, 2048, kernel_size, padding=kernel_size // 2)
        # self.conv13 = nn.Conv1d(2048, 1024, kernel_size, padding=kernel_size // 2)
        # self.conv14 = nn.Conv1d(1024, 1024, kernel_size, padding=kernel_size // 2)
        self.deconv_2 = nn.ConvTranspose1d(256, 256, kernel_size, stride=stride_2,
                                         padding=(kernel_size - 1) // 2, output_padding=stride_2 - 1)
        self.conv15 = nn.Conv1d(256, 128, kernel_size, padding=kernel_size // 2)
        # self.conv16 = nn.Conv1d(512, 256, kernel_size, padding=kernel_size // 2)
        # self.conv17 = nn.Conv1d(256, 256, kernel_size, padding=kernel_size // 2)
        # CovCnn部分：反卷积用于分辨率提升
        self.deconv_3 = nn.ConvTranspose1d(128, 128, kernel_size, stride=stride_3,
                                         padding=(kernel_size - 1) // 2, output_padding=stride_3 - 1)
        # self.conv18 = nn.Conv1d(256, 128, kernel_size, padding=kernel_size // 2)
        self.conv19 = nn.Conv1d(128, 64, kernel_size, padding=kernel_size // 2)
        self.conv20 = nn.Conv1d(64, 8, kernel_size, padding=kernel_size // 2)
        # self.conv21 = nn.Conv1d(32, 16, kernel_size, padding=kernel_size // 2)
        # self.conv22 = nn.Conv1d(16, 8, kernel_size, padding=kernel_size // 2)
        self.conv23 = nn.Conv1d(8, output_channels, kernel_size, padding=kernel_size // 2)

        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.4)
        self._initialize_weights()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        # x = self.leaky_relu(self.conv3(x))
        # x = self.leaky_relu(self.conv4(x))
        # x = self.leaky_relu(self.conv5(x))
        # x = self.leaky_relu(self.conv6(x))
        x = self.leaky_relu(self.conv7(x))
        x = self.leaky_relu(self.deconv_1(x))
        x = self.leaky_relu(self.conv8(x))
        # x = self.leaky_relu(self.conv9(x))
        # x = self.leaky_relu(self.conv10(x))
        # x = self.leaky_relu(self.conv11(x))
        # x = self.leaky_relu(self.conv12(x))
        # x = self.leaky_relu(self.conv13(x))
        # x = self.leaky_relu(self.conv14(x))
        x = self.leaky_relu(self.deconv_2(x))
        x = self.leaky_relu(self.conv15(x))
        # x = self.leaky_relu(self.conv16(x))
        # x = self.leaky_relu(self.conv17(x))
        x = self.leaky_relu(self.deconv_3(x))
        # x = self.leaky_relu(self.conv18(x))
        x = self.leaky_relu(self.conv19(x))
        x = self.leaky_relu(self.conv20(x))
        # x = self.leaky_relu(self.conv21(x))
        # x = self.leaky_relu(self.conv22(x))
        x = self.leaky_relu(self.conv23(x))
        return x

    # 权重初始化函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')  # He 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# 3. MAP框架下的模型训练 (Training the Model in MAP Framework)

def train_model(model, train_loader, criterion, optimizer, num_epochs=1000):
    """
    在MAP框架下训练模型，并记录训练损失。

    参数:
    - model: 待训练的模型
    - train_loader: 训练数据加载器
    - criterion: 损失函数
    - optimizer: 优化器
    - num_epochs: 训练轮数

    返回:
    - train_losses: 每个epoch的训练损失
    """
    model.to(device)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失

            # MAP框架：加入正则化项
            map_loss = loss + 0.01 * torch.sum(torch.abs(torch.cat([param.view(-1) for param in model.parameters()])))

            map_loss.backward()  # 反向传播
            # loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        if epoch % 100 == 0:
            train_losses.append(running_loss / len(train_loader))
            plt.figure(figsize=(10, 5))
            plt.plot(outputs[0].squeeze().detach().cpu().numpy(), label="Super-Resolution Output")
            plt.plot(targets[0].squeeze().detach().cpu().numpy(), label="Ground Truth")
            plt.title("Train")
            plt.legend()
            plt.show()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    return train_losses


# 4. 模型评估和结果可视化 (Model Evaluation and Visualization)

def evaluate_and_visualize(model, test_loader, train_losses, output_dir, SRP):
    """
    评估模型性能，并绘制结果对比图和损失曲线。

    参数:
    - model: 训练好的模型
    - test_loader: 测试数据加载器
    - train_losses: 训练过程中记录的损失
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_excel_path = os.path.join(output_dir, f'CNN_MAP_Reslut_{SRP}_10%.xlsx')
    all_targets = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.squeeze(all_targets,1)
    all_predictions = np.squeeze(all_predictions,1)
    all_targets_np = np.concatenate(all_targets)
    all_predictions_np = np.concatenate(all_predictions)
    # 将预测和标签保存到Excel
    df = pd.DataFrame({
        'Real': all_targets_np,
        'Reconstruction': all_predictions_np
    })
    df.to_excel(output_excel_path, index=False)
    print(f"Data saved to: {output_excel_path}")

    # 绘制预测结果和标签数据的对比图
    plt.figure(figsize=(10, 5))
    plt.plot(all_targets_np, label="Ground Truth")
    plt.plot(all_predictions_np, label="Super-Resolution Output")
    plt.title("Super-Resolution vs Ground Truth")
    plt.legend()
    plt.show()

    # 绘制训练过程中损失变化的曲线relu
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# 5. 主程序 (Main)

if __name__ == "__main__":

    output_excel_path = './Results'
    output_model_path = './Model'
    SRP = 10
    # 数据准备(CNN的输入不要限制在 0,1之间否则会出现梯度爆炸的情况)
    train_loader, test_loader = prepare_data(low_res_path = './data_set/FY-3A 飞轮E转速 降采样（×10, 缺失3%的数据） ×100.xlsx',
                                             high_res_path = './data_set/FY-3A 飞轮E转速 ×100.xlsx',
                                             SRP = SRP, window_size = 200, stride = 200,
                                             batch_size = 64,test_split=0.2)

    # SRP = stride_1*stride_2*stride_3; stride等价于超分辨因子
    # 40 = 2*5*4
    # 20 = 2*5*2
    # 10 = 2*5*1
    # 8 = 2*2*2
    # 4 = 2*1*2
    # 2 =1*2*1
    model = Cnn_CovCnn_Model(input_channels = 1, output_channels = 1, kernel_size = 5,
                             feature_channels = 16, stride_1 = 2,stride_2 = 5,stride_3 = 1)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # 模型训练
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=180)

    # 模型评估和结果可视化
    evaluate_and_visualize(model, test_loader, train_losses, output_excel_path, SRP)



    # 保存模型
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    CNN_1DcovCNN_MAP_model_save_path = os.path.join(output_model_path, f'CNN_1DcovCNN_MAP_model_{SRP}_real.pth')
    torch.save(model.state_dict(), CNN_1DcovCNN_MAP_model_save_path)
    print(f"CNN_1DcovCNN_MAP_model saved to: {CNN_1DcovCNN_MAP_model_save_path}")