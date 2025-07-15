# -*- coding: utf-8 -*-
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

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
    Sample_lows = Sample_lows.reshape(Sample_lows.shape[0], Sample_lows.shape[1])
    Sample_highs = Sample_highs.reshape(Sample_highs.shape[0], Sample_highs.shape[1])
    Sample_lows = torch.tensor(Sample_lows, dtype=torch.float32).to(device)
    Sample_highs = torch.tensor(Sample_highs, dtype=torch.float32).to(device)

    # 确保低分辨率和高分辨率样本数量相同
    assert Sample_lows.shape[0] == Sample_highs.shape[0], "Low resolution and high resolution data sizes do not match."

    # 计算训练集和测试集的大小
    dataset_size = len(Sample_highs)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    #创建数据集
    Sample_lows_train = Sample_lows[:train_size,:]
    Sample_highs_train = Sample_highs[:train_size,:]
    train_dataset = TensorDataset(Sample_lows_train, Sample_highs_train)
    Sample_lows_test = Sample_lows[train_size:,:]
    Sample_highs_test = Sample_highs[train_size:,:]
    test_dataset = TensorDataset(Sample_lows_test, Sample_highs_test)
    # 划分数据集

    # 创建数据加载器zai zaide shengde shen
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)#随机种子，让模型的训练从不同的位置开始
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 2. VAE编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim_1, latent_dim_2):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 40)
        self.fc2 = nn.Linear(40, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 624)
        self.fc5 = nn.Linear(624, 1048)
        self.fc6 = nn.Linear(1048, 2096)
        self.fc7 = nn.Linear(2096, 4192)
        self.fc8 = nn.Linear(4192, 2096)
        self.fc9 = nn.Linear(2096, 1028)
        self.fc9 = nn.Linear(2096, 1028)
        self.fc10= nn.Linear(1028, 512)
        self.fc11_mean = nn.Linear(512, 512)
        self.fc11_logvar = nn.Linear(512, 512)

    def forward(self, x):
        # x = torch.relu(self.bn1(self.fc1(x)))
        # x = torch.relu(self.fc2(x))
        # x = torch.relu(self.bn3(self.fc3(x)))
        # x = torch.relu(self.fc4(x))
        # x = torch.relu(self.bn5(self.fc5(x)))
        # x = torch.relu(self.fc6(x))
        # mean = self.fc7_mean(x)
        # logvar = self.fc7_logvar(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        mean = self.fc11_mean(x)
        logvar = self.fc11_logvar(x)
        return x, mean, logvar

# 3. Reparameterization trick
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

# 4. VAE解码器
class Decoder(nn.Module):
    def __init__(self, latent_dim_3, latent_dim_4, latent_dim_5, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc6 = nn.Linear(1024, 1500)
        self.bn6 = nn.BatchNorm1d(1500)
        self.fc7 = nn.Linear(1500, 2000)
        self.bn7 = nn.BatchNorm1d(2000)
        self.fc8 = nn.Linear(2000, 3000)
        self.bn8 = nn.BatchNorm1d(3000)
        self.fc9 = nn.Linear(3000, 4000)
        self.bn9 = nn.BatchNorm1d(4000)
        self.fc10 = nn.Linear(4000, 5000)
        self.bn10 = nn.BatchNorm1d(5000)
        self.fc11 = nn.Linear(5000, 6000)
        self.bn11 = nn.BatchNorm1d(6000)
        self.fc12 = nn.Linear(6000, 7000)
        self.bn12 = nn.BatchNorm1d(7000)
        self.fc13 = nn.Linear(7000, 6000)
        self.bn13 = nn.BatchNorm1d(6000)
        self.fc14 = nn.Linear(6000, 5000)
        self.bn14 = nn.BatchNorm1d(5000)
        self.fc15 = nn.Linear(5000, 4000)
        self.bn15 = nn.BatchNorm1d(4000)
        self.fc16 = nn.Linear(4000, 3000)
        self.bn16 = nn.BatchNorm1d(3000)
        self.fc17 = nn.Linear(3000, 2000)
        self.bn17 = nn.BatchNorm1d(2000)
        self.fc18 = nn.Linear(2000, 1500)
        self.bn18 = nn.BatchNorm1d(1500)
        self.fc19 = nn.Linear(1500, 1024)
        self.bn19 = nn.BatchNorm1d(1024)
        self.fc20 = nn.Linear(1024, 512)
        self.bn20 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, 256)
        self.bn21 = nn.BatchNorm1d(256)
        self.fc22 = nn.Linear(256, output_dim)

    def forward(self, z):
        # z = torch.relu(self.bn3(self.fc3(z)))
        # z = torch.relu(self.bn4(self.fc4(z)))
        # z = torch.relu(self.bn5(self.fc5(z)))
        # z = torch.relu(self.bn6(self.fc6(z)))
        # z = torch.relu(self.bn7(self.fc7(z)))
        # z = torch.relu(self.bn8(self.fc8(z)))
        # z = torch.relu(self.bn9(self.fc9(z)))
        # z = torch.relu(self.bn10(self.fc10(z)))
        # z = torch.relu(self.bn11(self.fc11(z)))
        # z = torch.relu(self.bn12(self.fc12(z)))
        # z = torch.relu(self.bn13(self.fc13(z)))
        # z = torch.relu(self.bn14(self.fc14(z)))
        # z = torch.relu(self.bn15(self.fc15(z)))
        # z = torch.relu(self.bn16(self.fc16(z)))
        # z = torch.relu(self.bn17(self.fc17(z)))
        # z = torch.relu(self.bn18(self.fc18(z)))
        # z = torch.relu(self.bn19(self.fc19(z)))
        # z = torch.relu(self.bn20(self.fc20(z)))
        # z = torch.relu(self.bn21(self.fc21(z)))
        # z = torch.relu(self.fc22(z))
        z = torch.relu(self.fc3(z))
        z = torch.tanh(self.fc4(z))
        z = torch.relu(self.fc5(z))
        z = torch.tanh(self.fc6(z))
        z = torch.relu(self.fc7(z))
        z = torch.tanh(self.fc8(z))
        z = torch.relu(self.fc9(z))
        z = torch.tanh(self.fc10(z))
        z = torch.relu(self.fc11(z))
        z = torch.tanh(self.fc12(z))
        z = torch.relu(self.fc13(z))
        z = torch.tanh(self.fc14(z))
        z = torch.relu(self.fc15(z))
        z = torch.tanh(self.fc16(z))
        z = torch.relu(self.fc17(z))
        z = torch.tanh(self.fc18(z))
        z = torch.relu(self.fc19(z))
        z = torch.tanh(self.fc20(z))
        z = torch.relu(self.fc21(z))
        z = torch.tanh(self.fc22(z))
        return z  # 输出为高分辨率数据
# 5. VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, SRP):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, input_dim * 4, input_dim * 8)
        self.decoder = Decoder(input_dim * 8, input_dim * 16, input_dim * 16, input_dim * SRP)

    def forward(self, x):
        x, mean, logvar = self.encoder(x)
        # z = reparameterize(mean, logvar)
        # x_recon = self.decoder(z)
        x_recon = self.decoder(x)
        return x_recon, mean, logvar

# 6. VAE的损失函数（重构损失 + KL散度）
def loss_function(recon_x, x, mean, logvar):
    reconstruction_loss = nn.MSELoss()(recon_x, x)  # 重构损失
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())  # KL散度
    return reconstruction_loss + kl_divergence

# 训练VAE
def train_model(model, train_loader, loss_function, optimizer, num_epochs=1000):
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
            outputs, mean, logvar = model(inputs)  # 前向传播
            loss = loss_function(outputs, targets, mean, logvar)

            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()
            train_losses.append(running_loss / len(train_loader))

        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
            plt.figure(figsize=(10, 5))
            plt.plot(outputs[0].squeeze().detach().cpu().numpy(), label="Super-Resolution Output")
            plt.plot(targets[0].squeeze().detach().cpu().numpy(), label="Ground Truth")
            plt.title("Train")
            plt.legend()
            plt.show()

    return train_losses


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

    # 使用当前时间生成不同文件名
    output_excel_path = os.path.join(output_dir, f'VAE_Reslut_{SRP}.xlsx')

    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, means, logvars= model(inputs)
            targets = targets.flatten()
            predictions = predictions.flatten()

            # 将预测和标签保存到列表中
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    # 将所有样本的数据合并为一个长的数组
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
    plt.plot(all_targets_np, label="Super-Resolution Output")
    plt.plot(all_predictions_np, label="Ground Truth")
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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He初始化
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 主程序入口
if __name__ == "__main__":
    # 数据准备
    window_size = stride = 200
    SRP = 4
    learning_rate = 0.00001
    output_model_path = './Model'
    output_excel_path = './Results'
    train_loader, test_loader = prepare_data(low_res_path='./data_set/FY-3A 飞轮E转速 降采样（×4, 缺失5%的数据）.xlsx',
                                             high_res_path='./data_set/FY-3A 飞轮E转速.xlsx',
                                             SRP=SRP, window_size=window_size, stride=stride,
                                             batch_size=64, test_split=0.2)

    # VAE模型实例化
    model = VAE(input_dim = window_size//SRP ,SRP = SRP)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练VAE
    train_losses = train_model(model, train_loader, loss_function, optimizer, num_epochs = 1000)

    # 测试集上生成高分辨率数据
    # 模型评估和结果可视化
    evaluate_and_visualize(model, test_loader, train_losses, output_dir = output_excel_path,SRP = SRP)

    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    model_save_path = os.path.join(output_model_path, f'VAE_SRP_model_{SRP}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
