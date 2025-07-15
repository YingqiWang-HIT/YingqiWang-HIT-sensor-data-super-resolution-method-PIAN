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

# 2. 生成器 (Generator)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
            nn.Tanh(),

        )

    def forward(self, x):
        return self.model(x)

# 3. 判别器 (Discriminattoor)
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # 这里移除了 Sigmoid 层
        )

    def forward(self, x):
        return self.model(x)
# 4. 损失函数和优化器
criterion = nn.BCELoss()
# Wasserstein GAN 损失函数
def critic_loss(real_outputs, fake_outputs):
    # 目标是最小化真实样本的输出并最大化假样本的输出
    return -torch.mean(real_outputs) + torch.mean(fake_outputs)

def generator_loss(fake_outputs):
    # 生成器的目标是最大化 Critic 对假样本的得分
    return -torch.mean(fake_outputs)


def train_wgan(generator, critic, train_loader, optimizer_g, optimizer_c, num_epochs=1000, n_critic=5, clip_value=0.01):
    generator.to(device)
    critic.to(device)
    generator_losses = []
    critic_losses = []

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Train Critic multiple times
            for _ in range(n_critic):
                # Critic on real data
                real_outputs = critic(targets)
                noise = torch.randn(inputs.size(0), inputs.size(1)).to(device)
                fake_data = generator(noise)
                fake_outputs = critic(fake_data.detach())

                c_loss = critic_loss(real_outputs, fake_outputs)
                optimizer_c.zero_grad()
                c_loss.backward()
                optimizer_c.step()

                # Weight clipping to enforce Lipschitz constraint
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train Generator
            fake_outputs = critic(fake_data)
            g_loss = generator_loss(fake_outputs)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        generator_losses.append(g_loss.item())
        critic_losses.append(c_loss.item())

        if epoch % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {g_loss.item()}, Critic Loss: {c_loss.item()}")
            plt.figure(figsize=(10, 5))
            plt.plot(fake_data[0].detach().cpu().numpy(), label="Super-Resolution Output")
            plt.plot(targets[0].detach().cpu().numpy(), label="Ground Truth")
            plt.title("Super-Resolution vs Ground Truth")
            plt.legend()
            plt.show()

    return generator_losses, critic_losses


# 5. 评估和可视化
def evaluate_and_visualize(generator, test_loader, output_dir, SRP):
    """
    评估模型性能，并绘制结果对比图。
    """

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_excel_path = os.path.join(output_dir, f'GAN_Reslut_{SRP}.xlsx')
    all_targets = []
    all_predictions = []

    generator.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            noise = torch.randn(inputs.size(0), inputs.size(1)).to(device)
            generated_data = generator(noise)
            targets = targets.flatten()
            generated_data = generated_data.flatten()

            # 将预测和标签保存到列表中
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(generated_data.cpu().numpy())

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
    plt.plot(all_predictions_np, label="Super-Resolution Output")
    plt.plot(all_targets_np, label="Ground Truth")
    plt.title("Super-Resolution vs Ground Truth")
    plt.legend()
    plt.show()

    # 绘制训练过程中损失变化的曲线relu
    plt.figure(figsize=(10, 5))
    plt.plot(generator_losses, label="generator_losses")
    plt.plot(critic_losses, label="critic_losses")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# 主程序入口
if __name__ == "__main__":
    # 数据准备
    window_size = stride = 200
    SRP = 8
    batch_size = 64
    learning_rate_g = 1e-4
    learning_rate_c = 1e-5
    num_epochs = 2000
    output_model_path = './Model'
    output_excel_path = './Results'

    train_loader, test_loader = prepare_data(
        low_res_path='./data_set/FY-3A 飞轮E转速 降采样（×8）.xlsx',
        high_res_path='./data_set/FY-3A 飞轮E转速.xlsx',
        SRP=SRP,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        test_split=0.2
    )

    # WGAN模型实例化
    generator = Generator(input_dim=window_size//SRP, output_dim=window_size)
    critic = Critic(input_dim=window_size)

    optimizer_g = optim.RMSprop(generator.parameters(), lr=learning_rate_g)
    optimizer_c = optim.RMSprop(critic.parameters(), lr=learning_rate_c)

    # 训练WGAN
    generator_losses, critic_losses = train_wgan(generator, critic, train_loader, optimizer_g, optimizer_c, num_epochs)

    # 模型评估和结果可视化
    evaluate_and_visualize(generator, test_loader, output_dir=output_excel_path, SRP=SRP)

    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    Gen_model_save_path = os.path.join(output_model_path, f'GAN_generator_model_{SRP}.pth')
    torch.save(generator.state_dict(),Gen_model_save_path)
    print(f"GAN_generator_model saved to: {Gen_model_save_path}")

    Dis_model_save_path = os.path.join(output_model_path, f'GAN_discriminator_model_{SRP}.pth')
    torch.save(critic.state_dict(), Dis_model_save_path)
    print(f"GAN_discriminator_model saved to: {Dis_model_save_path}")

