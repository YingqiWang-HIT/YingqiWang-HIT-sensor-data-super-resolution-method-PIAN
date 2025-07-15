######################
# So you want to train a Neural CDE model?
# Let's get started!
######################

import math
import torchcde
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.cuda.empty_cache()

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Define linear layers
        self.linear1 = torch.nn.Linear(hidden_channels, 64)
        self.linear2 = torch.nn.Linear(64, 128)
        self.linear3 = torch.nn.Linear(128, 256)
        self.linear4 = torch.nn.Linear(256, 512)
        self.linear5 = torch.nn.Linear(512, 1024)
        self.linear6 = torch.nn.Linear(1024, 2048)
        self.linear7 = torch.nn.Linear(2048, 4096)
        self.linear8 = torch.nn.Linear(4096, 5500)
        self.linear9 = torch.nn.Linear(5500, 4096)
        self.linear10 = torch.nn.Linear(4096, 2048)
        self.linear11 = torch.nn.Linear(2048, 1024)
        self.linear12 = torch.nn.Linear(1024, 512)
        self.linear13 = torch.nn.Linear(512, 256)
        self.linear14 = torch.nn.Linear(256, 128)
        self.linear15 = torch.nn.Linear(128, 64)
        self.linear16 = torch.nn.Linear(64, input_channels * hidden_channels)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        # Initialize TanhScaled activation
    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.leaky_relu(self.linear1(z))
        z = self.leaky_relu(self.linear2(z))
        z = self.leaky_relu(self.linear3(z))
        z = self.leaky_relu(self.linear4(z))
        z = self.leaky_relu(self.linear5(z))
        z = self.leaky_relu(self.linear6(z))
        z = self.leaky_relu(self.linear7(z))
        z = self.leaky_relu(self.linear8(z))
        z = self.leaky_relu(self.linear9(z))
        z = self.leaky_relu(self.linear10(z))
        z = self.leaky_relu(self.linear11(z))
        z = self.leaky_relu(self.linear12(z))
        z = self.leaky_relu(self.linear13(z))
        z = self.leaky_relu(self.linear14(z))
        z = self.leaky_relu(self.linear15(z))
        z = self.leaky_relu(self.linear16(z))

        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# 神经受控微分方程，手粗
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs, batch_time, SRP):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        # 获取初始值
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        # 获取积分时间步长
        tensor_arange = torch.arange(0., float(batch_time), step=1/SRP).to(device)
        # 求解积分
        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=tensor_arange)
        # 这个部分

        #z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


######################
# 划分数据集
# 输入参数：
# low_res_path[低频数据地址], high_res_path[高频数据地址]
# windows_size[滑窗内数据量，相当于每个batch内的数据量], stride[每次滑窗滑动的距离]
# SRP超分辨因子
# 输出参数：
# Train形状[batchsize,windows]，
# Test形状[batchsize，windows×SRP]
######################
def get_data(low_res_path, high_res_path, SRP, window_size, stride):
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
    Sample_lows = torch.tensor(Sample_lows, dtype=torch.float32)
    Sample_highs = torch.tensor(Sample_highs, dtype=torch.float32)
    Sample_lows = torch.squeeze(Sample_lows, dim=2)  # 删除第三维度
    Sample_highs = torch.squeeze(Sample_highs, dim=2)  # 删除第三维度

    assert Sample_lows.shape[0] == Sample_highs.shape[0], "Low resolution and high resolution data sizes do not match."

    t = torch.linspace(0., window_size//SRP, window_size//SRP)

    X = torch.stack([t.unsqueeze(0).repeat(Sample_lows.shape[0], 1), Sample_lows], dim=2)

    return X, Sample_highs

def plot_comparison(pred, true, epoch):
    """绘制预测值和真实值的对比图"""
    plt.figure(figsize=(10, 5))
    plt.plot(true.cpu().numpy(), label='True', color='blue')
    plt.plot(pred.cpu().detach().numpy(), label='Predicted', color='red', linestyle='dashed')
    plt.title(f'Prediction vs True values at Epoch {epoch + 1}')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(num_epochs=2000, l2_lambda=0.01, windows_size = 200, stride = 200, SRP=4,
         test_split = 0.2, output_model_path = './Model', output_excel_path = './Results'):


    train, label = get_data(low_res_path = './data_set/FY-3A 飞轮E转速 降采样（×4）.xlsx',
                            high_res_path = './data_set/FY-3A 飞轮E转速.xlsx',
                                             SRP = SRP, window_size = windows_size, stride = stride)

    ######################
    # input_channels=2 because we have both the horizontal and vertical position of a point in the spiral, and time.
    # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
    # output_channels=1 because we're doing binary classification.
    ######################
    model = NeuralCDE(input_channels=2, hidden_channels=8, output_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train = train.to(device)
    label = label.to(device)
    #训练数据集划分
    train_label_size = int(len(label)*(1 - test_split))
    train_size = int(len(train)*(1 - test_split))
    train_dataset = train[:train_size]
    train_label_dataset = label[:train_label_size]

    #测试数据集
    test_dataset = train[train_size:].to(device)
    test_label_dataset = label[train_label_size:].to(device)

    #获取三次样条采样的系数
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_dataset)
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_dataset).to(device)

    #创建数据集
    train_dataset = TensorDataset(train_coeffs, train_label_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)


    ######################
    # 获取三次样条采样的系数.
    # 创建数据集
    ######################
    loss_history = []
    previous_loss = float('inf')  # 初始化上一个损失为无穷大

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # Reset epoch loss for each epoch

        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            batch_coeffs, batch_y = batch_coeffs.to(device), batch_y.to(device)

            pred_y = model(batch_coeffs, batch_time = windows_size//SRP, SRP = SRP).squeeze(-1)
            #基于MAP的模型训练
            likelihood_loss = torch.nn.functional.mse_loss(pred_y, batch_y)
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2
            prior_loss = (l2_lambda / 2) * l2_reg
            loss = likelihood_loss + prior_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate loss for this epoch
            epoch_loss += loss.item()

        # 计算平均损失
        avg_loss = epoch_loss / len(train_dataloader)
        loss_history.append(avg_loss)

        print(f'Epoch: {epoch + 1}/{num_epochs}   Average Training Loss: {avg_loss:.4f}')
        # 进行预测并绘制预测结果与真实结果的对比图
        with torch.no_grad():
            pred_y_all = model(test_coeffs, batch_time=windows_size // SRP, SRP=SRP).squeeze(-1)  # 使用整个训练集进行预测
            pred_y_all = pred_y_all.flatten()
            test_label_dataset = test_label_dataset.flatten()
            plot_comparison(pred_y_all, test_label_dataset, epoch=epoch)
        if avg_loss > previous_loss * 2:  # 设定阈值，例如增加100%
            print("Loss suddenly increased. Stopping training.")
            break
        previous_loss = avg_loss  # 更新上一个损失

    # 绘制训练损失曲线
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


    # 保存数据
    pred_y_all = model(test_coeffs, batch_time=windows_size // SRP, SRP=SRP).squeeze(-1)  # 使用整个训练集进行预测
    pred_y_all = pred_y_all.flatten().cpu().detach().numpy()
    test_label_dataset = test_label_dataset.flatten().cpu().detach().numpy()
    if not os.path.exists(output_excel_path):
        os.makedirs(output_excel_path)
    output_excel_path = os.path.join(output_excel_path, f'NCDE_MAP_Reslut_{SRP}.xlsx')
    df = pd.DataFrame({
        'Real': test_label_dataset,
        'Reconstruction': pred_y_all
    })
    df.to_excel(output_excel_path, index=False)
    print(f"Data saved to: {output_excel_path}")

    # 保存模型
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    NCDE_MAP_model_save_path = os.path.join(output_model_path, f'NCDE_MAP_model_{SRP}.pth')
    # 保存的是模型的参数
    torch.save(model.state_dict(),NCDE_MAP_model_save_path)
    print(f"NCDE_MAP_model saved to: {NCDE_MAP_model_save_path}")


if __name__ == '__main__':
    main()
