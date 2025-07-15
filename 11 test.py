import ode_demo
import Decnn_MAP
import NCDE_MAP
import linear_interpolation
import Nearest_Neighbor_Interpolation
import Cubic_spline_interpolation
import GAN
import VAE
import pfiter
import torch
import torchcde
import matplotlib.pyplot as plt

SRP = 2
low_res_path = './data_set/FY-3A 飞轮E转速 降采样（×2, 缺失3%的数据）.xlsx'
high_res_path = './data_set/FY-3A 飞轮E转速.xlsx'


window_size_NCDE = 160
stride_NCDE = 160
#测试 NCDE 模型
# 创建模型实例
TEST, label = NCDE_MAP.get_data(low_res_path = low_res_path,high_res_path = high_res_path,SRP = SRP, window_size = window_size_NCDE, stride = stride_NCDE)
coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(TEST)

NeuralCDE = NCDE_MAP.NeuralCDE(input_channels=2, hidden_channels=8, output_channels=1)
NeuralCDE.load_state_dict(torch.load(f'./Model/NCDE_MAP_model_{SRP}.pth'))

pred_y_all = NeuralCDE(coeffs, batch_time=window_size_NCDE // SRP, SRP=SRP).squeeze(-1)  # 使用整个训练集进行预测
pred_y_all = pred_y_all.flatten().detach().numpy()
label = label.flatten().detach().numpy()

plt.figure(figsize=(10, 5))
plt.plot(label.cpu().numpy(), label='True', color='blue')
plt.plot(pred_y_all.cpu().detach().numpy(), label='Predicted', color='red', linestyle='dashed')
plt.title(f'NCDE_model_{SRP}')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
# 继续进行模型的加载和调用

