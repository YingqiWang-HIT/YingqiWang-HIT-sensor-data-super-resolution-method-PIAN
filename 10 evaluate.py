import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from dtaidistance import dtw  # 安装库: pip install dtaidistance

# 读取 Excel 文件
file_path = 'Results/CNN_MAP_Reslut_10_10%.xlsx'  # 替换为你的 Excel 文件路径
data = pd.read_excel(file_path)

# 假设第一列是真实值，第二列是预测值
y_true = data.iloc[:, 0].values  # 第一列：真实值
y_pred = data.iloc[:, 1].values  # 第二列：预测值

# 计算 MSE
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse}")

# 计算 RMSE
rmse = sqrt(mse)
print(f"RMSE: {rmse}")

# 计算 DTW
dtw_distance = dtw.distance(y_true, y_pred)
print(f"DTW Distance: {dtw_distance}")
