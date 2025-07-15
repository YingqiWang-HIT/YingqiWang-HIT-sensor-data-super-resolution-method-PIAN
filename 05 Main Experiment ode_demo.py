import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 通过调整
def get_batch(true_y, batch_time, batch_size):
    # 划分样本
    t = torch.linspace(0., float(true_y.shape[0]), true_y.shape[0])
    s = torch.from_numpy(np.random.choice(np.arange(len(true_y) - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

class ODEFunc(torch.nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.linear1 = torch.nn.Linear(1, 32)
        self.linear2 = torch.nn.Linear(32, 64)
        self.linear3 = torch.nn.Linear(64, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 32)
        self.linear6 = torch.nn.Linear(32, 1)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.relu()
        z = self.linear3(z)
        z = z.relu()
        z = self.linear4(z)
        z = z.relu()
        z = self.linear5(z)
        z = z.relu()
        z = self.linear6(z)
        z = z.relu()
        return z

class NeuralODE(torch.nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()

        self.func = ODEFunc()

    def forward(self, batch_y0, batch_t):
        pred_y = odeint(func = self.func, y0 = batch_y0, t = batch_t)
        return pred_y

if __name__ == '__main__':

    num_epochs = 1000
    test_freq = 10

    true_y = torch.tensor(pd.read_excel("origin+.xlsx").values, dtype=torch.float32).unsqueeze(1)

    model = NeuralODE().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    for itr in range(1, num_epochs + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(true_y = true_y, batch_time = 30 ,batch_size = 60)
        pred_y = model(batch_y0 = batch_y0 , batch_t = batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                pred_y = model(batch_y0=batch_y0[0], batch_t=batch_t).to(device)
                loss = torch.mean(torch.abs(pred_y - batch_y[:,0,:,:]))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # 绘制预测值与真实值的对比图
                plt.figure(figsize=(10, 6))

                plt.plot(batch_t.cpu().numpy(), batch_y.cpu().numpy()[:, 0 ,0, 0], label='true')
                plt.plot(batch_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], label='Predicted')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.title('True vs Predicted')
                plt.show()
    torch.save(model, 'ode_model.pth')
