import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 自定义目标函数：sin(x) + e^x
def target_func(x):
    return np.sin(x) + np.exp(x)

# 生成训练集和测试集
def generate_data(n, x_range=(-5, 5)):
    x = np.random.uniform(x_range[0], x_range[1], size=(n, 1))
    y = target_func(x)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 定义两层ReLU网络
class Net(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 参数设置
hidden_size = 100
learning_rate = 0.01
num_epochs = 2000

# 生成数据
train_x, train_y = generate_data(1000)
test_x = torch.linspace(-5, 5, 200).view(-1, 1)  # 均匀采样用于可视化
test_y = torch.tensor(target_func(test_x.numpy()), dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = Net(hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    
    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_x)
        test_loss = criterion(test_outputs, test_y)
    
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 可视化训练过程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss')

# 可视化拟合效果
model.eval()
with torch.no_grad():
    x_plot = torch.linspace(-5, 5, 500).view(-1, 1)
    y_pred = model(x_plot)
    y_true = torch.tensor(target_func(x_plot.numpy()), dtype=torch.float32)

plt.subplot(1, 2, 2)
plt.plot(x_plot.numpy(), y_true.numpy(), label='True Function')
plt.plot(x_plot.numpy(), y_pred.numpy(), linestyle='--', label='Model Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function Fitting Result')
plt.tight_layout()
plt.show()