# 1 加载必要的库
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 2 定义超参数
BATCH_SIZE = 16  # 每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # 是否用GPU还是CPU训练
EPOCHS = 10  # 训练数据集的轮次
# 3 构建pipeline,对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化降低模型复杂度
])

# 下载数据集
train_set = datasets.MNIST(
    "./data", train=True, download=True, transform=pipeline)

test_set = datasets.MNIST("./data", train=False,
                          download=True, transform=pipeline)

# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True)  # 顺序打乱shuffle=True

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# for num, (image, label) in enumerate(train_loader):
#     image_batch = torchvision.utils.make_grid(image, padding=2)
#     plt.imshow(np.transpose(image_batch.numpy(), (1, 2, 0)), vmin=0, vmax=255)
#     plt.show()
#     print(label)


class TinnyCNN(nn.Module):
    def __init__(self, cls_num=2):
        super(TinnyCNN, self).__init__()
        # MNIST 数据集的图像尺寸是 28×28
        # 第一个卷积层：输入通道1，输出通道32，核大小3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        # 第二个卷积层：输入通道32，输出通道64，核大小3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))

        # 全连接层
        # 第一次卷积输出(28-3+1)×(28-3+1) = 26×26  通道数从1变为32  结果：26×26×32=21,632
        # 第二次卷积输出(26-3+1)×(26-3+1) = 24×24  通道数从32变为64  结果：24×24×64=36864
        # 两次卷积操作完成，开始展平操作，准备进入全连接层
        # 输入特征数为 36864(由前一层计算得到,3维转为二维)，输出特征数为 128 (可以根据需要调整,一般为2的幂次，与上一层总特征数无关)
        self.fc1 = nn.Linear(36864, 128)
        # fc1 将 36864 维的特征压缩到了 128 维
        # 第二次全连接层：输入特征数为 128，输出特征数为 10（类别数）
        self.fc2 = nn.Linear(128, cls_num)

    def forward(self, x):
        # 第一个卷积层
        x = self.conv1(x)
        x = F.relu(x)

        # 第二个卷积层
        x = self.conv2(x)
        x = F.relu(x)
        # 展平操作，准备进入全连接层
        x = x.view(x.size(0), -1)

       # 第一个全连接层
        x = self.fc1(x)
        x = F.relu(x)

        # 第二个全连接层
        x = self.fc2(x)
        return x


model = TinnyCNN(10)  # 10个类别

lss_f = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# 使用Adam优化器，通常收敛更快
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
# 使用更灵活的学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # 监控损失值
    factor=0.1,        # 学习率调整因子
    patience=2,       # 等待多少个epoch损失不下降才调整
    verbose='True'       # 打印学习率变化信息
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for data, labels in train_loader:
        outputs = model(data)
        optimizer.zero_grad()
        loss = lss_f(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct_num = (predicted == labels).sum()
        acc = correct_num / labels.shape[0]
        print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    # 在每个epoch结束后调整学习率
    scheduler.step(avg_loss)  # 使用平均损失来调整学习率

    # 在测试集验证
    model.eval()
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        correct_num = (predicted == labels).sum()
        acc = correct_num / labels.shape[0]
        print("Epoch:{} Test Acc:{:.0%}".format(epoch, acc))
