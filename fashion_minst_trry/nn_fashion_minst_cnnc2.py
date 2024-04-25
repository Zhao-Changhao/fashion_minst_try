import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

class CNNC2(nn.Module):
    def __init__(self):
        super(CNNC2, self).__init__()
        self.conv1 = nn.Conv2d(1, 112, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(112)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(112, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 208)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(208, 160)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(160, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.dropout3(self.fc3(x))
        x = self.fc4(x)
        return x

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNC2().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
losses = []  # 用于存储每个epoch的平均loss

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# loss可视化
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.xticks(range(1, num_epochs+1))  # 确保每个epoch都有一个标记
plt.show()

# 显示9张图片和它们的预测
dataiter = iter(test_loader)
images, labels = dataiter.next()
images = images[:9]
labels = labels[:9]
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

# 图像显示及标签函数
def imshow(images, labels, predicted_labels, classes):
    images = images / 2 + 0.5  # 反归一化
    fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
    for idx, (image, ax) in enumerate(zip(images, axes)):
        image = image.numpy().transpose((1, 2, 0))
        ax.imshow(image)
        ax.set_title(f"Pred: {classes[predicted_labels[idx]]}\nTrue: {classes[labels[idx]]}")
        ax.axis('off')
    plt.show()

# 图像类别
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

imshow(torchvision.utils.make_grid(images), labels, predicted, classes)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(9)))
