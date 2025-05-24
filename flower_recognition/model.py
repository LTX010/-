import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision.models import resnet152, ResNet152_Weights
import torch.nn as nn
from tqdm import tqdm

# 数据预处理变换
transform = {
    'train':transforms.Compose([
    transforms.Resize((224, 224)),  # 统一图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.ToTensor(),          # 转为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
]),
    'test':transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}

# 构建数据集
dataset_path = r"花卉识别/flower7595/flowers"

# 加载数据集
full_dataset = datasets.ImageFolder(root=dataset_path)

# 按 80/20 比例划分训练和验证集
# 设置随机种子
torch.manual_seed(42)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 分别设置 transform
train_dataset.dataset.transform = transform['train']
test_dataset.dataset.transform = transform['test']

# 构建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载预训练权重
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)

# 冻结权重
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层全连接层
model.fc = nn.Linear(model.fc.in_features, 5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss() #定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #定义优化器

best_acc = 0.0
best_model = None

save_pth = "model/best_model.pth"
for epoch in range(10):
    model.train()
    running_loss = 0.0
    epoch_acc = 0
    epoch_num = 0
    train_num = 0
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_description("train epoch [{}/{}] loss:{:.4f}".format(epoch+1, 10, loss.item()))

        epoch_num += (output.argmax(dim=1) == labels.to(device)).view(-1).sum()
        train_num += labels.size(0)

    epoch_acc = epoch_num / train_num

    print("train epoch [{}/{}] acc:{:3f}".format(epoch+1, 10, epoch_acc))


    test_acc = 0.0
    test_num = 0
    model.eval()
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            test_bar.set_description("test epoch [{}/{}] loss:{:.4f}".format(epoch+1, 10, loss.item()))

            test_num += (output.argmax(dim=1) == labels.to(device)).view(-1).sum()
            test_acc = test_num / test_dataset.__len__()

    print("test epoch [{}/{}] acc:{:3f}".format(epoch+1, 10, test_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = model.state_dict()

    if epoch == 9:
        torch.save(best_model, save_pth)
print('Finished Training')

