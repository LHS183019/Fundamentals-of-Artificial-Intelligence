# email: xusinga@stu.pku.edu.cn / 2400013077@stu.pku.edu.cn
# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiStepLR
import kornia.augmentation as K

# 定义超参数
batch_size = 128
num_epochs = 100
warmup_epochs = 10
ONCOLAB = True
datapath = "data"
writer_file = "log"

# 定义数据预处理方式
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

# 普通的数据预处理方式
Testtransform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean,std) #将图像的每个通道的像素值归一化
    ]
)
# 数据增强的数据预处理方式(用了Kornia的Augmentation，所以增强没有写在这里)
Basictransform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

# 定义数据集
train_dataset = datasets.CIFAR10(root=datapath, train=True, download=True, transform=Basictransform)
test_dataset = datasets.CIFAR10(root=datapath, train=False, download=True, transform=Testtransform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True,prefetch_factor=3)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=4)

# 定义Flatten层，用于将输入向量展平为一维
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# 定义SE块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        self.reduction_ratio = reduction_ratio
        
        # Squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.Mish(inplace=True),  # 使用Mish激活函数
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        # kaiming初始化权重 针对Mish
        gain = torch.nn.init.calculate_gain('leaky_relu', 0.1)  # 由于nonlinearity没有mish 手动推导等效增益
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                with torch.no_grad():
                    m.weight.data *= gain  # 手动缩放权重

    def forward(self, x):
        batch_size, c, _, _ = x.size()
        # Squeeze
        squeezed = self.squeeze(x).view(batch_size, c)
        # Excitation
        weights = self.excitation(squeezed).view(batch_size, c, 1, 1)
        # Scale
        return x * weights.expand_as(x)


# 定义残差块(In-block POST SE-Res设计)
class SEResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(SEResidualBlock, self).__init__()
        # 定义残差模块的主体部分，包含一系列的卷积、批归一化和激活操作，每个块有两个卷积层
        self.residual = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            # inplace=True节省内存
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.se = SEBlock(outchannel,8)
        # 定义跳跃连接，初始为空序列（不对张量进行任何操作）
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # 如果步长不为1或者输入通道数和输出通道数不相等，则需要调整identity
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, identity):
        out = self.residual(identity)
        out = self.se(out) # SE块的输出与输入向量的shape一致
        out += self.shortcut(identity)
        out = F.relu(out)
        return out

# 定义模型
class SEResNet20forCifar10(nn.Module):
    '''
    残差网络
    '''
    def __init__(self):
        super(SEResNet20forCifar10, self).__init__()
        # RGB通道数
        self.in_channels = 3
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # 中间层
        self.in_channels = 16
        self.stage1 = self._make_layer(16, 3, 1)
        self.stage2 = self._make_layer(32, 3, 2)
        self.stage3 = self._make_layer(64, 3, 2)
        # 输出层
        self.output_layer = nn.Sequential(
            nn.AvgPool2d(8),
            Flatten(),
            nn.Linear(64, 10)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [SEResidualBlock(self.in_channels, out_channels, stride)] # 用于存储该层中的所有残差块
        for _ in range(1, num_blocks):
            layers.append(SEResidualBlock(out_channels, out_channels, stride=1))
        self.in_channels = out_channels # 更新输入通道数，以便下一个stage使用
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.output_layer(out)
        return out


# 辅助训练进度输出Teminal的自定义工具函数...
def clear_output():
    os_name = os.name
    if os_name == 'nt':  # Windows系统
        os.system('cls')
    elif os_name == 'posix':  # Linux和macOS系统
        os.system('clear')
prev_record = []
def print_prev_epoch_result():
    for string in prev_record:
        print(string)
def custom_print(string, on_colab=False,save_record=False):
    """清空终端输出(除了过往每个epoch的测试准确率打印会保留)再输出，
    如果on_colab=True的话效果等于正常的print()函数"""
    if not on_colab:
        clear_output() # 避免大量打印刷屏
        print_prev_epoch_result() # 保留过往训练信息
    print(string)
    if save_record:
        prev_record.append(string)


model = SEResNet20forCifar10()

# 定义GPU增强链
# 实例化模型
use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("GPU is not available, use CPU instead.")

model = model.to(device)

gpu_augment = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomCrop((32, 32),padding=4,padding_mode='reflect'),
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
    K.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    K.Normalize(torch.tensor(mean), torch.tensor(std))
).to(device)  # 确保增强器在GPU上

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(),lr=0.105, momentum=0.9, weight_decay=5e-4,nesterov=True)

# 定义学习率调度器
# StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),  # 0.01→0.1
        MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)  # 衰减起点为warmup后
    ],
    milestones=[warmup_epochs]
) # 预热+StepLR

# 训练模型
if __name__ == '__main__':

    # 训练日志
    writer = SummaryWriter(writer_file)

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        epoch_loss = 0.0
        epoch_corrects = 0
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)
            
            # GPU上执行增强（自动批处理）
            with torch.no_grad():  # 无需梯度
                images = gpu_augment(images)
                
            # 记录模型结构（只需一次）
            # if epoch == 0 and i == 0:
            #     writer.add_graph(model, images)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            accuracy = (outputs.argmax(1) == labels)
            epoch_corrects += accuracy.sum()
            accuracy = accuracy.float().mean()
            
            epoch_loss += loss.item() * images.size(0) # 乘以batchsize，用于后面计算epoch的平均loss
            
            # 打印训练信息
            if (i + 1) % 50 == 0:
                text = ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))
                custom_print(text,ONCOLAB)
                

        scheduler.step()
        writer.add_scalars(
            "AvgLoss", {"Train": epoch_loss / len(train_loader.dataset)}, epoch
        )
        writer.add_scalars(
            "Accuracy",
            {"Train": epoch_corrects.double() / len(train_loader.dataset)},
            epoch,
        )
        

        # 测试模式
        model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device,non_blocking=True)
                labels = labels.to(device,non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item() * images.size(0)

            # 在每个 epoch 结束后记录总损失和准确率
            writer.add_scalars(
                "AvgLoss", {"Test": running_loss / total}, epoch
            )
            writer.add_scalars(
                "Accuracy",
                {"Test": correct / total},
                epoch,
            )
            text = 'Test Accuracy of the model on the 10000 test images: {} % in epoch {}'.format(100 * correct / total,epoch)
            custom_print(text,ONCOLAB,save_record=True)
            prev_record.append(text)

    writer.close()
