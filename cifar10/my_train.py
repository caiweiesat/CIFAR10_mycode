import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set=torchvision.datasets.CIFAR10(root=r"D:\CIFAR10\cifar10",
                                        train=True,
                                        download=False,
                                        transform=transform)

train_loader=DataLoader(train_set,batch_size=64,shuffle=True)

test_set=torchvision.datasets.CIFAR10(root=r"D:\CIFAR10\cifar10",
                                      train=False,
                                      download=False,
                                      transform=transform)

test_loader=DataLoader(test_set,batch_size=64,shuffle=False)

model=MyNet()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

# 损失list
loss_list=[]
#正确率list
correct_list=[]
epochs=30

for i in range(epochs+1):
    if i==0:# 第0轮计算loss和正确率
        # loss
        model.train()
        epoch_loss = 0
        with torch.no_grad():
            for images, label in train_loader:
                pred = model(images)
                loss = criterion(pred, label)

                epoch_loss += loss.item()
            # 训练集损失
            avg_loss = epoch_loss / len(train_loader)
            loss_list.append(avg_loss)
        # 正确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, pred = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = correct / total
        print(f"第{i}轮正确率{acc:.2f}")
        correct_list.append(acc)
    else:# 每轮训练计算loss和正确率
        # loss
        model.train()
        epoch_loss=0
        for images,label in train_loader:
            pred=model(images)
            loss=criterion(pred,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()

        avg_loss=epoch_loss/len(train_loader)
        loss_list.append(avg_loss)

        # 正确率
        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for images,labels in test_loader:
                outputs=model(images)
                _,pred=torch.max(outputs,dim=1)
                total+=labels.size(0)
                correct+=(pred==labels).sum().item()
        acc=correct/total
        print(f"第{i}轮正确率{acc:.2f}")
        correct_list.append(acc)

""""""
test_num=5
""""""

# x轴：1~总轮数
x = list(range(0, epochs + 1))
# y轴：每轮损失
y = loss_list

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='red', marker='o', label='Train Loss')
# 强制X轴只显示整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"loss_curve{test_num}.png", dpi=300)
plt.close()


# x轴：1~总轮数
x = list(range(0, epochs + 1))
# y轴：每轮正确率
y = correct_list
plt.figure(figsize=(8, 4))
plt.plot(x, y, color='red', marker='o', label='Train Accuracy')
# 强制X轴只显示整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# 固定Y轴0~1
plt.ylim(0, 1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"Accuracy_curve{test_num}.png", dpi=300)
plt.close()