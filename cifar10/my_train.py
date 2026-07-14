import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from matplotlib.ticker import MaxNLocator
# 以下是自定义的py文件
from cnn import *
from read import *
from device_test_data import *
from simulation_approximation import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载数据集
(x_train, y_train), (x_test, y_test) = load_all_cifar10(r"D:\CIFAR10\cifar10\cifar-10-batches-py")

# 读取类别名称
meta_path = r"D:\CIFAR10\cifar10\cifar-10-batches-py\batches.meta"
meta_data = unpickle(meta_path)
class_names = meta_data["label_names"]

"""
一共6组实验
RGB 能量密度+偏振角
RGB 能量密度
RGB 偏振角
R   能量密度+偏振角
G   能量密度+偏振角
B   能量密度+偏振角
"""

#先获取R、G、B的单能量、单偏振角、能量和偏振角组合的可调灰度值
# 只有能量密度
R_energy_gray=gray_combination(energy_density=R_energy_density)
G_energy_gray=gray_combination(energy_density=G_energy_density)
B_energy_gray=gray_combination(energy_density=B_energy_density)
# 只有偏振角
R_polarized_gray=gray_combination(polarized_light=R_polarized_light)
G_polarized_gray=gray_combination(polarized_light=G_polarized_light)
B_polarized_gray=gray_combination(polarized_light=B_polarized_light)
# 能量密度和偏振角组合
R_energy_and_polarized_gray = gray_combination(energy_density=R_energy_density,
                                               polarized_light=R_polarized_light,
                                               energy_benchmark=R_energy_benchmark,
                                               slope=R_slope)
G_energy_and_polarized_gray = gray_combination(energy_density=G_energy_density,
                                               polarized_light=G_polarized_light,
                                               energy_benchmark=G_energy_benchmark,
                                               slope=G_slope)
B_energy_and_polarized_gray = gray_combination(energy_density=B_energy_density,
                                               polarized_light=B_polarized_light,
                                               energy_benchmark=B_energy_benchmark,
                                               slope=B_slope)

# 标准化参数
# norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# 模拟后的参数
norm = transforms.Normalize((0.4687, 0.4703, 0.4414), (0.3086, 0.2659, 0.2781))
# 然后获取RGB三种情况下的数据集，单通道的数据集由RGB通道数据集做切片获得
# RGB 能量密度
RGB_energy_x_train = x_train.copy()
RGB_energy_x_test = x_test.copy()

RGB_energy_x_train = gray_apx(RGB_energy_x_train, R_energy_gray, color="R")
RGB_energy_x_train = gray_apx(RGB_energy_x_train, G_energy_gray, color="G")
RGB_energy_x_train = gray_apx(RGB_energy_x_train, B_energy_gray, color="B")

RGB_energy_x_test = gray_apx(RGB_energy_x_test, R_energy_gray, color="R")
RGB_energy_x_test = gray_apx(RGB_energy_x_test, G_energy_gray, color="G")
RGB_energy_x_test = gray_apx(RGB_energy_x_test, B_energy_gray, color="B")

# 维度重排+转float+归一化
RGB_energy_x_train_tensor = torch.from_numpy(RGB_energy_x_train).permute(0,3,1,2).float()/255.0
RGB_energy_x_test_tensor = torch.from_numpy(RGB_energy_x_test).permute(0,3,1,2).float()/255.0
# 标准化
RGB_energy_x_train_tensor = norm(RGB_energy_x_train_tensor)
RGB_energy_x_test_tensor = norm(RGB_energy_x_test_tensor)

# RGB 偏振角
RGB_polarized_x_train = x_train.copy()
RGB_polarized_x_test = x_test.copy()

RGB_polarized_x_train = gray_apx(RGB_polarized_x_train, R_polarized_gray, color="R")
RGB_polarized_x_train = gray_apx(RGB_polarized_x_train, G_polarized_gray, color="G")
RGB_polarized_x_train = gray_apx(RGB_polarized_x_train, B_polarized_gray, color="B")

RGB_polarized_x_test = gray_apx(RGB_polarized_x_test, R_polarized_gray, color="R")
RGB_polarized_x_test = gray_apx(RGB_polarized_x_test, G_polarized_gray, color="G")
RGB_polarized_x_test = gray_apx(RGB_polarized_x_test, B_polarized_gray, color="B")

# 维度重排+转float+归一化
RGB_polarized_x_train_tensor = torch.from_numpy(RGB_polarized_x_train).permute(0,3,1,2).float()/255.0
RGB_polarized_x_test_tensor = torch.from_numpy(RGB_polarized_x_test).permute(0,3,1,2).float()/255.0
# 标准化
RGB_polarized_x_train_tensor = norm(RGB_polarized_x_train_tensor)
RGB_polarized_x_test_tensor = norm(RGB_polarized_x_test_tensor)

# RGB 能量密度+偏振角
RGB_energy_and_polarized_x_train = x_train.copy()
RGB_energy_and_polarized_x_test = x_test.copy()

RGB_energy_and_polarized_x_train = gray_apx(RGB_energy_and_polarized_x_train, R_energy_and_polarized_gray, color="R")
RGB_energy_and_polarized_x_train = gray_apx(RGB_energy_and_polarized_x_train, G_energy_and_polarized_gray, color="G")
RGB_energy_and_polarized_x_train = gray_apx(RGB_energy_and_polarized_x_train, B_energy_and_polarized_gray, color="B")

RGB_energy_and_polarized_x_test = gray_apx(RGB_energy_and_polarized_x_test, R_energy_and_polarized_gray, color="R")
RGB_energy_and_polarized_x_test = gray_apx(RGB_energy_and_polarized_x_test, G_energy_and_polarized_gray, color="G")
RGB_energy_and_polarized_x_test = gray_apx(RGB_energy_and_polarized_x_test, B_energy_and_polarized_gray, color="B")

# 维度重排+转float+归一化
RGB_energy_and_polarized_x_train_tensor = torch.from_numpy(RGB_energy_and_polarized_x_train).permute(0,3,1,2).float()/255.0
RGB_energy_and_polarized_x_test_tensor = torch.from_numpy(RGB_energy_and_polarized_x_test).permute(0,3,1,2).float()/255.0
# 标准化
RGB_energy_and_polarized_x_train_tensor = norm(RGB_energy_and_polarized_x_train_tensor)
RGB_energy_and_polarized_x_test_tensor = norm(RGB_energy_and_polarized_x_test_tensor)

# R 能量密度+偏振角
R_energy_and_polarized_x_train_tensor = RGB_energy_and_polarized_x_train_tensor[:,0:1,:,:]
R_energy_and_polarized_x_test_tensor = RGB_energy_and_polarized_x_test_tensor[:,0:1,:,:]
# G 能量密度+偏振角
G_energy_and_polarized_x_train_tensor = RGB_energy_and_polarized_x_train_tensor[:,1:2,:,:]
G_energy_and_polarized_x_test_tensor = RGB_energy_and_polarized_x_test_tensor[:,1:2,:,:]
# B 能量密度+偏振角
B_energy_and_polarized_x_train_tensor = RGB_energy_and_polarized_x_train_tensor[:,2:3,:,:]
B_energy_and_polarized_x_test_tensor = RGB_energy_and_polarized_x_test_tensor[:,2:3,:,:]
"""
一共6组实验
RGB 能量密度+偏振角
RGB 能量密度
RGB 偏振角
R   能量密度+偏振角
G   能量密度+偏振角
B   能量密度+偏振角
"""
# 实验数据tuple用来控制循环
research_data_tuple=((RGB_energy_and_polarized_x_train_tensor,RGB_energy_and_polarized_x_test_tensor),
                     (RGB_energy_x_train_tensor,RGB_energy_x_test_tensor),
                     (RGB_polarized_x_train_tensor,RGB_polarized_x_test_tensor),
                     (R_energy_and_polarized_x_train_tensor,R_energy_and_polarized_x_test_tensor),
                     (G_energy_and_polarized_x_train_tensor,G_energy_and_polarized_x_test_tensor),
                     (B_energy_and_polarized_x_train_tensor,B_energy_and_polarized_x_test_tensor))
# 每项实验共用的标签张量
train_y_tensor = torch.from_numpy(y_train).long()
test_y_tensor = torch.from_numpy(y_test).long()
for experiment_number,(train_x_tensor,test_x_tensor) in enumerate(research_data_tuple):
    # 数据集
    train_set = TensorDataset(train_x_tensor, train_y_tensor)
    test_set = TensorDataset(test_x_tensor, test_y_tensor)
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    if train_x_tensor.shape[1] == 3:
        model = MyNet()
    elif train_x_tensor.shape[1] == 1:
        model = MyNetSingleColor()
    else:
        raise ValueError(".shape[1]的值应是3或1")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)


    # 训练集损失list
    loss_train_list = []
    # 测试集损失list
    loss_test_list = []
    #正确率list
    correct_list = []
    # ==========训练轮数==========
    epochs = 30

    for i in range(epochs + 1):
        if i == 0:  # 第0轮计算loss和正确率,但不更新权重
            # 仅前向，确保评估模式时norm层可用
            model.train()
            with torch.no_grad():
                for images,labels in train_loader:
                    outputs=model(images)
            # 评估模式，正确率+训练集loss+测试集loss
            model.eval()
            correct = 0
            total = 0
            epoch_loss_train = 0
            epoch_loss_test = 0
            with torch.no_grad():
                for images, labels in train_loader:
                    outputs = model(images)
                    loss=criterion(outputs,labels)
                    epoch_loss_train += loss.item()
                for images, labels in test_loader:
                    outputs = model(images)
                    _, pred = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                    loss = criterion(outputs,labels)
                    epoch_loss_test += loss.item()
            # 正确率
            acc = correct / total
            correct_list.append(acc)
            # 训练集损失
            avg_loss_train = epoch_loss_train / len(train_loader)
            loss_train_list.append(avg_loss_train)
            # 测试集损失
            avg_loss_test = epoch_loss_test / len(test_loader)
            loss_test_list.append(avg_loss_test)

            print(f"第{i}轮正确率{acc:.2f},训练集损失:{avg_loss_train:.4f},测试集损失:{avg_loss_test:.4f}")

        else:  # 每轮训练计算loss和正确率,且更新权重
            model.train()
            for images, label in train_loader:
                pred = model(images)
                loss = criterion(pred, label)
                # 更新权重
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            epoch_loss_train = 0
            epoch_loss_test = 0
            with torch.no_grad():
                for images, labels in train_loader:
                    outputs = model(images)
                    loss = criterion(outputs,labels)
                    epoch_loss_train += loss.item()
                for images, labels in test_loader:
                    outputs = model(images)
                    _, pred = torch.max(outputs, dim=1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                    loss = criterion(outputs,labels)
                    epoch_loss_test += loss.item()
            # 正确率
            acc = correct / total
            correct_list.append(acc)
            # 训练集损失
            avg_loss_train = epoch_loss_train / len(train_loader)
            loss_train_list.append(avg_loss_train)
            # 测试集损失
            avg_loss_test = epoch_loss_test / len(test_loader)
            loss_test_list.append(avg_loss_test)
            print(f"第{i}轮正确率{acc:.2f},训练集损失:{avg_loss_train:.4f},测试集损失:{avg_loss_test:.4f}")


    experiment_name=("RGB_能量密度+偏振角",
                     "RGB_能量密度",
                     "RGB_偏振角",
                     "R_能量密度+偏振角",
                     "G_能量密度+偏振角",
                     "B_能量密度+偏振角")
    print(f"{experiment_name[experiment_number]}训练完成")
    # x轴：1~总轮数
    x = list(range(0, epochs + 1))
    # y轴：每轮损失
    y1 = loss_train_list
    y2 = loss_test_list
    plt.figure(figsize=(8, 4))
    plt.plot(x, y1, color='red', marker='o', label='Train Loss')
    plt.plot(x, y2, color='blue', marker='s', label='Test Loss')
    # 强制X轴只显示整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & test Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./cifar10_output/损失曲线{experiment_name[experiment_number]}.png", dpi=300)
    plt.close()
    with open(f"./cifar10_output/损失{experiment_name[experiment_number]}.txt",'w',encoding="utf-8") as f:
        for idx in range(epochs):
            f.write(f"epoch={idx+1}   train loss={y1[idx]:.4f}  |  test loss={y2[idx]:.4f}\n")

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
    plt.savefig(f"./cifar10_output/正确率曲线{experiment_name[experiment_number]}.png", dpi=300)
    plt.close()
    with open(f"./cifar10_output/正确率{experiment_name[experiment_number]}.txt",'w',encoding="utf-8") as f:
        for idx,accuracy in enumerate(correct_list):
            f.write(f"epoch={idx}   accuracy={accuracy:.2f}\n")

    break