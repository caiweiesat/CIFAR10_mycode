import pickle
import numpy as np
import matplotlib.pyplot as plt

from turn import *

def unpickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data

def load_all_cifar10(file_path):
    train_imgs, train_labs = [], []
    # 读取5个训练batch，硬编码绝对路径
    for i in range(1, 6):  #D:\PythonProject\cifar10\cifar-10-batches-py
        batch_path = file_path + fr"\data_batch_{i}"
        batch_data = unpickle(batch_path)
        train_imgs.append(batch_data["data"])
        train_labs.extend(batch_data["labels"])

    # 整合训练集 NHWC格式 (N,32,32,3)
    x_train = np.vstack(train_imgs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(train_labs)

    # 读取测试集
    test_path = file_path + r"\test_batch"
    test_batch = unpickle(test_path)
    x_test = test_batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_batch["labels"])

    return (x_train, y_train), (x_test, y_test)

def show_train_image(index, x_train, y_train, class_names):
    if index < 0 or index >= len(x_train):
        print(f"索引超出范围！有效范围 0 ~ {len(x_train) - 1}")
        return

    img = x_train[index]
    label_id = y_train[index]
    label_name = class_names[label_id]

    plt.figure()
    plt.imshow(img)
    plt.title(f"Index: {index} | Class: {label_name}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_all_cifar10(r"D:\PythonProject\cifar10\cifar-10-batches-py")

    # 读取类别名称
    meta_path = r"D:\PythonProject\cifar10\cifar-10-batches-py\batches.meta"
    meta_data = unpickle(meta_path)
    class_names = meta_data["label_names"]

    # 调用时显式传入数据，逻辑清晰
    show_train_image(4999, x_train, y_train, class_names)


    # 变化前
    before_change=x_train[531]
    # 目标
    target_change=x_train[1]

    # 变化前
    plt.imshow(before_change)
    plt.savefig("变化前.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 目标
    plt.imshow(target_change)
    plt.savefig("目标.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 变化后
    rgb_result=rgb_change(before_change, target_change)
    plt.imshow(rgb_result)
    plt.savefig("变化后.png", dpi=300, bbox_inches="tight")
    plt.close()




