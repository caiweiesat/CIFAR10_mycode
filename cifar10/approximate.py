from read import *
from simulation_approximation import gray_combination
from device_test_data import *

def gray_apx(x_data:np.ndarray,gray_dic:dict,color:str)->np.ndarray:
    apx_x_data=x_data.copy()

    # key的numpy数组
    keys_np = np.array(list(gray_dic.keys()))
    # 查找表
    lut = np.zeros(256, dtype=np.uint8)
    for gray in range(256):
        # 得到该灰度和每个key的距离
        dif_np = np.abs(gray - keys_np)
        # 最小值下标
        min_idx = np.argmin(dif_np)
        # 选取最短距离的key
        lut[gray] = keys_np[min_idx]

    if color == "R":color_through=0
    elif color =="G":color_through=1
    elif color =="B":color_through=2
    else:raise KeyError("颜色参数错误")
    # 根据查找表替换
    best_key = lut[apx_x_data[..., color_through]]
    apx_x_data[..., color_through] = best_key

    return apx_x_data

if __name__ == "__main__":
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_all_cifar10(r"D:\PythonProject\cifar10\cifar-10-batches-py")

    # 读取类别名称
    meta_path = r"D:\PythonProject\cifar10\cifar-10-batches-py\batches.meta"
    meta_data = unpickle(meta_path)
    class_names = meta_data["label_names"]

    apx_x_train=x_train.copy()
    """R"""
    R_all_gray=gray_combination(energy_density=R_energy_density,
                                polarized_light=R_polarized_light,
                                energy_benchmark=R_energy_benchmark,
                                slope=R_slope)

    apx_x_train = gray_apx(apx_x_train,R_all_gray,color="R")

    """G"""
    G_all_gray = gray_combination(energy_density=G_energy_density,
                                  polarized_light=G_polarized_light,
                                  energy_benchmark=G_energy_benchmark,
                                  slope=G_slope)

    apx_x_train = gray_apx(apx_x_train, G_all_gray, color="G")

    """B"""
    B_all_gray = gray_combination(energy_density=B_energy_density,
                                  polarized_light=B_polarized_light,
                                  energy_benchmark=B_energy_benchmark,
                                  slope=B_slope)

    apx_x_train = gray_apx(apx_x_train, B_all_gray, color="B")

    # 模拟前

    plt.imshow(x_train[0])
    plt.savefig("模拟前.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 模拟后
    plt.imshow(apx_x_train[0])
    plt.savefig("模拟后.png", dpi=300, bbox_inches="tight")
    plt.close()

