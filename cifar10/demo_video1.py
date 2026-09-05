import cv2
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from DenseNet import *
from simulation_approximation import *
from video_function import *

import torchvision.transforms as transforms
from pathlib import Path
from device_test_data import *
from read import *


if __name__ == "__main__":
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = load_all_cifar10(r"D:\PythonProject\cifar10\cifar-10-batches-py")

    # 读取类别名称
    meta_path = r"D:\PythonProject\cifar10\cifar-10-batches-py\batches.meta"
    meta_data = unpickle(meta_path)
    class_names = meta_data["label_names"]
    print(class_names)

    """
    一共6组实验
    RGB 能量密度+偏振角
    RGB 能量密度
    RGB 偏振角
    R   能量密度+偏振角
    G   能量密度+偏振角
    B   能量密度+偏振角
    """

    # 先获取R、G、B的能量和偏振角组合的可调灰度值
    # 能量密度和偏振角组合
    R_energy_and_polarized_gray,R_energy_and_polarized_lut = gray_combination(energy_density=R_energy_density,
                                                   polarized_light=R_polarized_light,
                                                   energy_benchmark=R_energy_benchmark,
                                                   slope=R_slope)
    G_energy_and_polarized_gray,G_energy_and_polarized_lut = gray_combination(energy_density=G_energy_density,
                                                   polarized_light=G_polarized_light,
                                                   energy_benchmark=G_energy_benchmark,
                                                   slope=G_slope)
    B_energy_and_polarized_gray,B_energy_and_polarized_lut = gray_combination(energy_density=B_energy_density,
                                                   polarized_light=B_polarized_light,
                                                   energy_benchmark=B_energy_benchmark,
                                                   slope=B_slope)

    # RGB 能量密度+偏振角 灰度lut
    R_energy_and_polarized_graylut = my_list_split(R_energy_and_polarized_lut,0)
    G_energy_and_polarized_graylut = my_list_split(G_energy_and_polarized_lut, 0)
    B_energy_and_polarized_graylut = my_list_split(B_energy_and_polarized_lut, 0)

    """偏振角"""
    # 模拟(能量+角度)需要的偏振角lut
    R_energy_and_polarized_anglelut = my_list_split(R_energy_and_polarized_lut, 1)
    G_energy_and_polarized_anglelut = my_list_split(G_energy_and_polarized_lut, 1)
    B_energy_and_polarized_anglelut = my_list_split(B_energy_and_polarized_lut, 1)

    """能量密度"""
    # 模拟(能量+角度)需要的能量密度lut
    R_energy_and_polarized_energylut = my_list_split(R_energy_and_polarized_lut, 2)
    G_energy_and_polarized_energylut = my_list_split(G_energy_and_polarized_lut, 2)
    B_energy_and_polarized_energylut = my_list_split(B_energy_and_polarized_lut, 2)

    # pic0->pic2
    pic0=x_train[0]
    pic2=x_train[2]

    # 获取图像的各种编码
    encoding_pic = gray_apx(pic0, R_energy_and_polarized_graylut, color="R")
    encoding_pic = gray_apx(pic0, G_energy_and_polarized_graylut, color="G", apx_x_data=encoding_pic)
    encoding_pic = gray_apx(pic0, B_energy_and_polarized_graylut, color="B", apx_x_data=encoding_pic)


    polarized_pic = gray_apx(pic0, R_energy_and_polarized_anglelut, color="R")
    polarized_pic = gray_apx(pic0, G_energy_and_polarized_anglelut, color="G", apx_x_data=polarized_pic)
    polarized_pic = gray_apx(pic0, B_energy_and_polarized_anglelut, color="B", apx_x_data=polarized_pic)


    energy_pic = gray_apx(pic0, R_energy_and_polarized_energylut, color="R")
    energy_pic = gray_apx(pic0, G_energy_and_polarized_energylut, color="G", apx_x_data=energy_pic)
    energy_pic = gray_apx(pic0, B_energy_and_polarized_energylut, color="B", apx_x_data=energy_pic)

    # 获取动画帧
    img_np=change_process(encoding_pic, pic2, R_lut=R_energy_and_polarized_graylut
                          , G_lut=G_energy_and_polarized_graylut
                          , B_lut=B_energy_and_polarized_graylut)

    # change_process会自动返回单个通道
    R_polarized_np = change_process(polarized_pic, pic2, R_lut=R_energy_and_polarized_anglelut)
    G_polarized_np = change_process(polarized_pic, pic2, G_lut=G_energy_and_polarized_anglelut)
    B_polarized_np = change_process(polarized_pic, pic2, B_lut=B_energy_and_polarized_anglelut)

    R_energy_np = change_process(energy_pic, pic2, R_lut=R_energy_and_polarized_energylut)
    G_energy_np = change_process(energy_pic, pic2, G_lut=G_energy_and_polarized_energylut)
    B_energy_np = change_process(energy_pic, pic2, B_lut=B_energy_and_polarized_energylut)

    """推理所需数据"""
    batch_tensor = torch.from_numpy(img_np).permute(0, 3, 1, 2).float() / 255.0
    batch_tensor = batch_tensor.to(device)  # 直接可以丢进model推理

    """加载模型"""
    save_path = r"D:\PythonProject\cifar10\cifar10_output\model_RGB_能量密度+偏振角.pth"

    checkpoint = torch.load(save_path, map_location=device)

    model = densenet_bc_100(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 把权重加载进你的model
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    """开始推理"""
    model.eval()
    pred_all = []
    split_bs = 32  # 调小，16/32都行
    total = batch_tensor.size(0)
    class_names_cn=\
        ['飞机', '汽车', '鸟',
         '猫', '鹿', '狗',
         '青蛙', '马', '船', '卡车']
    with torch.no_grad():
        for s in range(0, total, split_bs):
            e = min(s + split_bs, total)
            sub_x = batch_tensor[s:e]
            out = model(sub_x)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            pred_all.append(pred)

    pred_ids = np.concatenate(pred_all)
    # for idx, pred in enumerate(pred_ids):
    #     print(f"{idx}:{class_names_cn[pred]}")


    """绘制画布"""
    # 初始化视频设置
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "demo_video1.mp4",  # 输出文件名，就是导出的视频文件
        fourcc,
        60,  # ✅帧率在这里！单位：帧/秒 # 2845KB
        (1400, 900)
    )
    print("视频画面大小初始化完成")

    # 标题帧画面
    title_lines = ["Polarization–Intensity Synergistic", "Encoding And Adversarial Encryption"]
    title_fig_out = draw_title_canvas(title_lines, figsize=(14, 9), fontsize=48)
    # 画布转RGB
    title_rgb_frame = fig_to_rgb_np(title_fig_out)
    # 转BGR
    title_bgr = cv2.cvtColor(title_rgb_frame, cv2.COLOR_RGB2BGR)
    # 写入视频
    for _ in range(120):
        out.write(title_bgr)
    plt.close(title_fig_out)

    for change_index in range(1024):
        # 构造6张模拟热力图 (32x32)
        heat_1 = R_polarized_np[change_index]
        heat_2 = R_energy_np[change_index]
        heat_3 = G_polarized_np[change_index]
        heat_4 = G_energy_np[change_index]
        heat_5 = B_polarized_np[change_index]
        heat_6 = B_energy_np[change_index]


        arr_input = [heat_1, heat_2, heat_3, heat_4, heat_5, heat_6]

        # 模拟右侧RGB大图
        rgb_main = img_np[change_index]

        # 当前变化的像素点
        pixel_row = change_index // 32 # 行索引
        pixel_col = change_index %  32 # 列索引

        # 底部文本
        info = [
            f"Recognition Result:{class_names[pred_ids[change_index]]}",
            f"Polarization Angle\nR:{heat_1[pixel_row][pixel_col]}\nG:{heat_3[pixel_row][pixel_col]}\nB:{heat_5[pixel_row][pixel_col]}",
            f"Optical Intensity\nR:{heat_2[pixel_row][pixel_col]}\nG:{heat_4[pixel_row][pixel_col]}\nB:{heat_6[pixel_row][pixel_col]}",
            f"RGB Value\nR:{rgb_main[pixel_row][pixel_col][0]}\nG:{rgb_main[pixel_row][pixel_col][1]}\nB:{rgb_main[pixel_row][pixel_col][2]}"
        ]
        # info = [
        #     "Recognition Result: Target A",
        #     "Real‑time Values\nR:Polarization Angle\nG:Polarization Angle\nB:Polarization Angle",
        #     "Real‑time Values\nR:Light Intensity\nG:Light Intensity\nB:Light Intensity",
        #     "Real‑time Values\nR:Gray Scale\nG:Gray Scale\nB:Gray Scale"
        # ]


        # 加密过程视频
        fig_out = draw_frame_canvas(arr_input, rgb_main, info,vmin=0,vmax=130)


        # 画布转RGB
        rgb_frame = fig_to_rgb_np(fig_out)

        # 转BGR
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        out.write(bgr)
        plt.close(fig_out)
        print(f"写入第{change_index}帧")
    # 导出视频
    out.release()
    sys.exit()
