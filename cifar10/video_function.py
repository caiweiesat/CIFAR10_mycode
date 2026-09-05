import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]   #黑体，Windows自带
plt.rcParams["axes.unicode_minus"] = False     #解决负号乱码

def change_process(single_pic, target_pic, R_lut=None, G_lut=None, B_lut=None):
    # single_pic->target_pic的每格像素变化的过程,返回numpy数组[N,H,W,C]
    pic_arr = []

    if R_lut is not None and G_lut is not None and B_lut is not None:
        # 三通道图片变化过程
        R_lut_np = np.array(R_lut)
        G_lut_np = np.array(G_lut)
        B_lut_np = np.array(B_lut)

        origin_pic = single_pic.astype(R_lut_np[0].dtype)
        for row_index, row in enumerate(origin_pic):
            for col_index, col in enumerate(row):
                index = target_pic[row_index][col_index][0]
                origin_pic[row_index][col_index][0] = R_lut_np[index]

                index = target_pic[row_index][col_index][1]
                origin_pic[row_index][col_index][1] = G_lut_np[index]

                index = target_pic[row_index][col_index][2]
                origin_pic[row_index][col_index][2] = B_lut_np[index]

                pic_arr.append(origin_pic.copy())

        return np.array(pic_arr)

    elif R_lut is not None and G_lut is None and B_lut is None:
        # R通道图片变化过程
        R_lut_np = np.array(R_lut)

        origin_pic = single_pic.astype(R_lut_np[0].dtype)
        for row_index, row in enumerate(origin_pic):
            for col_index, col in enumerate(row):
                index = target_pic[row_index][col_index][0]
                origin_pic[row_index][col_index][0] = R_lut_np[index]

                pic_arr.append(origin_pic.copy())

        pic_arr_np=np.array(pic_arr)
        return pic_arr_np[..., 0]

    elif R_lut is None and G_lut is not None and B_lut is None:
        # G通道图片变化过程
        G_lut_np = np.array(G_lut)

        origin_pic = single_pic.astype(G_lut_np[0].dtype)
        for row_index, row in enumerate(origin_pic):
            for col_index, col in enumerate(row):
                index = target_pic[row_index][col_index][1]
                origin_pic[row_index][col_index][1] = G_lut_np[index]

                pic_arr.append(origin_pic.copy())

        pic_arr_np=np.array(pic_arr)
        return pic_arr_np[..., 1]

    elif R_lut is None and G_lut is None and B_lut is not None:
        # B通道图片变化过程
        B_lut_np = np.array(B_lut)

        origin_pic = single_pic.astype(B_lut_np[0].dtype)
        for row_index, row in enumerate(origin_pic):
            for col_index, col in enumerate(row):
                index = target_pic[row_index][col_index][2]
                origin_pic[row_index][col_index][2] = B_lut_np[index]

                pic_arr.append(origin_pic.copy())

        pic_arr_np=np.array(pic_arr)
        return pic_arr_np[..., 2]

    else:
        raise TypeError("检查查找表输入")

def draw_frame_canvas(arr_list, main_rgb, info_texts, figsize=(14, 9), vmin=None, vmax=None, cmap="viridis"):
    """
    :param arr_list: list,长度=6，6个左侧热力图(H,W) 0,2,4为角度数组(32,32)
        [top_left, top_right, mid_left, mid_right, bot_left, bot_right]
    :param main_rgb: np.ndarray 右侧大图RGB (H,W,3) uint8
    :param info_texts: list[str],长度4
        [识别结果文本, 偏振文本, 光照文本, 灰度文本]
    :param figsize:画布尺寸
    :param vmin: 普通热力图最小值，None=自适应
    :param vmax: 普通热力图最大值，None=自适应
    :param cmap: 普通热力图配色
    :return: figure对象
    """
    # figure整体背景黑色
    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.subplots_adjust(bottom=0.02, top=0.98, left=0, right=0.96)

    gs = fig.add_gridspec(nrows=3, ncols=3,
                          width_ratios=[1, 1, 1.9],
                          height_ratios=[1, 1, 1],
                          wspace=0.08, hspace=0.08)

    axes_left = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ]
    ax_main = fig.add_subplot(gs[0:3, 2])

    pos = ax_main.get_position()
    x0, y0, w, h = pos.bounds  # x0左，y0底部，w宽，h高
    shift_up = 0.135  # 向上偏移量，调大上移更多，0.05~0.12区间
    right_up = 0.02   # 向右偏移
    ax_main.set_position((x0 + right_up, y0 + shift_up, w, h))

    idx = 0
    arrow_len = 0.38
    N = 32
    for row in axes_left:
        for ax in row:
            data = arr_list[idx]
            if idx in (0, 2, 4):
                # 角度可视化子图：底色plasma + 每个格子画一对箭头
                ax.set_xlim(-0.5, N - 0.5)
                ax.set_ylim(-0.5, N - 0.5)
                ax.set_aspect("equal")
                ax.invert_yaxis()
                ax.imshow(data, cmap="plasma", alpha=0.35, vmin=0, vmax=90)

                # 遍历网格绘制双箭头
                for y in range(N):
                    for x in range(N):
                        theta = data[y, x]
                        cx, cy = x, y
                        ang1 = np.radians(0)
                        ang2 = np.radians(0 - theta)

                        dx1 = arrow_len * np.cos(ang1)
                        dy1 = arrow_len * np.sin(ang1)
                        ax.arrow(cx, cy, dx1, dy1,
                                 head_width=0.06, head_length=0.08,
                                 fc="black", ec="black", lw=0.8)

                        dx2 = arrow_len * np.cos(ang2)
                        dy2 = arrow_len * np.sin(ang2)
                        ax.arrow(cx, cy, dx2, dy2,
                                 head_width=0.06, head_length=0.08,
                                 fc="black", ec="black", lw=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # idx=1,3,5 普通热力图
                ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
            idx += 1

    ax_main.imshow(main_rgb)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    text_result, text_polar, text_light, text_gray = info_texts

    from matplotlib.patches import Rectangle

    # ---------------------- 大结果框 ----------------------
    cx_mid = 0.75765
    # 大结果框参数
    center_x = cx_mid # 0.7555 -  0.76 = 0.0045   0.00225
    center_y = 0.179
    res_w = 0.4445
    res_h = 0.075 # 0.055

    res_box_x0 = center_x - res_w / 2
    res_box_y0 = center_y - res_h / 2

    rect_res = Rectangle((res_box_x0, res_box_y0), res_w, res_h,
                         facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_res)
    fig.text(center_x, center_y, text_result, fontsize=15, ha="center", va="center")

    # ---------------------- 下方三个小框 ----------------------
    add_space = 0.018
    cx_space = 0.13


    box_y = 0.081 # 位置高度
    small_w = 0.13 + add_space
    small_h = 0.12

    # polar
    polar_cx = cx_mid - cx_space - add_space
    polar_x0 = polar_cx - small_w / 2
    polar_y0 = box_y - small_h / 2
    rect_polar = Rectangle((polar_x0, polar_y0), small_w, small_h,
                           facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_polar)
    fig.text(polar_cx, box_y, text_polar, fontsize=15, ha="center", va="center")

    # light
    light_cx = cx_mid  # 0.77 0.01225
    light_x0 = light_cx - small_w / 2
    light_y0 = box_y - small_h / 2
    rect_light = Rectangle((light_x0, light_y0), small_w, small_h,
                           facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_light)
    fig.text(light_cx, box_y, text_light, fontsize=15, ha="center", va="center")

    # gray
    gray_cx = cx_mid + cx_space + add_space
    gray_x0 = gray_cx - small_w / 2
    gray_y0 = box_y - small_h / 2
    rect_gray = Rectangle((gray_x0, gray_y0), small_w, small_h,
                          facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_gray)
    fig.text(gray_cx, box_y, text_gray, fontsize=15, ha="center", va="center")

    return fig

def fig_to_rgb_np(fig):
    fig.canvas.draw()
    raw = np.array(fig.canvas.renderer.buffer_rgba())
    return raw[..., :3]  # 扔掉alpha通道，输出RGB (H,W,3)

def draw_title_canvas(title_lines, figsize=(14,9), fontsize=32):
    """
    单独绘制黑底标题画布，对应你示例图片效果
    :param title_lines: list[str], 多行标题文本，例如 ["Polarization–intensity synergistic","encoding and adversarial encryption"]
    :param figsize: 画布尺寸
    :param fontsize: 标题字号
    :return: matplotlib figure对象
    """
    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()   # 关闭坐标轴
    ax.set_facecolor("black")

    total_lines = len(title_lines)
    start_y = 0.5 + (total_lines-1)*0.08
    for idx, line_text in enumerate(title_lines):
        y_pos = start_y - idx * 0.11
        ax.text(
            0.5, y_pos,
            line_text,
            color="white",
            fontsize=fontsize,
            ha="center",
            va="center",
            transform=ax.transAxes
        )
    return fig

def draw_frame_canvas_energy(arr_list, main_rgb, info_texts, figsize=(14, 9), vmin=None, vmax=None, cmap="viridis"):
    """
    :param arr_list: list,长度=3，对应【右上、右中、右下】3张子图
        [top_right, mid_right, bot_right]
    :param main_rgb: np.ndarray 右侧大图RGB (H,W,3) uint8
    :param info_texts: list[str],长度4
        [识别结果文本, 偏振文本, 光照文本, 灰度文本]
    :param figsize:画布尺寸
    :param vmin: 热力图最小值，None=自适应
    :param vmax: 热力图最大值，None=自适应
    :param cmap: 热力图配色
    :return: figure对象
    """
    right_offset = -0.128   #-0.125 - 0.13

    # figure整体背景黑色
    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.subplots_adjust(bottom=0.02, top=0.98, left=0, right=0.96)

    gs = fig.add_gridspec(nrows=3, ncols=3,
                          width_ratios=[1, 1, 1.9],
                          height_ratios=[1, 1, 1],
                          wspace=0.08, hspace=0.08)

    axes_left = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ]
    ax_main = fig.add_subplot(gs[0:3, 2])

    pos = ax_main.get_position()
    x0, y0, w, h = pos.bounds  # x0左，y0底部，w宽，h高
    shift_up = 0.135
    right_up = 0.02 + right_offset
    ax_main.set_position((x0 + right_up, y0 + shift_up, w, h))

    data_idx = 0
    for row in axes_left:
        ax_empty, ax_target = row
        ax_empty.set_facecolor("black")
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])

        data = arr_list[data_idx]
        ax_target.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
        ax_target.set_xticks([])
        ax_target.set_yticks([])

        # 核心：取出子图原有位置，叠加偏移
        subplot_right_offset=right_offset
        t_pos = ax_target.get_position()
        ax_target.set_position((t_pos.x0 + subplot_right_offset, t_pos.y0, t_pos.width, t_pos.height))

        data_idx += 1

    ax_main.imshow(main_rgb)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    text_result, text_polar, text_light, text_gray = info_texts

    from matplotlib.patches import Rectangle

    cx_mid = 0.75765
    center_x = cx_mid + right_offset
    center_y = 0.179
    res_w = 0.4445
    res_h = 0.075

    res_box_x0 = center_x - res_w / 2
    res_box_y0 = center_y - res_h / 2

    rect_res = Rectangle((res_box_x0, res_box_y0), res_w, res_h,
                         facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_res)
    fig.text(center_x, center_y, text_result, fontsize=15, ha="center", va="center")

    add_space = 0.018
    cx_space = 0.13

    box_y = 0.081
    small_w = 0.13 + add_space
    small_h = 0.12

    polar_cx = cx_mid - cx_space - add_space + right_offset
    polar_x0 = polar_cx - small_w / 2
    polar_y0 = box_y - small_h / 2
    rect_polar = Rectangle((polar_x0, polar_y0), small_w, small_h,
                           facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_polar)
    fig.text(polar_cx, box_y, text_polar, fontsize=15, ha="center", va="center")

    light_cx = cx_mid + right_offset
    light_x0 = light_cx - small_w / 2
    light_y0 = box_y - small_h / 2
    rect_light = Rectangle((light_x0, light_y0), small_w, small_h,
                            facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_light)
    fig.text(light_cx, box_y, text_light, fontsize=15, ha="center", va="center")

    gray_cx = cx_mid + cx_space + add_space + right_offset
    gray_x0 = gray_cx - small_w / 2
    gray_y0 = box_y - small_h / 2
    rect_gray = Rectangle((gray_x0, gray_y0), small_w, small_h,
                          facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_gray)
    fig.text(gray_cx, box_y, text_gray, fontsize=15, ha="center", va="center")

    return fig

def draw_frame_canvas_angle(arr_list, main_rgb, info_texts, figsize=(14, 9), vmin=None, vmax=None, cmap="viridis"):
    """
    :param arr_list: list,长度=3，对应【右上、右中、右下】3张子图，每个是(32,32)角度数组
        [top_right, mid_right, bot_right]
    :param main_rgb: np.ndarray 右侧大图RGB (H,W,3) uint8
    :param info_texts: list[str],长度4
        [识别结果文本, 偏振文本, 光照文本, 灰度文本]
    :param figsize:画布尺寸
    :param vmin: 本参数保留接口，角度子图固定vmin=0 vmax=90
    :param vmax: 本参数保留接口，角度子图固定vmin=0 vmax=90
    :param cmap: 本参数保留接口，角度子图固定cmap="plasma"
    :return: figure对象
    """
    right_offset = -0.128   #‑0.125 ‑ 0.13

    # figure整体背景黑色
    fig = plt.figure(figsize=figsize, facecolor="black")
    fig.subplots_adjust(bottom=0.02, top=0.98, left=0, right=0.96)

    gs = fig.add_gridspec(nrows=3, ncols=3,
                          width_ratios=[1, 1, 1.9],
                          height_ratios=[1, 1, 1],
                          wspace=0.08, hspace=0.08)

    axes_left = [
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ]
    ax_main = fig.add_subplot(gs[0:3, 2])

    pos = ax_main.get_position()
    x0, y0, w, h = pos.bounds  # x0左，y0底部，w宽，h高
    shift_up = 0.135
    right_up = 0.02 + right_offset
    ax_main.set_position((x0 + right_up, y0 + shift_up, w, h))

    arrow_len = 0.38
    N = 32
    data_idx = 0
    for row in axes_left:
        ax_empty, ax_target = row
        ax_empty.set_facecolor("black")
        ax_empty.set_xticks([])
        ax_empty.set_yticks([])

        data = arr_list[data_idx]
        # ========= 改为角度热力+双箭头绘制，子图位置保持原有偏移逻辑不变 =========
        ax_target.set_xlim(-0.5, N - 0.5)
        ax_target.set_ylim(-0.5, N - 0.5)
        ax_target.set_aspect("equal")
        ax_target.invert_yaxis()
        ax_target.imshow(data, cmap="plasma", alpha=0.35, vmin=0, vmax=90)

        for y in range(N):
            for x in range(N):
                theta = data[y, x]
                cx, cy = x, y
                ang1 = np.radians(0)
                ang2 = np.radians(0 - theta)

                dx1 = arrow_len * np.cos(ang1)
                dy1 = arrow_len * np.sin(ang1)
                ax_target.arrow(cx, cy, dx1, dy1,
                         head_width=0.06, head_length=0.08,
                         fc="black", ec="black", lw=0.8)

                dx2 = arrow_len * np.cos(ang2)
                dy2 = arrow_len * np.sin(ang2)
                ax_target.arrow(cx, cy, dx2, dy2,
                         head_width=0.06, head_length=0.08,
                         fc="black", ec="black", lw=0.8)
        ax_target.set_xticks([])
        ax_target.set_yticks([])

        # 位置偏移逻辑完全保留，子图坐标不变
        subplot_right_offset=right_offset
        t_pos = ax_target.get_position()
        ax_target.set_position((t_pos.x0 + subplot_right_offset, t_pos.y0, t_pos.width, t_pos.height))

        data_idx += 1

    ax_main.imshow(main_rgb)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    text_result, text_polar, text_light, text_gray = info_texts

    from matplotlib.patches import Rectangle

    cx_mid = 0.75765
    center_x = cx_mid + right_offset
    center_y = 0.179
    res_w = 0.4445
    res_h = 0.075

    res_box_x0 = center_x - res_w / 2
    res_box_y0 = center_y - res_h / 2

    rect_res = Rectangle((res_box_x0, res_box_y0), res_w, res_h,
                         facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_res)
    fig.text(center_x, center_y, text_result, fontsize=15, ha="center", va="center")

    add_space = 0.018
    cx_space = 0.13

    box_y = 0.081
    small_w = 0.13 + add_space
    small_h = 0.12

    polar_cx = cx_mid - cx_space - add_space + right_offset
    polar_x0 = polar_cx - small_w / 2
    polar_y0 = box_y - small_h / 2
    rect_polar = Rectangle((polar_x0, polar_y0), small_w, small_h,
                           facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_polar)
    fig.text(polar_cx, box_y, text_polar, fontsize=15, ha="center", va="center")

    light_cx = cx_mid + right_offset
    light_x0 = light_cx - small_w / 2
    light_y0 = box_y - small_h / 2
    rect_light = Rectangle((light_x0, light_y0), small_w, small_h,
                            facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_light)
    fig.text(light_cx, box_y, text_light, fontsize=15, ha="center", va="center")

    gray_cx = cx_mid + cx_space + add_space + right_offset
    gray_x0 = gray_cx - small_w / 2
    gray_y0 = box_y - small_h / 2
    rect_gray = Rectangle((gray_x0, gray_y0), small_w, small_h,
                          facecolor="white", edgecolor="black", transform=fig.transFigure)
    fig.patches.append(rect_gray)
    fig.text(gray_cx, box_y, text_gray, fontsize=15, ha="center", va="center")

    return fig

