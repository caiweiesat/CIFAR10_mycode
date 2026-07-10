"""
gray_combination用于得到器件可模拟的灰度值，
返回的字典的key为灰度，value为该灰度值需要的能量密度、偏振角、绝对差(用于函数内哈希排序,后续可能用于衡量模拟效果？)
"""
import numpy as np

def gray_combination(energy_density:list[list[float]]|None=None,
                     polarized_light:list[list[float|int]]|None=None,
                     energy_benchmark:float|None=None,
                     slope:float|None=None)->dict[int,list]:
    """
    三种情况，只有能量密度调节，只有偏振角调节，能量密度和偏振角组合调节
    """
    if energy_density is not None and polarized_light is None:
        print("只有能量密度调节")
        min_current=energy_density[0][0]
        max_current=energy_density[-1][0]
        # 灰度 将电流映射到0-255范围
        # 哈希结构，只存放最接近整数灰度的组合
        # 整数灰度(key):[能量密度,绝对差]
        all_gray={}
        for current,energy in energy_density:
            # 灰度(浮点数)
            gray_float = (current - min_current) / (max_current - min_current) * 255
            # 灰度(整数)
            gray_int = round(gray_float)
            # 绝对差
            absolute_difference = abs(gray_float - gray_int)
            # 如果该灰度已存在，则对比浮点数和整数的绝对值，保留最接近整数的组合
            if gray_int in all_gray:
                # 当前字典绝对差
                dict_absolute_difference = all_gray[gray_int][1]
                if absolute_difference < dict_absolute_difference:
                    all_gray[gray_int] = [energy, absolute_difference]
            else:
                all_gray[gray_int] = [energy, absolute_difference]
        return all_gray

    elif polarized_light is not None and energy_density is None:
        print("只有偏振角调节")
        polarized_light.sort()
        min_current=polarized_light[0][0]
        max_current=polarized_light[-1][0]
        # 灰度 将电流映射到0-255范围
        # 哈希结构，只存放最接近整数灰度的组合
        # 整数灰度(key):[偏振角,绝对差]
        all_gray = {}
        for current,angle in polarized_light:
            # 灰度(浮点数)
            gray_float = (current - min_current) / (max_current - min_current) * 255
            # 灰度(整数)
            gray_int = round(gray_float)
            # 绝对差
            absolute_difference = abs(gray_float - gray_int)
            # 如果该灰度已存在，则对比浮点数和整数的绝对值，保留最接近整数的组合
            if gray_int in all_gray:
                # 当前字典绝对差
                dict_absolute_difference = all_gray[gray_int][1]
                if absolute_difference < dict_absolute_difference:
                    all_gray[gray_int] = [angle, absolute_difference]
            else:
                all_gray[gray_int] = [angle, absolute_difference]
        return all_gray

    elif energy_density is not None and polarized_light is not None\
            and energy_benchmark is not None and slope is not None:
        print("偏振角和能量密度组合调节")
        # 能量比例 记录能量对比默认值(偏振角测试的能量密度)的比例
        # 格式为[[比例,能量密度],...]
        energy_radio = []
        for _,energy in energy_density:
            # 能量变化的比例
            radio_result = energy / energy_benchmark
            # radio的slope次方
            energy_radio.append([radio_result**slope, energy])

        # 可能的电流值 由不同的能量密度和偏振角组合而来
        # 格式为[[电流值,偏振角,能量密度],...]
        possible_current = []
        for radio, energy in energy_radio:
            for current, angle in polarized_light:
                possible_current.append([radio * current, angle, energy])
        # 排序是为了后续快速取出最大和最小值
        possible_current.sort()
        min_current = possible_current[0][0]
        max_current = possible_current[-1][0]

        # 灰度 将电流映射到0-255范围
        # 哈希结构，只存放最接近整数灰度的组合
        # 整数灰度(key):[角度,能量密度,绝对差]
        all_gray = {}
        for current, radio, energy in possible_current:
            # 灰度(浮点数)
            gray_float = (current - min_current) / (max_current - min_current) * 255
            # 灰度(整数)
            gray_int = round(gray_float)
            # 绝对差
            absolute_difference = abs(gray_float - gray_int)
            # 如果该灰度已存在，则对比浮点数和整数的绝对值，保留最接近整数的组合
            if gray_int in all_gray:
                # 当前字典绝对差
                dict_absolute_difference = all_gray[gray_int][2]
                if absolute_difference < dict_absolute_difference:
                    all_gray[gray_int] = [radio, energy, absolute_difference]
            else:
                all_gray[gray_int] = [radio, energy, absolute_difference]
        return all_gray

    else:raise KeyError("参数错误")

"""
gray_apx用于给指定的数据集按照字典变化最接近的灰度值
"""
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
    # 能量密度mW/cm2 635nm波长
    #[[电流(nA),能量密度(mW/cm2)],...]
    R_energy_density = [[3.19E-04,0.00797],
                        [1.70E-03,0.03755],
                        [4.28E-03,0.0879],
                        [1.09E-02,0.1951],
                        [2.77E-02,0.46929],
                        [1.37E-01,2.17483],
                        [1.10E+00,17.55245],
                        [2.70E+00,44.33566],
                        [7.14E+00,97.9021]]
    # 偏振光角度与对应的电流
    # 635nm波长 能量密度2.17 mW/cm2
    #[[电流,偏振角],...]
    R_polarized_light = [[0.15127, 0],
                       [0.269561, 15],
                       [0.60883, 30],
                       [0.779254, 45],
                       [1.08693, 60],
                       [1.429209, 75],
                       [1.74369, 90]]
    # 能量基准,在该能量下测试偏转角数据
    R_energy_benchmark=2.17

    only_energy=gray_combination(energy_density=R_energy_density)
    print(only_energy)
    for key in only_energy:
        print(key)
    only_polarized=gray_combination(polarized_light=R_polarized_light)
    print(only_polarized)
    for key in only_polarized:
        print(key)
    energy_and_polarized=gray_combination(energy_density=R_energy_density,
                                          polarized_light=R_polarized_light,
                                          energy_benchmark=R_energy_benchmark)
    print(energy_and_polarized)
    for key in energy_and_polarized:
        print(key)

    # 测试错误输入
    energy_and_polarized = gray_combination(energy_density=R_energy_density,
                                            polarized_light=R_polarized_light,
                                            )