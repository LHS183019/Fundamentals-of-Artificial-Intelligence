import matplotlib.pyplot as plt
import numpy as np

prefix = "output/"
title_name = "退火"
pic_name = "退火.png"

all_file = ["queens_output_origin.txt","queens_output_1.txt","queens_output_2.txt",
            "queens_output_3.txt","queens_output_4.txt","queens_output_5.txt",
            "queens_output_6.txt","queens_output_7.txt","queens_output_8.txt",
            "queens_output_9.txt"]
all_label = ["f_select_move","m_select_move","r_select_move",
             "r_select_imp_attempt","4n","m_select_swap",
             "r_select_swap","2n","n","n/2"]

config_compare_slection_on_move = [1,1,1,0]
config_compare_model_on_fselect = [1,0,0,0,1]
config_compare_model_on_mselect = [0,1,0,0,0,1]
config_compare_model_on_rselect = [0,0,1,0,0,0,1]
config_compare_slection_on_swap = [0,0,0,0,1,1,1]
config_compare_climb_max_step_on_swap = [0,0,0,0,1,0,0,1,1,1]
config_sim = [0,1]

enable = config_sim

for no,(file,label) in enumerate(zip(all_file,all_label)):
    if no >= len(enable):
        break
    if enable[no] != 1:
        continue 
    file_name = prefix+file

    # 读取文件
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # 提取"Total time"值
    times = []
    for line in lines:
        if "Total time" in line:
            # 提取时间值（去掉"Total time:"和"ms"）
            time_str = line.split("Total time: ")[1].split("ms")[0]
            times.append(float(time_str))
    # 设置x轴刻度
    x = list(map(str, np.arange(200, 200 + len(times))))
    plt.plot(x,times,label=label)

plt.xticks(rotation=45)
plt.xlabel('N queens')
plt.ylabel('Time (ms)')
plt.title(title_name)
plt.legend()
# 绘制图表
plt.savefig(pic_name)
plt.show()