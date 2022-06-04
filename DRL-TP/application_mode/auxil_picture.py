# -*- encoding: utf-8 -*-
"""
@File : auxil_picture.py 
@Author : hlq
@Modify Time : 4/21/22 10:13 PM
@Descipe: None
@Version : 1.0 
"""
import pandas as pd
from utils_file import *
import matplotlib.pylab as plt
import numpy as np

class Auxil:
    def __init__(self):
        self.json_obj = SaveInfosToJson()
        self.path_obj = self.json_obj.make_path
        self.legends = ["DIJKSTRA", "OSPF", "DRL-TP"]
        self.label_font = {
            "family": "Times New Roman",
            "weight": "heavy",
            "color": "black",
            "size": 18
        }
        self.legend_style = {
            "family": "Times New Roman",
            "weight": "heavy",
            # "color": "black",
            "size": 18
        }


    def curve_result_picture(self, datas, title_type):
        """

        :return:
        """
        legends = ["DIJKSTRA", "OSPF", "DRL-TP"]
        colors = ["blue", "green", "red"]
        sign_type = ["d-", "o-", "*-"]
        plt.figure(figsize=(12, 6.5))
        plt.grid(color="darkgray", linestyle="--", linewidth=0.5)
        for idx in range(len(datas)):
            plt.plot(np.arange(len(datas[idx])), datas[idx], sign_type[idx], ms=10, color=colors[idx])
        plt.xlabel("Send traffic size(Mbit/s)")
        plt.ylabel(title_type, fontdict=self.label_font)
        plt.legend(legends, prop=self.legend_style)
        ax = plt.gca()
        # 设置x轴刻度值大小
        for text_obj in ax.get_xticklabels():
            text_obj.set_fontname('Times New Roman')
            text_obj.set_weight("bold")
            text_obj.set_fontsize('16')
        # 设置y轴刻度值大小
        for text_obj in ax.get_yticklabels():
            text_obj.set_fontname('Times New Roman')
            text_obj.set_weight("bold")
            text_obj.set_fontsize('16')

        plt.show()



    def read_compare_data(self):
        """"""
        path = self.path_obj.get_save_path(type="COMPARE_PATH")
        path_dir = os.path.dirname(os.path.dirname(path))
        time_dir = "/20220421/"
        file_dir = path_dir + time_dir
        file_name_list = ["DIJKSTRA_DATA.csv", "OSPF_DATA.csv", "DRL_DATA.csv"]
        throughput = []
        delay = []
        loss = []
        for file_name in file_name_list:
            csv = pd.read_csv(file_dir + file_name)
            values = csv.values
            t_list, d_list, l_list = [], [], []
            for item in values[1:]:
                t, d, l = eval(item[0]), eval(item[1]), eval(item[2])
                t_list.append(t)
                d_list.append(d)
                l_list.append(l)
            throughput.append(t_list)
            delay.append(d_list)
            loss.append(l_list)
        return throughput, delay, loss



if __name__ == '__main__':
    auxil = Auxil()
    throughput, delay, loss = auxil.read_compare_data()
    auxil.curve_result_picture(throughput, "throughput")
    auxil.curve_result_picture(delay, "delay")
    auxil.curve_result_picture(loss, "loss")

