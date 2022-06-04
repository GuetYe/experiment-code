# _*_coding:utf-8_*_
import csv
import ast
import json
import os
import time
from ryu import cfg
from ryu.base import app_manager
from ryu.lib import hub

import network_discover
import network_monitor
import network_delay
import network_manager


CONF = cfg.CONF
import configuration as CF


class Operation(app_manager.RyuApp):
    # --- used to load class of custom module --- #
    algo = None
    if CONF.algo == "DRL":
        import drl_forwarding
        algo = {"routing_module": drl_forwarding.DRLForwarding}
    if CONF.algo == "DIJKSTRA":
        import dijskra_forwarding
        algo = {"routing_module": dijskra_forwarding.DIJSKRAForwarding}
    if CONF.algo == "OSPF":
        import ospf_forwarding
        algo = {"routing_module": ospf_forwarding.OSPFForwarding}
    class_module = {
        "topology_module": network_discover.TopoDiscover,
        "monitor_module": network_monitor.MonitorDetection,
        "delay_module": network_delay.DelayMeasure,
        "management_module": network_manager.ManagementPlane
    }
    class_module.update(algo)
    _CONTEXTS = class_module

    # WEIGHT_MODEL = {'delay': 'delay', 'bw': "bandwidth"}

    def __init__(self, *args, **kwargs):
        super(Operation, self).__init__(*args, **kwargs)
        self.name = "routing_module"
        # --- load module --- #
        self.topology_module = kwargs["topology_module"]
        self.monitor_module = kwargs["monitor_module"]
        self.delay_detector = kwargs["delay_module"]
        self.routing_module = kwargs["routing_module"]
        self.management_module = kwargs["management_module"]
        self.json_obj = self.routing_module.json_obj
        self.path_obj = self.routing_module.path_obj
        # self.metric_factors = ["free_bw", "delay", "packet_loss", "used_bw", "packet_drop", "packet_errors"]
        self.metric_factors = ["delay", "packet_loss", "throughput"]
        self.compare_count = 1
        self.init_csv_header = True
        self.save_type = CONF.algo
        self.main_spawn = hub.spawn(self.operator_main)

    def show_mean_metric_info(self, metric_type, throught_type_path, delay_type_path, loss_type_path):
        """

        :param metric:
        :param metric_type:
        :return:
        """

        # drl_metrics = os.listdir(metric_type_path)
        throught_metrics = os.listdir(throught_type_path)
        delay_metrics = os.listdir(delay_type_path)
        loss_metrics = os.listdir(loss_type_path)
        # bw_metrics = os.listdir(bw_type_path)
        # mertic_mean_values = [0. for _ in range(len(self.metric_factors) + 4)]
        indications = [0. for _ in range(3)]              # throughtput, delay, loss
        # total_file = len(throught_metrics)  # 50
        total_throughput_file = len(throught_metrics)
        # for file in drl_metrics:
        #     try:
        #         with open(metric_type_path + "/" + file, "r") as json_file:
        #             all_metric_infos = json.load(json_file)
        #             all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
        #         for src in all_metric_infos.keys():
        #             for dst, value in all_metric_infos[src].items():
        #                 for i in range(len(self.metric_factors)):
        #                     mertic_mean_values[i] += value[self.metric_factors[i]][0]
        #     except Exception as e:
        #         print(file)
        # compute throughput
        for file in throught_metrics:
            try:
                with open(throught_type_path + "/" + file, "r") as json_file:
                    all_metric_infos = json.load(json_file)
                    all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
                # for key, value in all_metric_infos.items():
                #     mertic_mean_values[-4] += value

                for src in all_metric_infos.keys():
                    for dst, value in all_metric_infos[src].items():
                        indications[0] += value[0]
            except Exception as e:
                print(file)
        # compute delay
        for file in delay_metrics:
            try:
                with open(delay_type_path + "/" + file, "r") as json_file:
                    all_metric_infos = json.load(json_file)
                    all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
                # for key, value in all_metric_infos.items():
                #     mertic_mean_values[-3] += value
                for src in all_metric_infos.keys():
                    for dst, value in all_metric_infos[src].items():
                        indications[1] += value[0]
            except Exception as e:
                print(file)
        # compute loss
        for file in loss_metrics:
            try:
                with open(loss_type_path + "/" + file, "r") as json_file:
                    all_metric_infos = json.load(json_file)
                    all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
                # for key, value in all_metric_infos.items():
                #     mertic_mean_values[-2] += value
                for src in all_metric_infos.keys():
                    for dst, value in all_metric_infos[src].items():
                        indications[2] += value[0]
            except Exception as e:
                print(file)

        # for file in bw_metrics:
        #     try:
        #         with open(bw_type_path + "/" + file, "r") as json_file:
        #             all_metric_infos = json.load(json_file)
        #             all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
        #         # for key, value in all_metric_infos.items():
        #         #     mertic_mean_values[-1] += value
        #         for src in all_metric_infos.keys():
        #             for dst, value in all_metric_infos[src].items():
        #                 mertic_mean_values[-1] += value
        #     except Exception as e:
        #         print(file)
        # for idx in range(len(mertic_mean_values)):
        #     mertic_mean_values[idx] = mertic_mean_values[idx] / total_file / (CF.NODES * (CF.NODES - 1))
        # mertic_mean_values[-4] = mertic_mean_values[-4] / total_throughput_file / (CF.NODES * (CF.NODES - 1))
        # mertic_mean_values[-3] = mertic_mean_values[-3] / total_throughput_file / (CF.NODES * (CF.NODES - 1))
        # # mertic_mean_values[-2] = mertic_mean_values[-2] / total_throughput_file / (CF.NODES * (CF.NODES - 1))
        # # mertic_mean_values[-1] =  1 + mertic_mean_values[-1] / total_throughput_file / len(CF.LINK_INFOS.keys())
        # mertic_mean_values[-2] = mertic_mean_values[-2] / total_throughput_file / (CF.NODES * (CF.NODES - 1))
        # if mertic_mean_values[-2] < 0:
        #     mertic_mean_values[-2] = abs(mertic_mean_values[-2])
        indications[0] = indications[0] / total_throughput_file / (CF.NODES * (CF.NODES - 1))
        indications[1] = indications[1] / total_throughput_file / (CF.NODES * (CF.NODES - 1))
        indications[2] = indications[2] / total_throughput_file / (CF.NODES * (CF.NODES - 1))

        print("================== %s ====================" % metric_type)
        print("the throughput|deay|loss len is : " , total_throughput_file, len(delay_metrics), len(loss_metrics))
        # print("average throughput of all links : ", mertic_mean_values[-4])
        # print("average delay of all links : ", mertic_mean_values[-3])
        # print("average loss of all links : ", mertic_mean_values[-2])
        # print("average use bw of all links : ", mertic_mean_values[-1])

        # infos = str("================== %s ====================" % metric_type) + "\n" +
        #         # str("mean free bw : %s" % (mertic_mean_values[0])) + "\n" + \
        #         # str("mean delay : %s" % (mertic_mean_values[1])) + "\n" + \
        #         # str("mean loss : %s" % (mertic_mean_values[2]) )+ "\n" + \
        #         # str("mean used_bw : %s" % ( mertic_mean_values[3])) + "\n" + \
        #         # str("mean drop : %s" % (mertic_mean_values[4])) + "\n" + \
        #         # str("mean errors : %s" % (mertic_mean_values[5])) + "\n" + \
        #         str("average throughput of all links %s: " % (indications[0]) + "\n" + \
        #         str("average delay of all links %s: " % (mertic_mean_values[-3])) + "\n" + \
        #         str("average loss of all links %s: " % (mertic_mean_values[-2])) + "\n" + \
        #         str("average use bw of all links %s: " % (mertic_mean_values[-1])) + "\n\n"\

        infos = str("================== %s ====================" % metric_type) + "\n" + \
                str("average throughput of all links %s: " % (indications[0])) + "\n" + \
                str("average delay of all links %s: " % (indications[1])) + "\n" + \
                str("average loss of all links %s: " % (indications[2])) + "\n"

        return indications, infos

    def save_data_to_file(self, data, infos):
        """

        :param data:
        :return:
        """
        # self.compare_file.write("-------------------- EXPERIMENT [%s] ----------------------" % self.compare_count)
        data_file = open(self.path_obj.get_save_path(type="COMPARE_PATH") + self.save_type +"_DATA.csv", "a")
        csv_writer = csv.writer(data_file)
        experiment_file = open(self.path_obj.get_save_path(type="COMPARE_PATH") + self.save_type + "_EXPERIMENT.txt", "a")
        if self.init_csv_header:
            csv_writer.writerow(
                ["",str("%s" % self.save_type),""])
            csv_writer.writerow(
                ["throughput", "delay", "packet_loss"])
            self.init_csv_header = False
        csv_writer.writerow(data)
        experiment_file.write(infos)
        print("\033[35;1m save data and info in %s \033[0m" % time.strftime("%Y%m%d", time.localtime(time.time())))
        data_file.close()
        experiment_file.close()

    def operator_drl_(self):
        """

        :return:
        """
        # if self.path_obj.show_drl_mean_metric and self.path_obj.show_drl_mean_throught and \
        #         self.path_obj.show_drl_mean_delay and self.path_obj.show_drl_mean_loss and \
        #         self.path_obj.show_drl_mean_use_bw:
        data = []
        # drl_data, drl_info = self.show_mean_metric_info(self.save_type, self.path_obj.past_drl_metric,
        #                                                 self.path_obj.past_drl_throught,
        #                                                 self.path_obj.past_drl_delay,
        #                                                 self.path_obj.past_drl_loss,
        #                                                 self.path_obj.past_drl_use_bw)
        drl_data, drl_info = self.show_mean_metric_info(self.save_type,
                                                        self.path_obj.past_drl_throught,
                                                        self.path_obj.past_drl_delay,
                                                        self.path_obj.past_drl_loss)
        data.extend(drl_data)
        infos = str(
            "=========================== EXPERICEMENT [%s] ==========================\n" % self.compare_count) + drl_info + "\n"
        self.save_data_to_file(data, infos)
        # self.path_obj.show_drl_mean_metric = False
        self.path_obj.show_drl_mean_throught = False
        self.path_obj.show_drl_mean_delay = False
        self.path_obj.show_drl_mean_loss = False
        # self.path_obj.show_drl_mean_use_bw = False
        self.compare_count += 1

    def operator_dijskra_(self):
        """

        :return:
        """
        # if self.path_obj.show_dijskra_mean_metric and self.path_obj.show_dijskra_mean_throught and \
        #         self.path_obj.show_dijskra_mean_delay and self.path_obj.show_dijskra_mean_loss and \
        #         self.path_obj.show_dijskra_mean_use_bw:
        data = []
        dijskra_data, dijskra_info = self.show_mean_metric_info(self.save_type,
                                                                self.path_obj.past_dijskra_metric,
                                                                self.path_obj.past_dijskra_throught,
                                                                self.path_obj.past_dijskra_delay)
        data.extend(dijskra_data)
        infos = str(
            "=========================== EXPERICEMENT [%s] ==========================\n" % self.compare_count) + dijskra_info + "\n"
        self.save_data_to_file(data, infos)
        # self.path_obj.show_dijskra_mean_metric = False
        self.path_obj.show_dijskra_mean_throught = False
        self.path_obj.show_dijskra_mean_delay = False
        self.path_obj.show_dijskra_mean_loss = False
        # self.path_obj.show_dijskra_mean_use_bw = False
        self.compare_count += 1

    def operator_ospf_(self):
        """

        :return:
        """
        # if self.path_obj.show_ospf_mean_metric and self.path_obj.show_ospf_mean_throught and \
        #         self.path_obj.show_ospf_mean_delay and self.path_obj.show_ospf_mean_loss and \
        #         self.path_obj.show_ospf_mean_use_bw:

        data = []
        ospf_data, ospf_info = self.show_mean_metric_info(self.save_type,
                                                          self.path_obj.past_ospf_throught,
                                                          self.path_obj.past_ospf_delay,
                                                          self.path_obj.past_ospf_loss)
        data.extend(ospf_data)
        infos = str(
            "=========================== EXPERICEMENT [%s] ==========================\n" % self.compare_count) + ospf_info + "\n"
        self.save_data_to_file(data, infos)
        # self.path_obj.show_ospf_mean_metric = False
        self.path_obj.show_ospf_mean_throught = False
        self.path_obj.show_ospf_mean_delay = False
        self.path_obj.show_ospf_mean_loss = False
        # self.path_obj.show_ospf_mean_use_bw = False
        self.compare_count += 1

    def operator_main(self):
        """
        main threading
        :return:
        """
        while True:
            if self.save_type == "DRL"  and self.path_obj.show_drl_mean_throught and \
                self.path_obj.show_drl_mean_delay and self.path_obj.show_drl_mean_loss:
                self.operator_drl_()
            if self.save_type == "DIJSKRA" and self.path_obj.show_dijskra_mean_throught and \
                self.path_obj.show_dijskra_mean_delay and self.path_obj.show_dijskra_mean_loss:
                self.operator_dijskra_()
            if self.save_type == "OSPF" and self.path_obj.show_ospf_mean_throught and \
                self.path_obj.show_ospf_mean_delay and self.path_obj.show_ospf_mean_loss:
                self.operator_ospf_()
            # hub.spawn(CF.OBSERVE_TIME)
            hub.sleep(CF.OBSERVE_TIME)

