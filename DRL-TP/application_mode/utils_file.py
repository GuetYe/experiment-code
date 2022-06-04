import ast
import csv
import json
import os
import time
import configuration as CF

class SaveInfosToJson:
    """

    """
    # class params #
    # swicth_nodes = 23
    swicth_nodes = 14
    save_all_metrics_flag = False
    save_all_k_path_flag = True
    # all_k_paths = swicth_nodes * (swicth_nodes -1) * 3
    # each_total_tos = swicth_nodes * (swicth_nodes - 1)
    each_total_tos = 2**8 - 1
    def __init__(self):
        self.make_path = MakePath()

    def save_all_metrics_infos(self, metrics, type="METRIC"):
        """
            Save metric infos
        :param metrics:
        :return:
        """
        if type == "METRIC":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
            print("\033[33;1m metric's file [%s] saved at %s\033[0m" % (
            self.make_path.metric_idx, self.generate_print_time()))

        if type == "DRL_METRIC_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
            print("\033[33;1m drl metric's file [%s] saved at %s\033[0m" % (
                self.make_path.drl_metric_file_idx, self.generate_print_time()))
        if type == "DIJSKRA_METRIC_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
            print("\033[33;1m dijskra metric's file [%s] saved at %s\033[0m" % (
                self.make_path.dijskra_metric_file_idx, self.generate_print_time()))
        if type == "OSPF_METRIC_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
            print("\033[33;1m ospf metric's file [%s] saved at %s\033[0m" % (
                self.make_path.ospf_metric_file_idx, self.generate_print_time()))

        if type == "DRL_THROUGHTPUT_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m drl throughput's file [%s] saved at %s\033[0m" % (
                    self.make_path.drl_throughput_file_idx, self.generate_print_time()))
        if type == "DIJSKRA_THROUGHTPUT_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m dijskra throughput's file [%s] saved at %s\033[0m" % (
                    self.make_path.dijskra_throughput_file_idx, self.generate_print_time()))
        if type == "OSPF_THROUGHTPUT_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m ospf throughput's file [%s] saved at %s\033[0m" % (
                    self.make_path.ospf_throughput_file_idx, self.generate_print_time()))

        if type == "DRL_DELAY_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m drl delay's file [%s] saved at %s\033[0m" % (
                    self.make_path.drl_delay_file_idx, self.generate_print_time()))
        if type == "DIJSKRA_DELAY_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m dijskra delay's file [%s] saved at %s\033[0m" % (
                    self.make_path.dijskra_delay_file_idx, self.generate_print_time()))
        if type == "OSPF_DELAY_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m ospf delay's file [%s] saved at %s\033[0m" % (
                    self.make_path.ospf_delay_file_idx, self.generate_print_time()))

        if type == "DRL_LOSS_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m drl loss's file [%s] saved at %s\033[0m" % (
                    self.make_path.drl_loss_file_idx, self.generate_print_time()))
        if type == "DIJSKRA_LOSS_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m dijskra loss's file [%s] saved at %s\033[0m" % (
                    self.make_path.dijskra_loss_file_idx, self.generate_print_time()))
        if type == "OSPF_LOSS_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m ospf loss's file [%s] saved at %s\033[0m" % (
                    self.make_path.ospf_loss_file_idx, self.generate_print_time()))

        if type == "DRL_USE_BW_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m drl use bw's file [%s] saved at %s\033[0m" % (
                    self.make_path.drl_use_bw_file_idx, self.generate_print_time()))
        if type == "DIJSKRA_USE_BW_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m dijskra use bw's file [%s] saved at %s\033[0m" % (
                    self.make_path.dijskra_use_bw_file_idx, self.generate_print_time()))
        if type == "OSPF_USE_BW_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(metrics, json_file, indent=2)
                print("\033[33;1m ospf use bw's file [%s] saved at %s\033[0m" % (
                    self.make_path.ospf_use_bw_file_idx, self.generate_print_time()))

    def save_all_path_infos(self, paths, type="TOPO"):
        """

        :param paths:
        :param type:
        :return:
        """
        if type == "TOPO":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(paths, json_file, indent=2)
            print("\033[33;1m topo's file saved at %s\033[0m" % (self.generate_print_time()))
        if type == "DRL_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(paths, json_file, indent=2)
            print("\033[34;1m drl path's file [%s] saved at %s\033[0m" % (
                self.make_path.drl_file_idx, self.generate_print_time()))
        if type == "DIJSKRA_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(paths, json_file, indent=2)
            print("\033[34;1m dijskra path's file [%s] saved at %s\033[0m" % (
                self.make_path.dijskra_file_idx, self.generate_print_time()))
        if type == "OSPF_PATH":
            with open(self.make_path.get_save_path(type=type), "w") as json_file:
                json.dump(paths, json_file, indent=2)
            print("\033[34;1m ospf path's file [%s] saved at %s\033[0m" % (
                self.make_path.ospf_file_idx, self.generate_print_time()))

    # def save_all_alternative_path_infos(self, alternative_paths):
    #     """
    #
    #     :param alternative_paths:
    #     :return:
    #     """
    #     with open(self.make_path.get_save_path(type="TOPO"), "w") as json_file:
    #         json.dump(alternative_paths, json_file, indent=2)
    #     print("\033[33;1m topo's file saved at %s\033[0m" % (self.generate_print_time()))
    #
    # def save_all_drl_path_infos(self, drl_metric):
    #     """
    #
    #     :param drl_paths:
    #     :return:
    #     """
    #     with open(self.make_path.get_save_path(type="DRL_PATH"), "w") as json_file:
    #         json.dump(drl_metric, json_file, indent=2)
    #     print("\033[34;1m drl's file [%s] saved at %s\033[0m" % (
    #     self.generate_print_time(), self.make_path.drl_file_idx))
    #
    #
    #
    # def save_all_dijskra_metric_infos(self, dijskra_metric):
    #     """
    #         Save dijskra metric
    #     :param dijskra_metric:
    #     :return:
    #     """
    #     with open(self.make_path.get_save_path(type="DIJSKRA_PATH"), "w") as json_file:
    #         json.dump(dijskra_metric, json_file, indent=2)
    #     print("\033[34;1m dijskra's file [%s] saved at %s\033[0m" % (self.generate_print_time(), self.make_path.dijskra_file_idx))

    def generate_print_time(self):
        """

        """
        return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))

class MakePath:
    def __init__(self):
        self.metric_dir = "metrics"
        self.topo_dir = "topos"
        self.file_dir = "files"
        self.model_dir = "modelTar"
        # path dir
        self.drl_path_dir = "results/drl/drl_path"
        self.dijskra_path_dir = "results/dijskra/dijskra_path"
        self.ospf_path_dir = "results/ospf/ospf_path"
        # metric dir
        self.drl_metric_dir = "results/drl/drl_metric"
        self.dijskra_metric_dir = "results/dijskra/dijskra_metric"
        self.ospf_metric_dir = "results/ospf/ospf_metric"
        # throughput rate
        self.drl_throughput_dir = "results/drl/drl_throughput"
        self.dijskra_throughout_dir = "results/dijskra/dijskra_throughput"
        self.ospf_throughput_dir = "results/ospf/ospf_throughput"

        self.drl_delay_dir = "results/drl/drl_delay"
        self.dijskra_delay_dir = "results/dijskra/dijskra_delay"
        self.ospf_delay_dir = "results/ospf/ospf_delay"

        self.drl_loss_dir = "results/drl/drl_loss"
        self.dijskra_loss_dir = "results/dijskra/dijskra_loss"
        self.ospf_loss_dir = "results/ospf/ospf_loss"

        self.drl_use_bw_dir = "results/drl/drl_use_bw"
        self.dijskra_use_bw_dir = "results/dijskra/dijskra_use_bw"
        self.ospf_use_bw_dir = "results/ospf/ospf_use_bw"

        self.result_path_dir = "result"
        self.compare_path_dir = "results/Comparative_data"
        # index
        self.metric_idx = 0
        self.topo_idx = 0
        self.file_idx = 0
        self.drl_file_idx = 0
        self.dijskra_file_idx = 0
        self.ospf_file_idx = 0
        self.drl_metric_file_idx = 0
        self.dijskra_metric_file_idx = 0
        self.ospf_metric_file_idx = 0
        self.compare_file_idx = 0

        self.drl_throughput_file_idx = 0
        self.dijskra_throughput_file_idx = 0
        self.ospf_throughput_file_idx = 0
        self.drl_delay_file_idx = 0
        self.dijskra_delay_file_idx = 0
        self.ospf_delay_file_idx = 0
        self.drl_loss_file_idx = 0
        self.dijskra_loss_file_idx = 0
        self.ospf_loss_file_idx = 0
        self.result_file_idx = 0
        self.dijskra_use_bw_file_idx = 0
        self.ospf_use_bw_file_idx = 0
        self.result_use_bw_idx = 0

        # file path
        self.metric_file = None
        self.topo_file = None
        self.file_file = None
        self.drl_file = None
        self.result_file = None
        self.dijskra_file = None
        self.ospf_file = None
        # metric and throught file
        self.drl_metric_file = None
        self.dijskra_metric_file = None
        self.ospf_metric_file = None
        self.compare_file = None

        self.drl_throughput_file = None
        self.dijskra_throughout_file = None
        self.ospf_throughput_file =  None
        self.drl_delay_file = None
        self.dijskra_delay_file = None
        self.ospf_delay_file = None
        self.drl_loss_file = None
        self.dijskra_loss_file = None
        self.ospf_loss_file = None
        self.drl_use_bw_file = None
        self.dijskra_use_bw_file = None
        self.ospf_use_bw_file = None

        # compute drl
        self.show_drl_mean_metric = False
        self.show_dijskra_mean_metric = False
        self.show_ospf_mean_metric = False
        # print consponding infos
        self.show_drl_mean_throught = False
        self.show_dijskra_mean_throught = False
        self.show_ospf_mean_throught = False
        self.show_drl_mean_delay = False
        self.show_dijskra_mean_delay = False
        self.show_ospf_mean_delay = False
        self.show_drl_mean_loss = False
        self.show_dijskra_mean_loss = False
        self.show_ospf_mean_loss = False
        self.show_drl_mean_use_bw = False
        self.show_dijskra_mean_use_bw = False
        self.show_ospf_mean_use_bw = False
        # past dir
        self.past_drl_metric = None
        self.past_dijskra_metric = None
        self.past_ospf_metric = None
        self.past_drl_throught = None
        self.past_dijskra_throught =  None
        self.past_ospf_throught = None
        self.past_drl_delay = None
        self.past_dijskra_delay = None
        self.past_ospf_delay = None
        self.past_drl_loss = None
        self.past_dijskra_loss = None
        self.past_ospf_loss = None
        self.past_drl_use_bw = None
        self.past_dijskra_use_bw = None
        self.past_ospf_use_bw = None

        self.each_state = 1   # change dir idx
        # self.each_state = 10 # change dir idx
        # self.output_interval = 5
        # drl dir idx
        try:
            self.drl_idx = len(os.listdir(self.drl_path_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.drl_idx = 1
        # dijskra dir idx
        try:
            self.dijskra_idx = len(os.listdir(self.dijskra_path_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.dijskra_idx = 1
        # ospf dir idx
        try:
            self.ospf_idx = len(os.listdir(self.ospf_path_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.ospf_idx = 1
        # result dir idx
        try:
            self.result_idx = len(os.listdir(self.result_path_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.result_idx = 1
        # dlr metric idx
        try:
            self.drl_metric_idx = len(os.listdir(self.drl_metric_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.drl_metric_idx = 1
        # dijskra metric idx
        try:
            self.dijskra_metric_idx = len(os.listdir(self.dijskra_metric_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.dijskra_metric_idx = 1
        # ospf metric idx
        try:
            self.ospf_metric_idx = len(os.listdir(self.ospf_metric_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.ospf_metric_idx = 1

        try:
            self.drl_throughtput_idx = len(os.listdir(self.drl_throughput_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.drl_throughtput_idx = 1
        try:
            self.dijskra_throughtput_idx = len(os.listdir(self.dijskra_throughout_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.dijskra_throughtput_idx = 1
        try:
            self.ospf_throughtput_idx = len(os.listdir(self.ospf_throughput_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.ospf_throughtput_idx = 1

        try:
            self.drl_delay_idx = len(os.listdir(self.drl_delay_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.drl_delay_idx = 1
        try:
            self.dijskra_delay_idx = len(os.listdir(self.dijskra_delay_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.dijskra_delay_idx = 1
        try:
            self.ospf_delay_idx = len(os.listdir(self.ospf_delay_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.ospf_delay_idx = 1

        try:
            self.drl_loss_idx = len(os.listdir(self.drl_loss_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.drl_loss_idx = 1
        try:
            self.dijskra_loss_idx = len(os.listdir(self.dijskra_loss_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.dijskra_loss_idx = 1
        try:
            self.ospf_loss_idx = len(os.listdir(self.ospf_loss_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.ospf_loss_idx = 1


        try:
            self.drl_use_bw_idx = len(os.listdir(self.drl_use_bw_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.drl_use_bw_idx = 1
        try:
            self.dijskra_use_bw_idx = len(os.listdir(self.dijskra_use_bw_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.dijskra_use_bw_idx = 1
        try:
            self.ospf_use_bw_idx = len(os.listdir(self.ospf_use_bw_dir + "/" + self.generate_time_dir())) + 1
        except Exception as e:
            self.ospf_use_bw_idx = 1



    def create_metric_dir(self):
        """

        """
        self.metric_idx += 1
        self.metric_file = self.metric_dir + "/" + self.generate_time_dir()
        if not os.path.exists(self.metric_file):
            os.makedirs(self.metric_file)
        # Path(self.metric_dir).mkdir(exist_ok=True, parents=True)


    def create_topo_dir(self):
        """
            Create topo dir
        :return:
        """
        # self.topo_idx += 1
        self.topo_file = self.topo_dir + "/" + self.generate_time_dir()
        if not os.path.exists(self.topo_file):
            os.makedirs(self.topo_file)

    def create_drl_dir(self):
        """
            Create drl dir
        :return:
        """
        # self.topo_idx += 1
        self.drl_file = self.drl_path_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(self.drl_idx)
        if not os.path.exists(self.drl_file):
            os.makedirs(self.drl_file)

    def create_dijskra_dir(self):
        """
            Create dijskra dir
        :return:
        """
        # self.topo_idx += 1
        self.dijskra_file = self.dijskra_path_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(self.dijskra_idx)
        if not os.path.exists(self.dijskra_file):
            os.makedirs(self.dijskra_file)


    def create_ospf_dir(self):
        """
            Create dijskra dir
        :return:
        """
        # self.topo_idx += 1
        self.ospf_file = self.ospf_path_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(self.ospf_idx)
        if not os.path.exists(self.ospf_file):
            os.makedirs(self.ospf_file)

    def create_compare_data_drl_dir(self):
        """
            Create compare dir
        :return:
        """
        self.drl_metric_file = self.drl_metric_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(self.drl_metric_idx)
        if not os.path.exists(self.drl_metric_file):
            os.makedirs(self.drl_metric_file)

    def create_compare_data_dijskra_dir(self):
        """
            Create compare dir
        :return:
        """
        self.dijskra_metric_file = self.dijskra_metric_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
            self.dijskra_metric_idx)
        if not os.path.exists(self.dijskra_metric_file):
            os.makedirs(self.dijskra_metric_file)

    def create_compare_data_ospf_dir(self):
        """
            Create compare dir
        :return:
        """
        self.ospf_metric_file = self.ospf_metric_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
            self.ospf_metric_idx)
        if not os.path.exists(self.ospf_metric_file):
            os.makedirs(self.ospf_metric_file)

    def create_drl_throught_dir(self):
        """
            Create drl dir
        :return:
        """
        self.drl_throughput_file = self.drl_throughput_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
            self.drl_throughtput_idx)
        if not os.path.exists(self.drl_throughput_file):
            os.makedirs(self.drl_throughput_file)

    def create_dijskra_throught_dir(self):
        """
            Create dijskra dir
        :return:
        """
        self.dijskra_throughput_file = self.dijskra_throughout_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
            self.dijskra_throughtput_idx)
        if not os.path.exists(self.dijskra_throughput_file):
            os.makedirs(self.dijskra_throughput_file)

    def create_ospf_throught_dir(self):
        """
            Create ospf dir
        :return:
        """
        self.ospf_throughput_file = self.ospf_throughput_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
            self.ospf_throughtput_idx)
        if not os.path.exists(self.ospf_throughput_file):
            os.makedirs(self.ospf_throughput_file)



    def create_drl_delay_dir(self):
        """
            Create drl dir
        :return:
        """
        self.drl_delay_file = self.drl_delay_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
            self.drl_delay_idx)
        if not os.path.exists(self.drl_delay_file):
            os.makedirs(self.drl_delay_file)

    def create_dijskra_delay_dir(self):
        """
            Create dijskra dir
        :return:
        """
        self.dijskra_delay_file = self.dijskra_delay_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
            self.dijskra_delay_idx)
        if not os.path.exists(self.dijskra_delay_file):
            os.makedirs(self.dijskra_delay_file)

    def create_ospf_delay_dir(self):
        """
            Create ospf dir
        :return:
        """
        self.ospf_delay_file = self.ospf_delay_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
            self.ospf_delay_idx)
        if not os.path.exists(self.ospf_delay_file):
            os.makedirs(self.ospf_delay_file)


    def create_drl_loss_dir(self):
        """
            Create drl dir
        :return:
        """
        self.drl_loss_file = self.drl_loss_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
            self.drl_loss_idx)
        if not os.path.exists(self.drl_loss_file):
            os.makedirs(self.drl_loss_file)

    def create_dijskra_loss_dir(self):
        """
            Create dijskra dir
        :return:
        """
        self.dijskra_loss_file = self.dijskra_loss_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
            self.dijskra_loss_idx)
        if not os.path.exists(self.dijskra_loss_file):
            os.makedirs(self.dijskra_loss_file)

    def create_ospf_loss_dir(self):
        """
            Create ospf dir
        :return:
        """
        self.ospf_loss_file = self.ospf_loss_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
            self.ospf_loss_idx)
        if not os.path.exists(self.ospf_loss_file):
            os.makedirs(self.ospf_loss_file)



    def create_drl_use_bw_dir(self):
        """
            Create drl dir
        :return:
        """
        self.drl_use_bw_file = self.drl_use_bw_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
            self.drl_use_bw_idx)
        if not os.path.exists(self.drl_use_bw_file):
            os.makedirs(self.drl_use_bw_file)

    def create_dijskra_use_bw_dir(self):
        """
            Create dijskra dir
        :return:
        """
        self.dijskra_use_bw_file = self.dijskra_use_bw_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
            self.dijskra_use_bw_idx)
        if not os.path.exists(self.dijskra_use_bw_file):
            os.makedirs(self.dijskra_use_bw_file)

    def create_ospf_use_bw_dir(self):
        """
            Create ospf dir
        :return:
        """
        self.ospf_use_bw_file = self.ospf_use_bw_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
            self.ospf_use_bw_idx)
        if not os.path.exists(self.ospf_use_bw_file):
            os.makedirs(self.ospf_use_bw_file)


    def create_compare_dir(self):
        """

        :return:
        """
        self.compare_file = self.compare_path_dir + "/" + self.generate_time_dir()
        if not os.path.exists(self.compare_file):
            os.makedirs(self.compare_file)


    def create_result_dir(self):
        """
            Create result dir
        :return:
        """
        self.result_file = self.result_path_dir + "/" + self.generate_time_dir() + "/" + "result_" + str(self.result_idx)
        if not os.path.exists(self.result_file):
            os.makedirs(self.result_file)

    def create_file_dir(self):
        """

        """
        pass


    def check_list_num_files(self, **kwargs):
        if kwargs["type"] == "TOPO":
            return len(os.listdir(self.topo_file))
        if kwargs["type"] == "METRIC":
            return len(os.listdir(self.metric_file))
        if kwargs["type"] == "DRL_PATH":
            return len(os.listdir(self.drl_file))
        if kwargs["type"] == "DIJSKRA_APTH":
            return len(os.listdir(self.dijskra_file))
        if kwargs["type"] == "OSPF_PATH":
            return len(os.listdir(self.ospf_file))
        if kwargs["type"] == "DRL_METRIC_PATH":
            return len(os.listdir(self.drl_metric_file))
        if kwargs["type"] == "DIJSKRA_METRIC_PATH":
            return len(os.listdir(self.dijskra_metric_file))
        if kwargs["type"] == "OSPF_METRIC_PATH":
            return len(os.listdir(self.ospf_metric_file))
        if kwargs["type"] == "DRL_THROUGHTPUT_PATH":
            return len(os.listdir(self.drl_throughput_file))
        if kwargs["type"] == "DIJSKRA_THROUGHTPUT_PATH":
            return len(os.listdir(self.dijskra_throughput_file))
        if kwargs["type"] == "OSPF_THROUGHTPUT_PATH":
            return len(os.listdir(self.ospf_throughput_file))
        if kwargs["type"] == "DRL_DELAY_PATH":
            return len(os.listdir(self.drl_delay_file))
        if kwargs["type"] == "DIJSKRA_DELAY_PATH":
            return len(os.listdir(self.dijskra_delay_file))
        if kwargs["type"] == "OSPF_DELAY_PATH":
            return len(os.listdir(self.ospf_delay_file))
        if kwargs["type"] == "DRL_LOSS_PATH":
            return len(os.listdir(self.drl_loss_file))
        if kwargs["type"] == "DIJSKRA_LOSS_PATH":
            return len(os.listdir(self.dijskra_loss_file))
        if kwargs["type"] == "OSPF_LOSS_PATH":
            return len(os.listdir(self.ospf_loss_file))
        if kwargs["type"] == "DRL_USE_BW_PATH":
            return len(os.listdir(self.drl_use_bw_file))
        if kwargs["type"] == "DIJSKRA_USE_BW_PATH":
            return len(os.listdir(self.dijskra_use_bw_file))
        if kwargs["type"] == "OSPF_USE_BW_PATH":
            return len(os.listdir(self.ospf_use_bw_file))
        # if kwargs["type"] == "COMPARE_PATH":
        #     return len(os.listdir(self.compare_file))
        if kwargs["type"] == "FILE":
            return len(os.listdir(self.file_file))


    def get_save_path(self, **kwargs):
        """
        Get save path
        """
        if kwargs["type"] == "TOPO":
            self.create_topo_dir()
            return self.topo_file + "/" + "topo.json"
        if kwargs["type"] == "METRIC":
            self.create_metric_dir()
            self.metric_idx = self.check_list_num_files(type="METRIC") + 1
            return self.metric_file + "/" + "metric_" + str(self.metric_idx) + ".json"

        if kwargs["type"] == "DRL_PATH":
            self.create_drl_dir()
            self.drl_file_idx = self.check_list_num_files(type="DRL_PATH") + 1
            if self.drl_file_idx % self.each_state == 0:
                # self.past_drl_metric = self.drl_path_dir + "/" + self.generate_time_dir() + "/" + "result_" + str(self.drl_idx)
                self.drl_idx += 1
                # self.show_drl_mean_metric = True
            return self.drl_file + "/" + "drl_" + str(self.drl_file_idx) + ".json"

        if kwargs["type"] == "DIJSKRA_PATH":
            self.create_dijskra_dir()
            self.dijskra_file_idx = self.check_list_num_files(type="DIJSKRA_APTH") + 1
            if self.dijskra_file_idx % self.each_state == 0:
                self.dijskra_idx += 1
            return self.dijskra_file + "/" + "dijskra_" + str(self.dijskra_file_idx) + ".json"

        if kwargs["type"] == "OSPF_PATH":
            self.create_ospf_dir()
            self.ospf_file_idx = self.check_list_num_files(type="OSPF_PATH") + 1
            if self.ospf_file_idx % self.each_state == 0:
                self.ospf_idx += 1
            return self.ospf_file + "/" + "ospf_" + str(self.ospf_file_idx) + ".json"

        if kwargs["type"] == "DRL":
            self.create_ospf_dir()
            self.ospf_file_idx = self.check_list_num_files(type="OSPF_PATH") + 1
            if self.ospf_file_idx % self.each_state == 0:
                self.ospf_idx += 1
            return self.ospf_file + "/" + "ospf_" + str(self.ospf_file_idx) + ".json"

        if kwargs["type"] == "DRL_METRIC_PATH":
            self.create_compare_data_drl_dir()
            self.drl_metric_file_idx = self.check_list_num_files(type="DRL_METRIC_PATH") + 1
            if self.drl_metric_file_idx % self.each_state == 0:
                self.past_drl_metric = self.drl_metric_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
                    self.drl_metric_idx)
                self.drl_metric_idx += 1
                self.show_drl_mean_metric = True
            return self.drl_metric_file + "/" + "drl_metric_" + str(self.drl_metric_file_idx) + ".json"

        if kwargs["type"] == "DIJSKRA_METRIC_PATH":
            self.create_compare_data_dijskra_dir()
            self.dijskra_metric_file_idx = self.check_list_num_files(type="DIJSKRA_METRIC_PATH") + 1
            if self.dijskra_metric_file_idx % self.each_state == 0:
                self.past_dijskra_metric = self.dijskra_metric_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
                    self.dijskra_metric_idx)
                self.dijskra_metric_idx += 1
                self.show_dijskra_mean_metric = True
            return self.dijskra_metric_file + "/" + "dijskra_metric_" + str(self.dijskra_metric_file_idx) + ".json"

        if kwargs["type"] == "OSPF_METRIC_PATH":
            self.create_compare_data_ospf_dir()
            self.ospf_metric_file_idx = self.check_list_num_files(type="OSPF_METRIC_PATH") + 1
            if self.ospf_metric_file_idx % self.each_state == 0:
                self.past_ospf_metric = self.ospf_metric_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
                    self.ospf_metric_idx)
                self.ospf_metric_idx += 1
                self.show_ospf_mean_metric = True
            return self.ospf_metric_file + "/" + "ospf_metric_" + str(self.ospf_metric_file_idx) + ".json"

        if kwargs["type"] == "DRL_THROUGHTPUT_PATH":
            self.create_drl_throught_dir()
            self.drl_throughput_file_idx = self.check_list_num_files(type="DRL_THROUGHTPUT_PATH") + 1
            if self.drl_throughput_file_idx % self.each_state == 0:
                self.past_drl_throught = self.drl_throughput_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
                    self.drl_throughtput_idx)
                self.drl_throughtput_idx += 1
                self.show_drl_mean_throught = True
            return self.drl_throughput_file + "/" + "drl_throughtput_" + str(self.drl_throughput_file_idx) + ".json"

        if kwargs["type"] == "DIJSKRA_THROUGHTPUT_PATH":
            self.create_dijskra_throught_dir()
            self.dijskra_throughput_file_idx = self.check_list_num_files(type="DIJSKRA_THROUGHTPUT_PATH") + 1
            if self.dijskra_throughput_file_idx % self.each_state == 0:
                self.past_dijskra_throught = self.dijskra_throughout_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
                    self.dijskra_throughtput_idx)
                self.dijskra_throughtput_idx += 1
                self.show_dijskra_mean_throught = True
            return self.dijskra_throughput_file + "/" + "dijskra_throughtput_" + str(self.dijskra_throughput_file_idx) + ".json"

        if kwargs["type"] == "OSPF_THROUGHTPUT_PATH":
            self.create_ospf_throught_dir()
            self.ospf_throughput_file_idx = self.check_list_num_files(type="OSPF_THROUGHTPUT_PATH") + 1
            if self.ospf_throughput_file_idx % self.each_state == 0:
                self.past_ospf_throught = self.ospf_throughput_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
                    self.ospf_throughtput_idx)
                self.ospf_throughtput_idx += 1
                self.show_ospf_mean_throught = True
            return self.ospf_throughput_file + "/" + "ospf_throughtput_" + str(self.ospf_throughput_file_idx) + ".json"

        if kwargs["type"] == "DRL_DELAY_PATH":
            self.create_drl_delay_dir()
            self.drl_delay_file_idx = self.check_list_num_files(type="DRL_DELAY_PATH") + 1
            if self.drl_delay_file_idx % self.each_state == 0:
                self.past_drl_delay = self.drl_delay_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
                    self.drl_delay_idx)
                self.drl_delay_idx += 1
                self.show_drl_mean_delay = True
            return self.drl_delay_file + "/" + "drl_delay_" + str(self.drl_delay_file_idx) + ".json"

        if kwargs["type"] == "DIJSKRA_DELAY_PATH":
            self.create_dijskra_delay_dir()
            self.dijskra_delay_file_idx = self.check_list_num_files(type="DIJSKRA_DELAY_PATH") + 1
            if self.dijskra_delay_file_idx % self.each_state == 0:
                self.past_dijskra_delay = self.dijskra_delay_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
                    self.dijskra_delay_idx)
                self.dijskra_delay_idx += 1
                self.show_dijskra_mean_delay = True
            return self.dijskra_delay_file + "/" + "dijskra_delay_" + str(self.dijskra_delay_file_idx) + ".json"

        if kwargs["type"] == "OSPF_DELAY_PATH":
            self.create_ospf_delay_dir()
            self.ospf_delay_file_idx = self.check_list_num_files(type="OSPF_DELAY_PATH") + 1
            if self.ospf_delay_file_idx % self.each_state == 0:
                self.past_ospf_delay = self.ospf_delay_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
                    self.ospf_delay_idx)
                self.ospf_delay_idx += 1
                self.show_ospf_mean_delay = True
            return self.ospf_delay_file + "/" + "ospf_delay_" + str(self.ospf_delay_file_idx) + ".json"

        if kwargs["type"] == "DRL_LOSS_PATH":
            self.create_drl_loss_dir()
            self.drl_loss_file_idx = self.check_list_num_files(type="DRL_LOSS_PATH") + 1
            if self.drl_loss_file_idx % self.each_state == 0:
                self.past_drl_loss = self.drl_loss_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
                    self.drl_loss_idx)
                self.drl_loss_idx += 1
                self.show_drl_mean_loss = True
            return self.drl_loss_file + "/" + "drl_loss_" + str(self.drl_loss_file_idx) + ".json"

        if kwargs["type"] == "DIJSKRA_LOSS_PATH":
            self.create_dijskra_loss_dir()
            self.dijskra_loss_file_idx = self.check_list_num_files(type="DIJSKRA_LOSS_PATH") + 1
            if self.dijskra_loss_file_idx % self.each_state == 0:
                self.past_dijskra_loss = self.dijskra_loss_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
                    self.dijskra_loss_idx)
                self.dijskra_loss_idx += 1
                self.show_dijskra_mean_loss = True
            return self.dijskra_loss_file + "/" + "dijskra_loss_" + str(self.dijskra_loss_file_idx) + ".json"

        if kwargs["type"] == "OSPF_LOSS_PATH":
            self.create_ospf_loss_dir()
            self.ospf_loss_file_idx = self.check_list_num_files(type="OSPF_LOSS_PATH") + 1
            if self.ospf_loss_file_idx % self.each_state == 0:
                self.past_ospf_loss = self.ospf_loss_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
                    self.ospf_loss_idx)
                self.ospf_loss_idx += 1
                self.show_ospf_mean_loss = True
            return self.ospf_loss_file + "/" + "ospf_loss_" + str(self.ospf_loss_file_idx) + ".json"


        if kwargs["type"] == "DRL_USE_BW_PATH":
            self.create_drl_use_bw_dir()
            self.drl_use_bw_file_idx = self.check_list_num_files(type="DRL_USE_BW_PATH") + 1
            if self.drl_use_bw_file_idx % self.each_state == 0:
                self.past_drl_use_bw = self.drl_use_bw_dir + "/" + self.generate_time_dir() + "/" + "drl_" + str(
                    self.drl_use_bw_idx)
                self.drl_use_bw_idx += 1
                self.show_drl_mean_use_bw = True
            return self.drl_use_bw_file + "/" + "drl_use_bw_" + str(self.drl_use_bw_file_idx) + ".json"

        if kwargs["type"] == "DIJSKRA_USE_BW_PATH":
            self.create_dijskra_use_bw_dir()
            self.dijskra_use_bw_file_idx = self.check_list_num_files(type="DIJSKRA_USE_BW_PATH") + 1
            if self.dijskra_use_bw_file_idx % self.each_state == 0:
                self.past_dijskra_use_bw = self.dijskra_use_bw_dir + "/" + self.generate_time_dir() + "/" + "dijskra_" + str(
                    self.dijskra_use_bw_idx)
                self.dijskra_use_bw_idx += 1
                self.show_dijskra_mean_use_bw = True
            return self.dijskra_use_bw_file + "/" + "dijskra_use_bw_" + str(self.dijskra_use_bw_file_idx) + ".json"

        if kwargs["type"] == "OSPF_USE_BW_PATH":
            self.create_ospf_use_bw_dir()
            self.ospf_use_bw_file_idx = self.check_list_num_files(type="OSPF_USE_BW_PATH") + 1
            if self.ospf_use_bw_file_idx % self.each_state == 0:
                self.past_ospf_use_bw = self.ospf_use_bw_dir + "/" + self.generate_time_dir() + "/" + "ospf_" + str(
                    self.ospf_use_bw_idx)
                self.ospf_use_bw_idx += 1
                self.show_ospf_mean_use_bw = True
            return self.ospf_use_bw_file + "/" + "ospf_use_bw_" + str(self.ospf_use_bw_file_idx) + ".json"


        if kwargs["type"] == "COMPARE_PATH":
            self.create_compare_dir()
            return self.compare_file + "/"

        if kwargs["type"] == "RESULT_PATH":
            self.create_result_dir()
            self.result_file_idx = self.check_list_num_files(type="OSPF_PATH") + 1
            if self.result_file_idx % self.each_state == 0:
                self.result_idx += 1
            return self.result_file + "/" + "result_" + str(self.result_file_idx) + ".json"

        if kwargs["type"] == "FILE":
            self.create_file_dir()
            return


    def get_topo_data(self):
        """
            Get topo's info.
        :return:
        """
        try:
            # with open(self.topo_dir + "/" + self.generate_time_dir() + "/" + "topo.json") as json_file:
            #     all_metric_infos = json.load(json_file)
            #     all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
            # return all_metric_infos
            with open(self.topo_dir + "/" + "20220418" + "/" + "topo.json") as json_file:
                all_metric_infos = json.load(json_file)
                all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
            return all_metric_infos
        except:
            return None

    def get_indicators_path(self):
        """

        :return:
        """


    def load_dqn_model_pth(self):
        """
            Load dqn pth.tar
        :return:
        """
        return self.model_dir + "/" + "DQN.pth.tar"
        # return self.model_dir + "/" + "v4/" + "DDPG.pth.tar"


    def generate_time_dir(self):
        """
        Get time dir
        """
        return time.strftime("%Y%m%d", time.localtime(time.time()))



if __name__ == '__main__':
    to_json = SaveInfosToJson()
    to_json.save_all_metrics_infos(" ")
