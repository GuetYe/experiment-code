# _*_coding:utf-8 _*_
"""
function: Generate environment
date    : 2021/9/25
author  : hlq
"""
import numpy as np

class Env:
    def __init__(self):
        self.nodes = 14
        self.k_paths = 8
        self.action_dim = self.nodes * (self.nodes - 1)
        self.metric_factors = ["free_bw", "delay", "packet_loss", "used_bw", "packet_drop", "packet_errors"]
        self.tf_factor = [0.6, 0.3, 0.1, 0.1, 0.1, 0.1]                 # factor of traffic matrix
        self.rf_factor = [0.5, -0.4, -0.3, -0.3, -0.3, -0.3]            # factor of reward's function
        self.MIN_FREE_BW = 1e-4
        self.all_metric_infos_dict = {}

    def normal_traffic_matrics(self, traffic):
        """
            Std_Mean technology to normalize traffic matrix.
        :return:
        """
        if traffic is None:
            return
        for idx in range(traffic.shape[0]):
            mean_item = traffic[idx, :, :].mean()
            std_item = traffic[idx, :, :].std() + 1e-10
            traffic[idx, :, :] = (traffic[idx, :, :] - mean_item) / (std_item)

        return traffic

    def generate_matrix_elements(self, m_dict, src, dst, idx):
        """
            Generate each element of the traffic matrix :formula = bw * 0.6 + delay * 0.3 + loss * 0.1
        :param m_dict:
        :param src:
        :param dst:
        :param idx:
        :return:
        """

        # In order to pursue the maximum value, it is necessary to take free_bw = 1 / free_bw
        if m_dict[src][dst][self.metric_factors[0]][idx] > self.MIN_FREE_BW:
            free_bw = 1 / m_dict[src][dst][self.metric_factors[0]][idx]
        else:
            free_bw = 1 / self.MIN_FREE_BW
        return free_bw * self.tf_factor[0] + \
               m_dict[src][dst][self.metric_factors[1]][idx] * self.tf_factor[1] + \
               m_dict[src][dst][self.metric_factors[2]][idx] * self.tf_factor[2] + \
               m_dict[src][dst][self.metric_factors[3]][idx] * self.tf_factor[3] + \
               m_dict[src][dst][self.metric_factors[4]][idx] * self.tf_factor[4] + \
               m_dict[src][dst][self.metric_factors[5]][idx] * self.tf_factor[5]


    def generate_traffic(self, one_state_metric_infos):
        """
            Generate traffic matrix
        :param one_state_metric_infos:
        :return:
        """
        traffic_metrics = np.zeros((self.k_paths, self.nodes, self.nodes))
        for src in one_state_metric_infos.keys():
            for dst in one_state_metric_infos[src].keys():
                for idx in range(self.k_paths):
                    traffic_metrics[idx][int(src) - 1][int(dst) - 1] = \
                        self.generate_matrix_elements(one_state_metric_infos, src, dst, idx)

        state = self.normal_traffic_matrics(traffic=traffic_metrics)
        return state

