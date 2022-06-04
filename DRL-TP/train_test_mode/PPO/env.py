"""
function: Generate environment
date    : 2021/9/25
author  : hlq
"""
import random
import numpy as np
from utils import FileOperator

#=======================================#
#               Environment             #
#=======================================#
class Env:
    def __init__(self, metric_type="Train"):
        print(" ..... LOAD PPO ENV ......")
        self.traffic = None         # traffic (k_path, nodes, nodes)
        self.seq_interval = 3       # Sequence interval
        self.state = None           # state (metric_num, k_path, nodes, nodes)
        self.s = None               # init state
        self.idx = None             # state idx
        # self.total_cont_steps = 30
        self.total_cont_steps = 60
        self.nodes = 14
        self.action_dim = self.nodes * (self.nodes - 1)
        self.k_paths = 8

        self.all_metric_infos_dict = {}  # used to compute reward
        self.metric_factors = ["free_bw", "delay", "packet_loss", "used_bw", "packet_drop", "packet_errors"]
        # self.weight_factors = [0.6, 0.3, 0.1]
        self.weight_factors = [0.6, 0.3, 0.1, 0.1, 0.1, 0.1]
        # self.weight_factors = [0.5, 0.4, 0.4, 0.4, 0.4, 0.4]
        # self.weight_factors = [0.4, 0.5, 0.7, 0.7, 0.7, 0.7]

        self.beta_factor = [0.5, -0.4, -0.3, -0.3, -0.3, -0.3]
        # self.beta_factor = [0.4, -0.6, -0.6, -0.6, -0.6, -0.6]
        # self.beta_factor = [0.3, -1.0, -1.0, -1.0, -1.0, -1.0]
        # self.beta_factor = [1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        # self.beta_factor = [0.7, -0.7, -0.1, -0.1, -0.1, -0.1]
        # self.beta_factor = [0.4, -0.4, -0.4, -0.4, -0.4, -0.4]
        # self.beta_factor = [0.5, -0.3, -0.3, -0.3, -0.3, -0.3]
        # self.beta_factor = [0.4, -0.6, -0.6, -0.6, -0.6, -0.6]
        # self.beta_factor = [0.3, -1.0, -1.0, -1.0, -1.0, -1.0]
        self.operator_file_obj = FileOperator()
        self.operator_file_obj.read_metric_files(self, metric_type)
        self.generate_env()

    def reset(self):
        """
            init env
        :return:
        """
        self.cont_steps = 0                       # exit loop idx
        self.rnn_idx = 0                          # used to predict
        self.idx = random.randrange(self.state.shape[0])
        self.s = self.state[self.idx, :, :, :]
        return self.s

    def step(self, action):
        """
            Update env
        :return:
        """
        self.cont_steps += 1
        done = False
        if self.cont_steps >= self.total_cont_steps:
            done = True
        reward = self.get_reward(self.idx, action)
        # reward = self.P[self.idx][action]  # get reward
        self.idx += 1
        self.rnn_idx += 1
        if self.idx == self.state.shape[0]:
            self.idx = 0
        next_state = self.state[self.idx, :, :, :]  # get next_state
        # reset idx
        return next_state, reward, done, " "

    def predict_step(self, action):
        """
            Update predict state to get reward , done
        :param action:
        :return:
        """
        done = False
        self.cont_steps += 1
        if self.idx == self.state.shape[0]:
            self.idx = 0
        if self.cont_steps >= self.total_cont_steps:
            done = True
        reward = self.get_reward(self.idx + 1, action)
        # reward = self.P[self.idx + 1][action]  # get reward
        return reward, done, " "

    def generate_env(self):
        """
            generate env p
        :return:
        """
        self.generate_state()  # get state, shape(len/action_num, action_num, nodes, nodes)
        print("=================================")
        print("Environment generation succeeded!")
        print("total state space: %s" % self.state.shape[0])
        print("total state shape: ", self.state.shape)
        print("state space: ", self.state[0].shape)
        print("action space: %s" % self.action_dim)
        print("=================================")

    def normal_traffic_matrics(self, traffic):
        """
            Maximum and minimum normalization
        :return:
        """
        if traffic is None:
            return
        for t in range(self.traffic.shape[0]):
            for idx in range(traffic.shape[1]):
                # min_item = traffic[idx, :, :].min()
                # max_item = traffic[idx, :, :].max()
                mean_item = traffic[t, idx, :, :].mean()
                std_item = traffic[t, idx, :, :].std() + 1e-10
                traffic[t, idx, :, :] = (traffic[t, idx, :, :] - mean_item) / (std_item)

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
        free_bw = 0.001
        # In order to pursue the maximum value, it is necessary to take free_bw = 1 / free_bw
        if m_dict[src][dst][self.metric_factors[0]][idx] > free_bw:
            free_bw = 1 / m_dict[src][dst][self.metric_factors[0]][idx]
        else:
            free_bw = 1 / free_bw
        return free_bw * self.weight_factors[0] + \
               m_dict[src][dst][self.metric_factors[1]][idx] * self.weight_factors[1] + \
               m_dict[src][dst][self.metric_factors[2]][idx] * self.weight_factors[2] + \
                m_dict[src][dst][self.metric_factors[3]][idx] * self.weight_factors[3] + \
                m_dict[src][dst][self.metric_factors[4]][idx] * self.weight_factors[4] + \
                m_dict[src][dst][self.metric_factors[5]][idx] * self.weight_factors[5]

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
                    # traffic_metrics[idx][int(src) - 1][int(dst) - 1] = \
                    # one_state_metric_infos[src][dst][self.metric_factors[0]][idx]
                    traffic_metrics[idx][int(src) - 1][int(dst) - 1] = self.generate_matrix_elements(
                        one_state_metric_infos, src, dst, idx)
                    # self.get_reward(one_state_metric_infos, src, dst, idx)
        traffic_metrics = traffic_metrics.reshape((1, self.k_paths, self.nodes, self.nodes))  # (1, 3, 14, 14)
        if self.traffic is not None:
            self.traffic = np.vstack((self.traffic, traffic_metrics))
        else:
            self.traffic = traffic_metrics

    def normal_metric_infos(self, all_metrics_matrixs):
        """
            Normalize reward
        :param all_metrics_matrixs: free_bw, delay, loss, used_bw, drops, errors
        :return:
        """
        # rewards = []
        normal_all_metrics = []
        for idx in range(self.k_paths):
            one_path_metric_normal_infos = []
            for i in range(len(all_metrics_matrixs)):
                metric = all_metrics_matrixs[i]
                # with open("check_right.txt", "a") as f:
                #     np.savetxt(f, metric.reshape((3, -1)), fmt="%f", delimiter=" ")
                # print(metric)
                # del eye items of each metric
                del_eye_item_metric_matrix = np.delete(metric[idx, :, :], range(0, metric.shape[1] * metric.shape[2],
                                                                                (metric.shape[1] + 1))).reshape(
                    metric.shape[1], (metric.shape[2] - 1))
                # normal metric matrix
                # min_item = del_eye_item_metric_matrix.min()
                # max_item = del_eye_item_metric_matrix.max() + 1e-9
                mean_item = del_eye_item_metric_matrix.mean()
                std_item = del_eye_item_metric_matrix.std() + 1e-10
                for col in range(del_eye_item_metric_matrix.shape[0]):
                    del_eye_item_metric_matrix[col, :] = (del_eye_item_metric_matrix[col, :] - mean_item) / (std_item)
                one_path_metric_normal_infos.append(del_eye_item_metric_matrix)
                # total_reward += np.sum(del_eye_item_metric_matrix) * self.beta_factor[i]
            normal_all_metrics.append(one_path_metric_normal_infos)

        normal_all_metrics = np.array(normal_all_metrics)

        return normal_all_metrics

    def generate_all_metric_infos(self, one_state_metric_infos, metric_idx):
        """
            Generate all metric infos
        :param one_state_metric_infos:
        :param idx:
        :return:
        """
        # all_temp_metric = np.zeros((self.k_paths, self.nodes, self.nodes))
        self.all_metric_infos_dict.setdefault(metric_idx, [])
        # --- collect info to normalize reward--- #
        free_bw_matrix = np.zeros((self.k_paths, self.nodes, self.nodes))
        delay_matrix = np.zeros((self.k_paths, self.nodes, self.nodes))
        packet_loss_matrix = np.zeros((self.k_paths, self.nodes, self.nodes))
        used_bw_matrix = np.zeros((self.k_paths, self.nodes, self.nodes))
        packet_drop_matrix = np.zeros((self.k_paths, self.nodes, self.nodes))
        packet_errors_matrix = np.zeros((self.k_paths, self.nodes, self.nodes))
        for idx in range(self.k_paths):
            for src in one_state_metric_infos.keys():
                # self.reward.setdefault(src, {})
                for dst in one_state_metric_infos[src].keys():
                    free_bw_matrix[idx][int(src) - 1][int(dst) - 1] = \
                    one_state_metric_infos[src][dst][self.metric_factors[0]][idx]
                    delay_matrix[idx][int(src) - 1][int(dst) - 1] = \
                    one_state_metric_infos[src][dst][self.metric_factors[1]][idx]
                    packet_loss_matrix[idx][int(src) - 1][int(dst) - 1] = \
                    one_state_metric_infos[src][dst][self.metric_factors[2]][idx]
                    used_bw_matrix[idx][int(src) - 1][int(dst) - 1] = \
                    one_state_metric_infos[src][dst][self.metric_factors[3]][idx]
                    packet_drop_matrix[idx][int(src) - 1][int(dst) - 1] = \
                    one_state_metric_infos[src][dst][self.metric_factors[4]][idx]
                    packet_errors_matrix[idx][int(src) - 1][int(dst) - 1] = \
                    one_state_metric_infos[src][dst][self.metric_factors[5]][idx]

        # shape: 3, 6 , 14, 13
        nomal_metri_infos = self.normal_metric_infos(
            [free_bw_matrix, delay_matrix, packet_loss_matrix, used_bw_matrix, packet_drop_matrix,
             packet_errors_matrix])

        # shape: {0:[[p1,p2,p3], [p1,p2,p3], ... (14*13)], 1:[]}
        for src in range(nomal_metri_infos.shape[2]):  # 14
            for dst in range(nomal_metri_infos.shape[3]):  # 13
                src_dst_metric_price = []
                for k in range(nomal_metri_infos.shape[0]):  # 3
                    one_path_price = 0.
                    # check_list = []
                    for m in range(nomal_metri_infos.shape[
                                       1]):  # 6 ["free_bw", "delay", "packet_loss", "used_bw", "packet_drop", "packet_errors"]
                        # check_list.append(nomal_metri_infos[k, m, src, dst])
                        one_path_price += nomal_metri_infos[k, m, src, dst] * self.beta_factor[m]
                    src_dst_metric_price.append(one_path_price)
                self.all_metric_infos_dict[metric_idx].append(src_dst_metric_price)
                # print(check_list)
        # print(self.all_metric_infos_dict[metric_idx])
        # with open("check_right.txt", "a") as f:
        #     f.write("****************************************\n")
        #     np.savetxt(f, np.array(self.all_metric_infos_dict[metric_idx]), fmt="%f", delimiter=" ")

    def generate_state(self):
        """
            Generate state
        :return:
        """
        self.state = self.normal_traffic_matrics(self.traffic)  # normalize traffic


    def get_reward(self, idx, action):
        """
            Get reward
        :return:
        """
        reward = 0.
        for i in range(len(action)):
            reward += self.all_metric_infos_dict[idx][i][action[i]]
        return reward


    def get_state_idx(self):
        """
            Get state's idx
        :return:
        """
        return self.idx

    def get_total_state_space(self):
        """
            Get total state's space
        :return:
        """
        return self.state.shape[0]
