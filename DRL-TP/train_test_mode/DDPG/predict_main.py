import ast
import json
import os
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import PathOperator
from rnn_model import GRU
from dataSet import dataLoader
from drawTools import draw

#=======================================#
#      Train GRU algorithm offline      #
#=======================================#
class traffic_predict:
    def __init__(self):
        self.k_paths = 8
        self.nodes = 14
        self.metric_factors = ["free_bw", "delay", "packet_loss", "used_bw", "packet_drop", "packet_errors"]
        self.weight_factors = [0.6, 0.3, 0.1, 0.1, 0.1, 0.1]
        self.metric_dir = "../metrics"
        self.metric_idx = 0
        self.traffic = None
        self.seq = 7
        self.input_traffic = None
        self.target_traffic = None
        self.loader = dataLoader(batch_size=1)

        self.lr = 0.0001
        self.gru = GRU(input_dim=1, output_dim=1, hidden_size=256, num_layers=1, num_directions=False).cuda()
        self.optimizer = torch.optim.Adam(self.gru.parameters(), lr = self.lr)
        self.criterion = torch.nn.MSELoss().cuda()

        self.train_loss = []
        self.episodes = 100
        self.save_loss_frequency = 500
        self.save_model_frequency = 500

        self.path_obj = PathOperator()
        self.draw_obj = draw()

    def normal_traffic_matrics(self, traffic):
        """
            Maximum and minimum normalization
        :return:
        """
        if traffic is None:
            return
        for t in range(self.traffic.shape[0]):
            for idx in range(traffic.shape[1]):
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
                    traffic_metrics[idx][int(src) - 1][int(dst) - 1] = one_state_metric_infos[src][dst][self.metric_factors[0]][idx]
        traffic_metrics = traffic_metrics.reshape((1, self.k_paths, self.nodes, self.nodes))  # (1, 3, 14, 14)
        if self.traffic is not None:
            self.traffic = np.vstack((self.traffic, traffic_metrics))
        else:
            self.traffic = traffic_metrics

    def read_metric_traffic_files(self, type):
        """
            Read all metric files
        """
        start_time = time.time()
        file_list = os.listdir(self.metric_dir)
        if file_list is None:
            return
        for each_folder in file_list:
            metric_folder = self.metric_dir + "/" + each_folder
            metric_folder_files = os.listdir(metric_folder)  # get metric_1.json, ...
            if metric_folder_files is None:
                return
            metric_folder_files.sort(key=lambda x: int(x.split('_')[1].split(".")[0]))
            if type == "Test":
                metric_folder_files = random.sample(metric_folder_files, 50)
            for file in metric_folder_files:
                metric_folder_file = metric_folder + "/" + file
                try:
                    with open(metric_folder_file, "r") as json_file:  # get metrics/20210929/metric_1.json
                        all_metric_infos = json.load(json_file)
                        all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
                        self.generate_traffic(all_metric_infos)  # generate traffic matrix
                    self.metric_idx += 1
                    # env_obj.get_reward(all_metric_infos)  # generate reward
                except Exception as e:
                    print(metric_folder_file)
                    continue
        end_time = time.time()
        print("The number of loading traffic matrices is %s and takes %s seconds" % (
            self.metric_idx + 1, round(end_time - start_time, 2)))



    def generate_seq_traffic_matrix(self, type):
        """
            Get seq traffic
        :return
        """
        start_time = time.time()
        self.read_metric_traffic_files(type)
        self.traffic = self.normal_traffic_matrics(self.traffic)  # normalize traffic
        end_time = time.time()
        print("=======================================")
        print("load traffic spend %s s" % (round((end_time - start_time), 2)))
        print("raw traffic shape: ", self.traffic.shape)
        for t in range(0, self.traffic.shape[0] - self.seq):
            x = self.traffic[t: self.seq + t, :, :, :].reshape(1, self.seq, self.k_paths, -1)
            y = self.traffic[self.seq + t , :, :, :].reshape(1, 1, self.k_paths, -1)
            if self.input_traffic is None:
                self.input_traffic = x
                self.target_traffic = y
            else:
                self.input_traffic = np.vstack((self.input_traffic, x))
                self.target_traffic = np.vstack((self.target_traffic, y))
        print("input traffic shape: ", self.input_traffic.shape)
        print("target traffic shape: ", self.target_traffic.shape)

    def load_data(self, type):
        """
            Get traind loader
        :return
        """
        self.generate_seq_traffic_matrix(type)
        train_loader = self.loader.load_set(self.input_traffic, self.target_traffic)

        return train_loader

    def train_model(self, type="Train"):
        """
            Train model
        :return
        """
        train_loader = self.load_data(type)
        for epoch in range(self.episodes):
            hidden = self.gru.init_h_state(self.nodes * self.nodes).cuda()
            epoch_loss = 0.
            for i, data in enumerate(train_loader):
                input, target = data
                input = torch.tensor(input, dtype=torch.float).squeeze(0).permute(1, 2, 0).cuda()
                target = torch.tensor(target, dtype=torch.float).squeeze(0).cuda()
                out, out_hidden = self.gru(input, hidden)
                out = out.permute(2, 0, 1).squeeze(0)
                self.optimizer.zero_grad()
                loss = self.criterion(out, target.squeeze(0))
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                hidden = out_hidden.detach()
                if (i + 1) % self.save_loss_frequency == 0:
                    print("epoch[%s/%s] loss %s" % (epoch + 1, self.episodes, epoch_loss / (i + 1)))
                # save gru model
            if len(self.train_loss) == 0 or (epoch_loss / self.input_traffic.shape[0] < min(self.train_loss)):
                torch.save(self.gru.state_dict(), self.path_obj.get_save_path(type="GRU"))
                # self.train_loss.append(loss.item())
            self.train_loss.append(epoch_loss / self.input_traffic.shape[0])
        try:
            with open(self.path_obj.get_save_path(type="GRU").split(".")[0] + ".txt", "w") as f:
                np.savetxt(f, np.array(self.train_loss), fmt="%f", delimiter=" ")
        except Exception as e:
            pass
        print("Train done!!!", self.wacth_time())
        self.draw_curve(self.train_loss)

    def test_model(self, type="Test"):
        """
            Test model
        :return
        """
        test_loader = self.load_data(type)
        # load the trained model
        self.gru.load_state_dict(torch.load(self.path_obj.load_gru_model_pth()))
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                hidden = self.gru.init_h_state(self.nodes * self.nodes).cuda()
                input, target = data  # 7, 3, 196 | 1, 3, 196
                input = torch.tensor(input, dtype=torch.float).squeeze(0).permute(1, 2, 0).cuda()  # 3, 196, 7
                target = torch.tensor(target,  dtype=torch.float).squeeze(0).cuda()
                # out shape: 3, 196, 1 | out_hidden shape: 1, 196, 256
                out, out_hidden = self.gru(input, hidden)
                out = out.permute(2, 0, 1).squeeze(0)  # 3, 196
                with open(self.path_obj.get_save_path(type="PREDICT"), "w") as f:
                    np.savetxt(f, out.cpu().numpy(), fmt="%f", delimiter=" ")
                    np.savetxt(f, target.squeeze(0).cpu().numpy(), fmt="%f", delimiter=" ")

        print("Test Done!!!", self.wacth_time())


    def draw_curve(self, loss):
        """

        """
        plt.title("LOSS")
        plt.plot(np.arange(len(loss)), loss)
        plt.savefig(self.path_obj.get_save_path(type="GRU_LOSS"))
        plt.show()

    def wacth_time(self):
        """
        time counter
        :return:
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))



if __name__ == '__main__':
    predict = traffic_predict()
    print(predict.path_obj.get_save_path(type="GRU"))
