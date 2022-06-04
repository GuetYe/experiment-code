import time
import ast
import json
import os
import random
import numpy as np
from pathlib import Path

# ==================================== #
#          Path Operator Class         #
# ==================================== #
class PathOperator:
    """

    """
    def __init__(self):
        print(" ..... LOAD PPO PATHOPERATOR ......")
        self.rnn_idx = 0
        self.dqn_idx = 0
        self.reward_idx = 0
        self.loss_idx = 0
        self.drl_file_idx = 0
        self.model_dir = "../modelDict"
        self.file_dir = "../predictTM"
        self.result_dir = "../results"
        self.rnn_name = "GRU"
        self.dqn_name = "DQN"
        self.ddpg_name = "DDPG"
        self.ppo_name = "PPO"
        self.reward_name = "REWARD"
        self.loss_name = "LOSS"
        self.log_name = "LOG"
        self.drl_path_name = "PATH"
        self.traffic_name = "PTM"
        self.result_name = "RESULT"
        self.rnn_path = None
        self.dqn_path = None
        self.ddpg_path = None
        self.ppo_path = None
        self.file_path = None
        self.reward_path = None
        self.loss_path = None
        self.log_path = None
        self.drl_path = None
        # path
        try:
            self.drl_idx = len(os.listdir(self.result_dir + "/" + self.generate_time_sub_dir() + "/" + self.drl_path_name)) + 1
        except Exception as e:
            self.drl_idx = 1

    def check_list_num_files(self, **kwargs):
        # if kwargs["type"] == "TM":
        #     return len(os.listdir(self.TM))
        # if kwargs["type"] == "PTM":
        #     return len(os.listdir(self.PTM))
        # if kwargs["type"] == "METRIC":
        #     return len(os.listdir(self.metric_file))
        if kwargs["type"] == "REWARD":
            return len(os.listdir(self.reward_path))

        if kwargs["type"] == "LOSS":
            return len(os.listdir(self.loss_path))

        if kwargs["type"] == "PATH":
            return len(os.listdir(self.drl_path))

    def create_rnn_model_dir(self):
        """
        :return:
        """
        self.rnn_path = self.model_dir + "/" + self.generate_time_sub_dir() + "/" + self.rnn_name
        # self.rnn_path = Path("/").joinpath(self.model_dir, self.generate_time_sub_dir()(), self.rnn_name)
        Path(self.rnn_path).mkdir(exist_ok=True, parents=True)

    def create_dqn_model_dir(self):
        """
        :return:
        """
        self.dqn_path = self.model_dir + "/" + self.generate_time_sub_dir() + "/" + self.dqn_name
        # self.dqn_path = Path("/").joinpath(self.model_dir, self.generate_time_sub_dir()(), self.dqn_name)
        Path(self.dqn_path).mkdir(exist_ok=True, parents=True)


    def create_ddpg_model_dir(self):
        """
        :return:
        """
        self.ddpg_path = self.model_dir + "/" + self.generate_time_sub_dir() + "/" + self.ddpg_name
        if not os.path.exists(self.ddpg_path):
            os.makedirs(self.ddpg_path)


    def create_ppo_model_dir(self):
        """
        :return:
        """
        self.ppo_path = self.model_dir + "/" + self.generate_time_sub_dir() + "/" + self.ppo_name
        if not os.path.exists(self.ppo_path):
            os.makedirs(self.ppo_path)



    def create_traffic_dir(self):
        """
            Create prediction traffic dir
        :return:
        """
        # print("....", self.generate_time_sub_dir()())
        self.file_path = self.file_dir + "/" + self.generate_time_sub_dir()
        # self.file_path = Path("/").joinpath(self.file_dir, self.generate_time_sub_dir()())
        Path(self.file_path).mkdir(exist_ok=True, parents=True)

    def create_reward_dir(self):
        """
            Create result dir
        :return:
        """
        self.reward_path = self.result_dir + "/" + self.generate_time_sub_dir() + "/" + self.reward_name
        if not os.path.exists(self.reward_path):
            os.makedirs(self.reward_path)

    def create_loss_dir(self):
        """
            Create result dir
        :return:
        """
        self.loss_path = self.result_dir + "/" + self.generate_time_sub_dir() + "/" + self.loss_name
        if not os.path.exists(self.loss_path):
            os.makedirs(self.loss_path)


    def create_log_dir(self):
        """
            Create loss pth
        :return:
        """
        self.log_path =  self.result_dir + "/" + self.generate_time_sub_dir() + "/" + self.log_name
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def create_drl_path_dir(self):
        """
            Create drl path dir
        """
        self.drl_path =  self.result_dir + "/" + self.generate_time_sub_dir() + "/" + self.drl_path_name + "/" + "test_" + str(self.drl_idx)
        if not os.path.exists(self.drl_path):
            os.makedirs(self.drl_path)

    def get_save_path(self, **kwargs):
        """
            Get corresponding path
        :param kwargs:
        :return:
        """
        if kwargs["type"] == self.rnn_name:
            self.rnn_idx += 1
            self.create_rnn_model_dir()
            return self.rnn_path + "/" + self.rnn_name + "_" + str(self.rnn_idx) + ".pth"
            # return Path("/").joinpath(self.rnn_path, self.rnn_name + "_" + str(self.rnn_idx))
        if kwargs["type"] == self.dqn_name:
            self.dqn_idx += 1
            self.create_dqn_model_dir()
            return self.dqn_path + "/" + self.dqn_name + "_" + str(self.dqn_idx) + ".pth"
            # return Path("/").joinpath(self.dqn_path, self.dqn_name + "_" + str(self.dqn_idx))

        if kwargs["type"] == self.ddpg_name:
            self.create_ddpg_model_dir()
            return self.ddpg_path + "/"

        if kwargs["type"] == self.ppo_name:
            self.create_ppo_model_dir()
            return self.ppo_path + "/"

        if kwargs["type"] == self.traffic_name:
            self.create_traffic_dir()
            return self.file_path

        if kwargs["type"] == self.reward_name:
            # self.result_idx += 1
            self.create_reward_dir()
            # self.reward_path = self.result_path + "/" + self.reward_name
            self.reward_idx = self.check_list_num_files(type=self.reward_name) + 1
            return self.reward_path +"/" + self.reward_name + "_" + str(self.reward_idx) + ".jpg"

        if kwargs["type"] == self.loss_name:
            # self.result_idx += 1
            self.create_loss_dir()
            self.loss_idx = self.check_list_num_files(type=self.loss_name) + 1
            return self.loss_path + "/" + self.loss_name + "_" + str(self.loss_idx) + ".jpg"

        if kwargs["type"] == self.log_name:
            self.create_log_dir()
            return self.log_path + "/"

        if kwargs["type"] == self.drl_path_name:
            # self.result_idx += 1
            self.create_drl_path_dir()
            self.drl_file_idx = self.check_list_num_files(type=self.drl_path_name) + 1
            return self.drl_path + "/" + self.drl_path_name + "_" + str(self.drl_file_idx) + ".json"


    def search_lastest_pth(self, late_pth, breakpoint_pths):
        """
            Get the lastest pth
        :return:
        """
        for item in breakpoint_pths:
            late_pth = max(late_pth, int(item.split(".")[0].split("_")[1]))
        return (late_pth, self.model_dir +"/" + self.generate_time_sub_dir() +"/" + self.dqn_name +  "/" + "DQN_" + str(late_pth) + ".pth")

    def load_breakpoint_pth(self, **kwargs):
        """
            Get breakpoint excute
        :return:
        """
        try:
            breakpoint_dir = self.model_dir +"/" + self.generate_time_sub_dir() +"/" + self.dqn_name
            breakpoint_pths = os.listdir(breakpoint_dir)

        except Exception as e:
            return
        latest_pth = 0
        for item in breakpoint_pths:
            latest_pth = max(latest_pth, int(item.split(".")[0].split("_")[1]))
        return (latest_pth, self.model_dir +"/" + self.generate_time_sub_dir() +"/" + self.dqn_name +  "/" + "DQN_" + str(latest_pth) + ".pth")


    def load_gru_model_pth(self):
        """
            return model pth
        :return
        """
        return self.model_dir + "/" + "GRU.pth"


    def generate_time_sub_dir(self):
        """
            Get time dir
        :return:
        """
        return time.strftime("%Y%m%d", time.localtime(time.time()))



# ==================================== #
#    Weight Factor Attenuation Class   #
# ==================================== #
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        print(" ..... LOAD PPO LINEARSCHDULE ......")
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# ==================================== #
#          File Operator Class         #
# ==================================== #
class FileOperator():
    def __init__(self):
        print(" ..... LOAD PPO FILEOPERATOR ......")
        self.metric_idx = 0
        self.each_total_max_num = 100
        self.current_count = 0
        # self.read_file = read_file
        self.write_file = None
        self.nodes = 14
        # self.metric_dir = r"metrics"
        self.metric_dir = r"E:\HLQ\metrics"
        self.file = None
        self.metric_file = None
        self.detect_tm_idx = 0
        self.predict_tm_idx = 0

    def check_is_empty(self, dir):
        """
            Check dir of metric is null
        :return:
        """
        file_list = os.listdir(dir)
        if file_list:
            return file_list
        return


    def read_metric_files(self, env_obj, metric_type):
        """
            read files
        :param traffic_file:
        :return:
        """
        start_time = time.time()
        file_list = self.check_is_empty(self.metric_dir)      # get 2021/9/29, ...
        if file_list is None:
            return
        for each_folder in file_list:
            metric_folder = self.metric_dir + "/" + each_folder
            metric_folder_files = self.check_is_empty(metric_folder)   # get metric_1.json, ...
            if metric_folder_files is None:
                return
            metric_folder_files.sort(key=lambda x: int(x.split('_')[1].split(".")[0]))
            if metric_type == "Test":
                metric_folder_files = random.sample(metric_folder_files, 10)
            for file in metric_folder_files:
                metric_folder_file = metric_folder + "/" + file
                try:
                    with open(metric_folder_file, "r") as json_file:      # get metrics/20210929/metric_1.json
                        all_metric_infos = json.load(json_file)
                        all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
                        env_obj.generate_traffic(all_metric_infos)          # generate traffic matrix
                        env_obj.generate_all_metric_infos(all_metric_infos, self.metric_idx)
                    self.metric_idx += 1
                        # env_obj.get_reward(all_metric_infos)  # generate reward
                except Exception as e:
                    print(metric_folder_file)
                    continue
        end_time = time.time()
        print("The number of loading traffic matrices is %s and takes %s seconds" % (self.metric_idx + 1, round(end_time - start_time, 2)))


    def write_traffic_file(self, traffics, write_file):
        """
            Save prediction traffic to file
        :param traffic_file:
        :param prediction_traffic:
        :return:
        """
        if self.current_count % self.each_total_max_num == 0:
            file_index = int(self.current_count // self.each_total_max_num)
            self.write_file = write_file + "/" + "AbilenePTM_" + str(file_index)
            # self.write_file = Path("/").joinpath(write_file, "AbilenePTM_" + str(file_index))
            print("\033[35;1m generate file 【%s】 in %s \033[0m" % (
            "AbilenePTM_" + str(file_index), time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(time.time()))))
        traffics = traffics.reshape((3, -1))
        with open(self.write_file, "a") as f:
            np.savetxt(f, traffics, fmt="%f", delimiter=" ")

        self.current_count += 1

    def write_info_to_json_file(self, **kwargs):
        """
            Save info to json file
        """
        if kwargs["type"] == "PATH":
            with open(kwargs["path"], "w") as json_file:
                json.dump(kwargs["data"], json_file, indent=2)


if __name__ == '__main__':
    exploration = LinearSchedule(300000, 0.1)
    import time
    for t in range(10000 * 30):
        value = exploration.value(t)
        print(value)

        # time.sleep(0.5)
