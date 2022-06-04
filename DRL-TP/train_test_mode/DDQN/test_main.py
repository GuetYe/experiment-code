import torch
from utils import *
from ddpg_model import *
from ddpg_agent import *
from env import *

#=======================================#
#                Test mode              #
#=======================================#
class DRL:
    def __init__(self):
        self.k_path = 8
        self.action_dim = 14 * 13
        self.batch_size = 1
        self.hidden_size = 256
        self.gamma = 0.99
        self.lr = 0.0001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.total_test_episodes = 10
        self.env = Env(metric_type="Test")
        self.agent = Agent(self.k_path, self.action_dim, self.batch_size, self.hidden_size, self.gamma, self.lr)
        self.all_path = None
        self.path_obj = PathOperator()
        self.file_obj = FileOperator()



    def get_paths(self, action):
        """

        """
        drl_path = {}
        if self.all_path is None:
            with open("../topos/20211101/topo.json", "r") as json_file:  # get metrics/20210929/metric_1.json
                all_metric_infos = json.load(json_file)
                all_metric_infos = ast.literal_eval(json.dumps(all_metric_infos))
            self.all_path = all_metric_infos
        action_idx = 0
        for src in self.all_path.keys():
            drl_path.setdefault(src, {})
            for dst, paths in self.all_path[src].items():
                drl_path[src][dst] = paths[action[action_idx]]
                # print("src:%s -> dst:%s == drl_path:%s" % (src, dst, paths[action[action_idx]]))
                action_idx += 1

        return drl_path


    def test_model(self):
        """
            Test model
        :return:
        """
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        tar_path = "../TrainOne/modelTar/DQN.pth.tar"
        self.agent.policy_network.load_state_dict(torch.load(tar_path))
        # breakpoint execution
        for t in range(self.total_test_episodes):
            state = self.env.reset()
            episode_reward = 0.
            while(True):
                action = self.agent.action(t, state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            print("state[%s] , reward ：%s " % (t + 1, episode_reward))
            drl_path = self.get_paths(action)
            self.file_obj.write_info_to_json_file(type="PATH", path=self.path_obj.get_save_path(type="PATH"),
                                                  data=drl_path)

            # print("state[%s] , reward ：%s " % (t + 1, reward))
        print("Test done !!! ")

    def change_pth(self):
        """

        :return:
        """
        self.agent.policy_network.load_state_dict(torch.load("modelDict/20211016_reward_v2/DQN/DQN_50.pth"))
        torch.save(self.agent.policy_network.state_dict(), "DQN.pth.tar", _use_new_zipfile_serialization=False)



if __name__ == '__main__':
    drl = DRL()
    drl.test_model()
