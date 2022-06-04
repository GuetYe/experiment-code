from env import Env
from agent import *
from utils_file import MakePath


class DRL:
    """
        DRL(Konwledge plane)
            As an important part of the knowledge plane, this class mainly selects the action of intelligent routing algorithm,
            that is, the best routing forwarding path.
    """
    def __init__(self):
        print("creating context network_agent")
        print("instanting app of None of NetworkAgent")
        self.batch_size = 1
        self.hidden_size = 256
        self.gamma = 0.99
        self.lr = 0.0001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.metric_factors = ["free_bw", "delay", "packet_loss", "used_bw", "packet_drop", "packet_errors"]
        self.env = Env()
        self.agent = Agent(self.env.k_paths, self.env.action_dim, self.batch_size, self.hidden_size, self.gamma, self.lr,
                           self.device, None)
        self.all_path = None
        self.path_obj = MakePath()
        # load dqn model
        dqn_path = self.path_obj.load_dqn_model_pth()  # idx and pths
        self.agent.policy_network.load_state_dict(torch.load(dqn_path, map_location="cpu"))

    def get_paths(self, action):
        """
            Get the optimal forwarding path of all switches.
        :param action:
        :return:
        """
        drl_path = {}
        if self.all_path is None:
            self.all_path = self.path_obj.get_topo_data()
        action_idx = 0
        for src in self.all_path.keys():
            drl_path.setdefault(src, {})
            for dst, paths in self.all_path[src].items():
                drl_path[src][dst] = paths[action[action_idx]]
                action_idx += 1
        return drl_path
    

    def get_optimal_forwarding_path(self, all_metric_infos):
        """

        :param all_metric_infos: candidate, node, node
        :return:
        """
        state = self.env.generate_traffic(all_metric_infos)   # get state
        action = self.agent.action(state)                     # get action
        drl_path = self.get_paths(action)                     # get forwarding paths.

        return drl_path

    def util_test_model(self):
        state = torch.randn(8, 14, 14)
        action = self.agent.action(state)
        print action


if __name__ == '__main__':
    drl = DRL()
    # drl.test_model(None)
    drl.util_test_model()

