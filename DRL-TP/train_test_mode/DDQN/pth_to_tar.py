"""
date:2021/10/21
author:hlq
function:convert pth to tar
"""

import torch
from ddqn_agent import Agent

class HandleClass:
    def __init__(self):
        self.k_path = 8
        self.action_dim = 14 * 13
        self.batch_size = 1
        self.hidden_size = 256
        self.gamma = 0.99
        self.lr = 0.0001
        self.agent = Agent(self.k_path, self.action_dim, self.batch_size, self.hidden_size, self.gamma,
                                        self.lr)
        self.dqn_pth_path = "../modelDict/20211021/DQN/DQN_50.pth"
        self.gru_pth_path = "../modelDict/GRU.pth"
        self.dqn_tar_path = "../modelTar"
        self.gru_tar_path = "../modelTar"

    def change_pth_to_tar(self):
        """

        :return
        """
        # dqn
        self.agent.policy_network.load_state_dict(torch.load(self.dqn_pth_path))
        torch.save(self.agent.policy_network.state_dict(), self.dqn_tar_path + "/DQN.pth.tar", _use_new_zipfile_serialization=False)
        # gru
        self.agent.gru.load_state_dict(torch.load(self.gru_pth_path))
        torch.save(self.agent.gru.state_dict(), self.gru_tar_path + "/GRU.pth.tar", _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    handle_class = HandleClass()
    handle_class.change_pth_to_tar()
