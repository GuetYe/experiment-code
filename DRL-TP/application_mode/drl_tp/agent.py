"""
function: Agent
date    : 2021/9/25
author  : HLQ
"""

from torch import optim
from dqn_model import *
# from ddpg_model import *

#=======================================#
#               Agent                   #
#=======================================#
class Agent:
    def __init__(self, k_path, action_dim, batch_size, hidden_size, gamma, lr,  device, writer):
        # print(" ..... LOAD DRLRP AGNET ......")
        # ---  DQN --- #
        self.k_path = k_path
        self.action_dim = action_dim
        self.policy_network = Dueling_DQN(self.action_dim).to(device)
        # self.target_network = Dueling_DQN(self.action_dim).to(device)
        # self.policy_network = Actor(self.action_dim).to(device)
        # self.memory = MemoryBuffer(100000)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.device = device
        # --- GRU/LSTM --- #
        self.num_layers = 1
        self.num_directions = False
        self.rnn_lr = lr
        self.init_hidden_flag = True
        self.hidden_state = None
        self.writer = writer


    def action(self, state):
        """
            Choose action
        :return:
        """
        state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)  # 1, 3, 14, 14
        with torch.no_grad():
            actions = self.policy_network(state)
            # print(len(actions))
            actions = torch.stack([action for action in actions], dim=1).detach()  # batch, 8, 182
        action = actions.squeeze(0).max(0)[1].cpu().numpy()                              # get action 1, 14 * 13
        return action


