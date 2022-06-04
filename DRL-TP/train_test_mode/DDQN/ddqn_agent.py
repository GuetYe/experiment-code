"""
function: Agent
date    : 2021/9/25
author  : HLQ
"""
from torch import optim
from ddqn_model import *
from rnn_model import RNNPredict
from bufferReplay import *
from utils import LinearSchedule

#=======================================#
#               Agent                   #
#=======================================#
class Agent:
    def __init__(self, k_path, action_dim, batch_size, hidden_size, gamma, lr):
        print(" ..... LOAD DRLRP AGNET ......")
        # ---  DQN --- #
        self.k_path = k_path
        self.action_dim = action_dim
        self.policy_network = Dueling_DQN(self.action_dim).cuda()
        self.target_network = Dueling_DQN(self.action_dim).cuda()
        self.memory = MemoryBuffer(100000)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss().cuda()
        # self.device = device
        self.learning_starts = 5000
        self.exploration = LinearSchedule(200000, 0.1)

        # --- GRU/LSTM --- #
        self.num_layers = 1
        self.num_directions = False
        self.rnn_lr = lr
        self.init_hidden_flag = True
        self.hidden_state = None
        self.rnn_obj = RNNPredict(1, 1, self.hidden_size, self.num_layers, self.num_directions, rnn_type="gru")
        self.gru = self.rnn_obj.rnn_network.cuda()
        self.rnn_optimizer = optim.Adam(self.gru.parameters(), lr=self.rnn_lr)
        self.rnn_criterion = nn.MSELoss().cuda()


    def action(self, t, state):
        """
        Choose action
        :return:
        """

        epsion = random.random()  # (0, 1)
        # threshold = self.exploration.value(t)  # decay value
        state = torch.tensor(state, dtype=torch.float).cuda().unsqueeze(0)  # 1, 3, 14, 14
        if epsion > self.exploration.value(t):
            with torch.no_grad():
                actions = self.policy_network(state)
                actions = torch.stack([action for action in actions], dim=1).detach()  # batch, 3, 182
            action = actions.squeeze(0).max(0)[1].cpu().numpy()                       # get action 1, 14 * 13
        else:
            action = np.array([np.random.randint(state.shape[1]) for _ in range(self.action_dim)])
        return action

    def update(self):
        """
            Update network & params
        :return:
        """

        # --- Get experience to buffer --- #
        state, action, reward, next_state, done = self.memory.sample(name="RL",batch_size=self.batch_size)
        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.tensor(reward, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        done = torch.tensor(np.float32(done)).cuda()

        #--- Get Q value based on current state  --- #
        q_value_action = self.policy_network(state)
        q_value_action = torch.stack([action for action in q_value_action], dim=1)
        q_value_action = q_value_action.cuda()
        q_value = q_value_action.gather(1, action.unsqueeze(1))
        q_value = q_value.squeeze(1)                               #

        # --- Get Target Q Value based on next state --- #
        p_value_action = self.policy_network(next_state)
        p_value_action = torch.stack([action for action in p_value_action], dim=1).detach()
        # get next actions
        p_value_action = p_value_action.cuda()
        _, next_action = p_value_action.max(1)                 # get max action index, batch, 182

        t_value_action = self.target_network(next_state)
        t_value_action = torch.stack([action for action in t_value_action], dim=1)
        # t_value_action = t_value_action.to(self.device)
        t_value_action = t_value_action.cuda()
        t_value = t_value_action.gather(1, next_action.unsqueeze(1)).squeeze(1)  # get value
        # --- Get target value --- #
        for i in range(t_value.shape[0]):
            t_value[i, :] = reward[i] + self.gamma * (1 - done[i]) * t_value[i, :]
        # --- Update network --- #
        loss = self.criterion(q_value, t_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_predict_state(self, seq_previous_traffic):
        """
            Update rnn network
        :param seq_previous_traffic: seq, k, 196
        :return:
        """
        with torch.no_grad():
            input_state = torch.tensor(seq_previous_traffic,  dtype=torch.float).permute(1, 2, 0).cuda()
            if self.init_hidden_flag:
                # self.hidden_state = self.gru.init_h_state(input_state.shape[1]).to(self.device)
                self.hidden_state = self.gru.init_h_state(input_state.shape[1]).cuda()
                # num_layers, batch, hidden
                self.init_hidden_flag = False
            # out shape: k, 196, 1 | out_hidden shape: 1, 196, 256
            out, hidden = self.gru(input_state, self.hidden_state)
            # k, 196
            out = out.permute(2, 0, 1).squeeze(0).cpu().numpy()
            self.hidden_state = hidden.detach()

            return out




