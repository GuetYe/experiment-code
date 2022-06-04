"""
function: Agent
date    : 2021/10/26
author  : HLQ
"""
import torch
from torch import optim
from ddpg_model import *
from rnn_model import RNNPredict
from bufferReplay import *
import torch.nn.functional as F
from utils import LinearSchedule

#=======================================#
#               Agent                   #
#=======================================#
class Agent:
    def __init__(self, k_path, action_dim, batch_size, hidden_size, gamma, lr, writer):
        print(" ..... LOAD DDPG AGNET ......")
        # ---  DQN --- #
        self.k_path = k_path
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = 0.005
        self.update_iteration = 100
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.writer = writer
        # --- create network --- #
        self.policy_critic = Critic(action_dim).cuda()          # Q_target
        self.policy_actor = Actor(action_dim).cuda()            # Q_policy
        self.target_critic = Critic(action_dim).cuda()          # target_Q,update Q(s',a')
        self.target_actor = Actor(action_dim).cuda()            # target_P，calculate a'
        # --- init target_network --- #
        for target_param, param in zip(self.target_critic.parameters(), self.policy_critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.policy_actor.parameters()):
            target_param.data.copy_(param.data)
        # --- build optimizer --- #
        self.actor_optimizer = optim.Adam(self.policy_actor.parameters(), lr=lr/10)
        self.critic_optimizer = optim.Adam(self.policy_critic.parameters(), lr=lr)
        # --- build memory_buffer object --- #
        self.memory = MemoryBuffer(1000000)
        # --- GRU/LSTM --- #
        self.num_layers = 1
        self.num_directions = False
        self.rnn_lr = lr
        self.init_hidden_flag = True
        self.hidden_state = None
        self.rnn_obj = RNNPredict(1, 1, self.hidden_size, self.num_layers, self.num_directions, rnn_type="gru")
        self.gru = self.rnn_obj.rnn_network.cuda()
        self.rnn_optimizer = optim.Adam(self.gru.parameters(), lr=self.rnn_lr)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0


    def action(self, state):
        """
            Choose action
        :return:
        """
        state = torch.tensor(state, dtype=torch.float).cuda().unsqueeze(0)  # 1, k_paths, 14, 14
        actions = self.policy_actor(state)
        return actions.squeeze(0).cpu().numpy()


    def update(self):
        """
            Update agent.
        :return
        """
        state, action, reward, next_state, done = self.memory.sample(name="RL", batch_size=self.batch_size)

        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.tensor(reward, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        done = torch.tensor(np.float32(done)).cuda()  # batch

        # --- get policy's loss to update policy Q--- #
        policy_loss = self.policy_critic(state, self.policy_actor(state))  # Q(s,a) , (batch, 1)
        policy_loss = - policy_loss.mean()
        self.writer.add_scalar('Loss/actor_loss', policy_loss, global_step=self.num_actor_update_iteration)
        # --- get next action to compute Q(s',a') --- #
        next_action = self.target_actor(next_state)                        # shape: batch, action_dim
        next_value = self.target_critic(next_state, next_action)  # Q(s',a')
        # --- compute target_q_value = reward + r*Q(s',a') --- #
        target_q_value = reward + self.gamma * next_value * (1 - done)
        # --- compute predict_q_value = Q(s,a) --- #
        predict_q_value = self.policy_critic(state, action)
        # --- compute loss and update param  --- #
        mse_loss = F.mse_loss(predict_q_value, target_q_value)
        self.writer.add_scalar('Loss/critic_loss', mse_loss, global_step=self.num_critic_update_iteration)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        mse_loss.backward()
        self.critic_optimizer.step()
        # 因为DDQN是单步更新，所以在此更新网络 #
        # --- update target network --- #
        for target_param, policy_param in zip(self.target_critic.parameters(), self.policy_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + policy_param.data * self.tau)
        for target_param, policy_param in zip(self.target_actor.parameters(), self.policy_actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + policy_param.data * self.tau)

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

    # def update(self):
    #     """
    #         Update network & params
    #     :return:
    #     """
    #
    #     # --- Get experience to buffer --- #
    #     state, action, reward, next_state, done = self.memory.sample(name="RL",batch_size=self.batch_size)
    #     # state = torch.tensor(state, device=self.device, dtype=torch.float)               # 3, 3, 14, 14
    #     # action = torch.LongTensor(action).to(self.device)                                # 3, 182
    #     # reward = torch.tensor(reward, device=self.device, dtype=torch.float)             # 3
    #     # next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)     # 3, 3, 14, 14
    #     # done = torch.tensor(np.float32(done), device=self.device)                        # batch
    #
    #     state = torch.tensor(state, dtype=torch.float).cuda()  # 3, 3, 14, 14
    #     action = torch.LongTensor(action).cuda()  # 3, 182
    #     reward = torch.tensor(reward, dtype=torch.float).cuda()   # 3
    #     next_state = torch.tensor(next_state, dtype=torch.float).cuda()   # 3, 3, 14, 14
    #     done = torch.tensor(np.float32(done)).cuda()   # batch
    #
    #     #--- Get Q value based on current state  --- #
    #     q_value_action = self.policy_network(state)      # list's len batch[3, 182]
    #     q_value_action = torch.stack((q_value_action[0], q_value_action[1],
    #                                   q_value_action[2]), dim=1)
    #     # q_value_action = q_value_action.to(self.device)
    #     q_value_action = q_value_action.cuda()
    #     q_value = q_value_action.gather(1, action.unsqueeze(1))    # batch, 1, 182
    #     q_value = q_value.squeeze(1)                               # batch, 182
    #
    #     # --- Get Target Q Value based on next state --- #
    #     p_value_action = self.policy_network(next_state)  #  list's len batch[3, 182]
    #     p_value_action = torch.stack((p_value_action[0], p_value_action[1],
    #                                   p_value_action[2]), dim=1).detach()
    #     # get next actions
    #     # p_value_action = p_value_action.to(self.device)
    #     p_value_action = p_value_action.cuda()
    #     _, next_action = p_value_action.max(1)                 # get max action index, batch, 182
    #
    #     t_value_action = self.target_network(next_state)
    #     t_value_action = torch.stack((t_value_action[0], t_value_action[1],
    #                                   t_value_action[2]), dim=1)
    #     # t_value_action = t_value_action.to(self.device)
    #     t_value_action = t_value_action.cuda()
    #     t_value = t_value_action.gather(1, next_action.unsqueeze(1)).squeeze(1)  # get value  --> 3, 182
    #     # --- Get target value --- #
    #     # for i in range(t_next_state_value.shape[0]):
    #     #     t_next_state_value[i, :] = reward[i] + self.gamma * (1 - done[i]) * t_next_state_value[i, :]
    #     for i in range(t_value.shape[0]):
    #         t_value[i, :] = reward[i] + self.gamma * (1 - done[i]) * t_value[i, :]
    #     # --- Update network --- #
    #     loss = self.criterion(q_value, t_value)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def get_predict_state(self, seq_previous_traffic):
        """
            Update rnn network
        :param seq_previous_traffic: seq, 3, 196
        :return:
        """
        with torch.no_grad():
            # input_state = torch.tensor(seq_previous_traffic, device=self.device, dtype=torch.float).permute(1, 2, 0)  # 3, 196, seq
            input_state = torch.tensor(seq_previous_traffic,  dtype=torch.float).permute(1, 2, 0).cuda()  # 3, 196, seq
            if self.init_hidden_flag:
                # self.hidden_state = self.gru.init_h_state(input_state.shape[1]).to(self.device)
                self.hidden_state = self.gru.init_h_state(input_state.shape[1]).cuda()
                # num_layers, batch, hidden
                self.init_hidden_flag = False
            # out shape: 3, 196, 1 | out_hidden shape: 1, 196, 256
            out, hidden = self.gru(input_state, self.hidden_state)
            # 3, 196
            out = out.permute(2, 0, 1).squeeze(0).cpu().numpy()
            self.hidden_state = hidden.detach()

            return out

if __name__ == '__main__':
    k_path = 8
    action_dim = 182
    batch_size = 64
    hidden_size = 256
    gamma = 0.9
    lr = 0.0001
    agent = Agent(k_path, action_dim, batch_size, hidden_size, gamma, lr, "")
    agent.update()


