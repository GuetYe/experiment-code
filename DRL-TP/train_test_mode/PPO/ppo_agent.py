"""
function: Agent
date    : 2021/11/02
author  : HLQ
"""
import torch
from torch import optim
# from DRLPR.dqn_model import *
from ppo_model import *
from rnn_model import RNNPredict
# from TrainTwo.bufferReplay import *
from bufferReplay import RolloutBuffer
import numpy as np
import torch.nn.functional as F
from utils import LinearSchedule

#=======================================#
#                Agent                  #
#=======================================#
class Agent:
    def __init__(self, action_dim, lr_actor, lr_critic, lr_rnn, gamma, K_epochs, eps_clip, device):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.device = device

        self.policy_net = ActorCritic(action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy_net.critic.parameters(), 'lr': lr_critic}
        ])

        self.target_net = ActorCritic(action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.MseLoss = nn.MSELoss()
        self.hidden_size = 256
        self.num_layers = 1
        self.num_directions = False
        self.rnn_obj = RNNPredict(1, 1, self.hidden_size, self.num_layers, self.num_directions, rnn_type="gru")
        self.gru = self.rnn_obj.rnn_network.cuda()
        self.rnn_optimizer = optim.Adam(self.gru.parameters(), lr=lr_rnn)

        self.init_hidden_flag = False
        self.hidden_state = None


    def action(self, state):
        """
            Choose action
        :return:
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, action_logprob = self.target_net.act(state)

        state = state.squeeze(0).cpu()                     # k_paths, nodes, nodes
        action = action.squeeze(0).cpu()                   # 182
        action_logprob = action_logprob.squeeze(0).cpu()   # 182
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        return action

    def update(self):
        """
        :return:
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.stack([state for state in self.buffer.states], dim=0).to(self.device)               # batch, k_paths, nodes, nodes
        old_actions = torch.stack([action for action in self.buffer.actions], dim=0).to(self.device)           # batch, action_dim
        old_logprobs = torch.stack([logprob for logprob in self.buffer.logprobs], dim=0).to(self.device)       # batch, action_dim

        pi_theta = old_logprobs.sum(1)                                                                         # batch
        # Optimize policy_net for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_net.evaluate(old_states, old_actions)
            # print("logprobs | state_values | dist_entropy", logprobs.shape, state_values.shape, dist_entropy.shape)
            pi_theta__old = logprobs.sum(1)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)    # batch

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(pi_theta - pi_theta__old.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            # print("advantages", advantages.shape)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy.sum(1)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy_net
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # clear buffer
        self.buffer.clear()

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
            # out shape: k, 196, 1 | out_hidden shape: 1, 196, 256
            out, hidden = self.gru(input_state, self.hidden_state)
            out = out.permute(2, 0, 1).squeeze(0).cpu().numpy()
            self.hidden_state = hidden.detach()

            return out




