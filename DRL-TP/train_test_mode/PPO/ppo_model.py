import torch
import torch.nn as nn
from torch.distributions import Categorical

#=======================================#
#               Critic                  #
#=======================================#
class Critic(nn.Module):
    def __init__(self, action_dim, init_w=3e-3):
        super(Critic, self).__init__()
        self.in_channel = 1
        self.action_dim = action_dim
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # ResidualBlock(32, 32),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, state):
        """
            state : batch, k_path, 14, 14
            action: batch, action_dim
        :return
        """
        values = []
        # paths num
        for i in range(state.shape[1]):
            x = state[:, i, :, :].unsqueeze(1)
            batch_size = x.shape[0]
            x = self.conv_layer(x)      # batch, 64, 4, 4
            x = x.view(batch_size, -1)  # batch, 64* 4* 4
            x = self.linear_layer(x)
            values.append(x)


        values = torch.stack([value for value in values], dim=1)     # batch, 1
        values = values.max(1)[0]

        return values

#=======================================#
#               Actor                   #
#=======================================#
class Actor(nn.Module):
    def __init__(self, action_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.in_channel = 1
        self.action_dim = action_dim
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # ResidualBlock(32, 32),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )


    def forward(self, state):
        """

        """
        actions = []
        for i in range(state.shape[1]):
            x = state[:, i, :, :].unsqueeze(1)
            batch_size = x.shape[0]
            x = self.conv_layer(x)
            x = x.view(batch_size, -1)
            x = self.linear_layer(x)   # batch, action_dim
            # print(x.shape)
            actions.append(x)
        # batch, k_path, action_dim  -> batch, action_dim, k_path
        actions = torch.stack([action for action in actions], dim=1).permute(0, 2, 1)
        # print(actions.shape)
        # print(actions.shape)
        # return actions.max(1)[1]           # action idx
        return actions

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = Actor(action_dim)
        self.critic = Critic(action_dim)

    def act(self, state):
        """

        """
        action_probs = self.actor(state)          # batch, k_path, action_dim
        dist = Categorical(action_probs)
        action = dist.sample()                    # batch, action_dim
        # print(action.shape)
        # print(type(action))
        action_logprob = dist.log_prob(action)    # batch, action_dim

        return action.detach(), action_logprob

    def evaluate(self, state, action):
        """

        """
        action_probs = self.actor(state)          # batch, action_dim, k_paths
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)   # bacth, action_dim
        dist_entropy = dist.entropy()
        state_values = self.critic(state)         # batch, 1

        return action_logprobs, state_values, dist_entropy


# class PPO:
#     def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,  action_std_init=0.6):
#
#         # self.has_continuous_action_space = has_continuous_action_space
#         #
#         # if has_continuous_action_space:
#         #     self.action_std = action_std_init
#         #
#         # self.gamma = gamma
#         # self.eps_clip = eps_clip
#         # self.K_epochs = K_epochs
#         #
#         # self.buffer = RolloutBuffer()
#
#         self.policy = ActorCritic(action_dim,  action_std_init)
#         self.optimizer = torch.optim.Adam([
#             {'params': self.policy.actor.parameters(), 'lr': lr_actor},
#             {'params': self.policy.critic.parameters(), 'lr': lr_critic}
#         ])
#
#         self.policy_old = ActorCritic(action_dim, action_std_init)
#         self.policy_old.load_state_dict(self.policy.state_dict())
#
#         self.MseLoss = nn.MSELoss()
#
#     # def set_action_std(self, new_action_std):
#     #
#     #     if self.has_continuous_action_space:
#     #         self.action_std = new_action_std
#     #         self.policy.set_action_std(new_action_std)
#     #         self.policy_old.set_action_std(new_action_std)
#     #
#     #     else:
#     #         print("--------------------------------------------------------------------------------------------")
#     #         print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
#     #         print("--------------------------------------------------------------------------------------------")
#     #
#     # def decay_action_std(self, action_std_decay_rate, min_action_std):
#     #     print("--------------------------------------------------------------------------------------------")
#     #
#     #     if self.has_continuous_action_space:
#     #         self.action_std = self.action_std - action_std_decay_rate
#     #         self.action_std = round(self.action_std, 4)
#     #         if (self.action_std <= min_action_std):
#     #             self.action_std = min_action_std
#     #             print("setting actor output action_std to min_action_std : ", self.action_std)
#     #         else:
#     #             print("setting actor output action_std to : ", self.action_std)
#     #         self.set_action_std(self.action_std)
#     #
#     #     else:
#     #         print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
#     #
#     #     print("--------------------------------------------------------------------------------------------")
#
#     def select_action(self, state):
#
#
#         with torch.no_grad():
#             state = torch.FloatTensor(state)
#             action, action_logprob = self.policy_old.act(state)
#
#         # self.buffer.states.append(state)
#         # self.buffer.actions.append(action)
#         # self.buffer.logprobs.append(action_logprob)
#
#         return action.item()
#
#     def update(self):
#
#         # Monte Carlo estimate of returns
#         rewards = []
#         discounted_reward = 0
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
#
#         # Normalizing the rewards
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#
#         # convert list to tensor
#         old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
#         old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
#
#         # Optimize policy for K epochs
#         for _ in range(self.K_epochs):
#             # Evaluating old actions and values
#             logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
#
#             # match state_values tensor dimensions with rewards tensor
#             state_values = torch.squeeze(state_values)
#
#             # Finding the ratio (pi_theta / pi_theta__old)
#             ratios = torch.exp(logprobs - old_logprobs.detach())
#
#             # Finding Surrogate Loss
#             advantages = rewards - state_values.detach()
#             surr1 = ratios * advantages
#             surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
#
#             # final loss of clipped objective PPO
#             loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
#
#             # take gradient step
#             self.optimizer.zero_grad()
#             loss.mean().backward()
#             self.optimizer.step()
#
#         # Copy new weights into old policy
#         self.policy_old.load_state_dict(self.policy.state_dict())
#
#         # clear buffer
#         self.buffer.clear()
#
#     def save(self, checkpoint_path):
#         torch.save(self.policy_old.state_dict(), checkpoint_path)
#
#     def load(self, checkpoint_path):
#         self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
#         self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


if __name__ == '__main__':
    state = torch.randn(16, 8, 14, 14)
    import random
    action = [[random.randint(0, 8) for _ in range(182)] for _ in range(16)]
    action = torch.Tensor(action).cuda()
    # critic = Critic(182)
    # actions = critic(state, action)
    # print(actions.shape)
    actor_critic = ActorCritic(182)
    actor_critic.act(state)
    actor_critic.evaluate(state, action)
    # action = None
    # for i in range(len(actions)- 1):
    #     if action is None:
    #         action = actions[i]
    #     action = torch.stack([action for action in actions], dim=1).detach()
    # print(action.shape)
