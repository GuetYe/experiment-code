import torch
import torch.nn as nn

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
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4 + self.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, state, action):
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
            x_action = torch.cat([x, action], dim=1)
            x = self.linear_layer(x_action)
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
            nn.Tanh()
        )


    def forward(self, state):
        """

        :param state:
        :return:
        """
        actions = []
        for i in range(state.shape[1]):
            x = state[:, i, :, :].unsqueeze(1)
            batch_size = x.shape[0]
            x = self.conv_layer(x)
            x = x.view(batch_size, -1)
            x = self.linear_layer(x)        # batch, action_dim
            actions.append(x)
        actions = torch.stack([action for action in actions], dim=1)
        return actions.max(1)[1]            # action idx


