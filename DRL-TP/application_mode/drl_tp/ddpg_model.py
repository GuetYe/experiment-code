import torch
import torch.nn as nn
class Actor(nn.Module):
    def __init__(self, action_dim, init_w=3e-3):
        # print("DDPG model")
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
            nn.Tanh()
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
            actions.append(x)

        return  actions
        # print(actions[0].shape)
        # actions = torch.stack([action for action in actions], dim=1)
        # # print(actions.shape)
        # return actions.max(1)[1]           # action idx