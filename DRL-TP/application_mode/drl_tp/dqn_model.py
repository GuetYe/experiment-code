# _*_coding:utf-8 _*_
"""
function:DQN Network
date    :2021/10/16
author  :HLQ
"""
import torch
import torch.nn as nn
# ============================= DQN Network ================================ #

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_chanel):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, out_chanel, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(),
            nn.Conv2d(out_chanel, out_chanel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(),
            nn.Conv2d(out_chanel, out_chanel, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.residual(x)

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        self.conv_layer = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.Dropout(),
            nn.Linear(512, self.action_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layer(x)
        x = x.view(batch_size, -1)
        print(x.shape)
        x = self.linear_layer(x)
        return x


class Dueling_DQN(nn.Module):
    """
        Dueling DQN model
    """
    def __init__(self, action_dim):
        super(Dueling_DQN, self).__init__()
        # print(" ..... LOAD DRLRP DUELINGDQN ......")
        self.in_channel = 1
        self.action_dim = action_dim
        self.conv_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 8, kernel_size=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # ResidualBlock(32, 32),
            nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.linear_adv = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim)
        )

        self.linear_val = nn.Sequential(
            nn.Linear( 64 * 4 * 4,  512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(128, 1)
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
            adv = self.linear_adv(x)
            val = self.linear_val(x).expand(batch_size, self.action_dim)  # expand to --> (batch, action_dim)
            # adv.mean(1) --> shape (batch, ) --> [unsqueeze(1)] --> (batch, 1) --> [expand] ---> (batch, action_dim)
            x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_dim)
            actions.append(x)

        return actions


if __name__ == '__main__':
    x = torch.randn(32, 3, 14, 14)
    dqn = Dueling_DQN(14)
    # print(dqn(x).shape)
    actions = torch.zeros((32, 3, 14))
    for i in range(x.shape[1]):
        data = x[:, i, :, :].unsqueeze(1)
        data = dqn(data)
        actions[:, i, :] = data

    # print(actions.shape)
    # print(actions)
    data, idx = actions.data.max(1)
    print(idx.shape)
    print(data)

