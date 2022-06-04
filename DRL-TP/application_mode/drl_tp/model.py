import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layer, bidirectional=False):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.num_direct = 1
        if bidirectional:
            self.num_direct = 2
        self.gru = nn.GRU(input_dim, hidden_size, num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size * self.num_direct, output_dim)

    def forward(self, state, hidden):
        """

        :param state:
        :param hidden:
        :return:
        """
        outs = []
        for i in range(state.shape[0]):
            x = state[i, :, :].unsqueeze(2)
            out, hidden = self.gru(x, hidden)
            if self.num_direct == 2:
                hidden_cat = torch.cat((hidden[-1], hidden[-2]), dim=1)
            else:
                hidden_cat = hidden[-1]
            outs.append(self.linear(hidden_cat))

        return torch.stack(outs, dim=0), hidden

    def init_h_state(self, batch):
        return torch.zeros(self.num_layer * self.num_direct, batch, self.hidden_size, dtype=torch.float)

if __name__ == '__main__':
    x = torch.randn(3, 14 * 14, 7)
    print(x.shape)
    gru = GRU(1, 1, 256, 1, False)
    hidden = gru.init_h_state(196)
    print(hidden.shape)
    out, hidden = gru(x, hidden)
    print(out.shape)




