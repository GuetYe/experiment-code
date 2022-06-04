"""
function:RNN Network
date    :2021/10/16
author  :HLQ
"""

import torch
import torch.nn as nn

# =============================   GRU/LSTM ================================= #
#=======================================#
#                    GRU                #
#=======================================#
class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, num_directions=False):
        super(GRU, self).__init__()
        print(" ..... LOAD DRLRP GRU ......")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direct = 1
        if num_directions:
            self.num_direct = 2
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * self.num_direct, output_dim)

    def forward(self, state, hidden):
        """
            state's shape : 3, 196, 7
            hidden's sahape: 1, 196, 256
        """

        outs = []
        for i in range(state.shape[0]):
            x = state[i, :, : ].unsqueeze(2)
            # out shape   ： batch, seq_len, num_directions * hidden_size
            # hidden shape： num_layers * num_directions, batch, hidden_size
            out, hidden = self.gru(x, hidden)
            if self.num_direct == 2:
                hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
            else:
                hidden_cat = hidden[-1]
            outs.append(self.linear(hidden_cat))  # batch, 1

        return torch.stack(outs, dim=0), hidden

    def init_h_state(self, batch_size):
        """
            init hidden size, 14 * 14
        :param batch_size:
        :return:
        """
        return torch.zeros(self.num_layers * self.num_direct, batch_size, self.hidden_size, dtype=torch.float)

#=======================================#
#                   LSTM                #
#=======================================#
class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, num_directions=False):
        super(LSTM, self).__init__()
        print(" ..... LOAD DRLRP LSTM ......")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direct = 1
        if num_directions:
            self.num_direct = 2
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size * self.num_direct, output_dim)

    def forward(self, x, hidden):
        # out shape   ：batch, seq, num_directions * hidden_size
        # hidden shape：num_layers * num_directions, batch, hidden_size
        out, hidden = self.lstm(x, hidden)
        outs = []
        # Ensure that each batch performs a linear layer, out.size(0) -> batch
        for record in range(out.size(0)):
            outs.append(self.linear(out[record,: :]))
        return torch.stack(outs, dim=0), hidden

    def init_h_c_state(self, batch_size):
        """

        :return:
        """
        return (torch.zeros(self.num_layers * self.num_direct, batch_size, self.hidden_size),
                torch.zeros(self.num_layers * self.num_direct, batch_size, self.hidden_size))


class RNNPredict:
    def __init__(self, input_dim, output_dim, hidden_size, num_layers, num_directions, rnn_type="gru"):
        self.seq = 7
        # self.total_state_space = total_state_space
        self.current_idx = None
        if rnn_type == "gru":
            self.rnn_network = GRU(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers, num_directions=num_directions)
        else:
            self.rnn_network = LSTM(input_dim=input_dim, output_dim=output_dim, hidden_size=hidden_size, num_layers=num_layers, num_directions=num_directions)

    def can_predict_traffic(self, env_obj):
        """
            Judge whether the current meets the forecast conditions
        :param env_obj:
        :return:
        """
        # 满足seq
        if (env_obj.rnn_idx + 1) > self.seq:
            if env_obj.get_state_idx() >= (self.seq - 1) and env_obj.get_state_idx() < (env_obj.get_total_state_space() - 1):
                self.current_idx = env_obj.get_state_idx()
                return True
        return False

    def generate_predict_traffic(self, env_obj):
        """
            Generate time series traffic matrix
        :param env_obj:
        :return:
        """
        if self.can_predict_traffic(env_obj):
            # shape: seq, 3, 14, 14  --> seq, 3, 196
            seq_previous_traffic = env_obj.state[self.current_idx - self.seq + 1: self.current_idx + 1, :, :, :].reshape(self.seq, env_obj.k_paths, -1)
            return seq_previous_traffic
        return None


if __name__ == '__main__':
    x = torch.randn((196, 7, 3))
    rnn_predict = RNNPredict(1, 1, 256, 1, False, rnn_type="gru")
    gru = rnn_predict.rnn_network
    hidden = gru.init_h_state(196)
    out, hidden = gru(x, hidden)
    print(out.shape)

