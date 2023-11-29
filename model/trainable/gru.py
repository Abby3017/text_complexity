import torch
import torch.nn as nn


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=True, dropout=drop_prob, bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, output_size)
        self.linear1 = nn.Linear(output_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out, hidden = self.gru(input)
        linear_out = self.linear(out[:, -1, :])
        linear_out = self.relu(linear_out)
        linear_out = self.linear1(linear_out)
        res = self.sigmoid(linear_out) * 6
        return res, hidden

    def init_hidden(self, batch_size, device, num_directions=2):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers * num_directions, batch_size,
                            self.hidden_size).zero_().to(device)
        return hidden
