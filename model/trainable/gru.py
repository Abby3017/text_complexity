import io

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


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


class EfcamdatDataset(Dataset):
    def __init__(self, data, emb):
        self.data = data
        self.emb = emb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentences']
        words = word_tokenize(text)
        word_ids = []
        for word in words:
            if word in self.emb:
                word_ids.append(torch.tensor(list(self.emb[word])))
            else:
                word_ids.append(torch.tensor(list(self.emb["UNK"])))
        words = torch.stack(word_ids)
        target = torch.tensor(self.data.iloc[idx]['cefr_numeric'])
        return words, target, text

# https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(x) for x in tokens[1:]]
    return data
