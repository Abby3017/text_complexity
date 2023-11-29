import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


class RobertaEfcamdatDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['sentences']
        target = self.data.iloc[idx]['cefr_numeric']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        return {
            'text': text,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(target, dtype=torch.long)
        }


class RobertaNet(torch.nn.Module):
    def __init__(self, model_name, hidden_size=768, output_size=200):
        super(RobertaNet, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir="/cluster/work/sachan/abhinav/model/roberta/cache", resume_download=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.linear1 = nn.Linear(output_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids):
        output = self.model(input_ids=ids, attention_mask=mask,
                            token_type_ids=token_type_ids)
        hidden_state = output[1]
        # output[0] ==  [128,256, 768] -- last hidden state
        # output[0].squeeze() == [256, 768]
        # output[0][:, 0] == [1, 768]
        # output[1] ==  [1,768] -- pooler output
        # output[2] == all hidden states [1,256, 768]
        # pooler = hidden_state[:, -1, :]  # read last hidden state
        # for pooler, we can use hidden_state.squeeze()
        pooler = hidden_state.squeeze()
        linear_out = self.linear(pooler)
        linear_out = self.relu(linear_out)
        linear_out = self.linear1(linear_out)
        res = self.sigmoid(linear_out) * 6
        return res
