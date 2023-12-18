import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


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
        # 'text': text,
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
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

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        hidden_state = output.pooler_output
        pooler = hidden_state.squeeze()
        linear_out = self.linear(pooler)
        linear_out = self.relu(linear_out)
        linear_out = self.linear1(linear_out)
        res = self.sigmoid(linear_out) * 6
        return SequenceClassifierOutput(loss=None, logits=res, hidden_states=output.hidden_states, attentions=output.attentions)
        # return res
