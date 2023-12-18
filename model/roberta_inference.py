import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from text_complexity.model.pretrained.roberta import (RobertaEfcamdatDataset,
                                                      RobertaNet)

# srun -n 1 -A es_sachan --cpus-per-task=4 --mem-per-cpu=12G --gpus=1 --gres=gpumem:24G --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --pty python3 text_complexity/model/roberta_inference.py

MAX_LEN = 256
BATCH_SIZE = 16
MODEL_NAME = 'roberta-base'

torch.cuda.init()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = "/cluster/work/sachan/abhinav/model/roberta/"
model_folder_name = 'efcamdat_run2'
model_folder = os.path.join(folder_path, model_folder_name)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

log_path = model_folder + "/result_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"

logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
logging.info("device: " + str(device))
print("device: " + str(device))
print("Model output taken pooler_output")
logging.info("Model output taken pooler_output")


def evaluate(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(
                device, dtype=torch.long)
            output = model(ids, mask, token_type_ids)
            outputs.extend(output.squeeze().detach().cpu().numpy().tolist())
            targets.extend(batch['labels'].cpu().numpy().tolist())
    outputs = np.array(outputs)
    targets = np.array(targets)

    errors = abs(outputs-targets)
    print("Mean Absolute Error: {}".format(np.mean(errors)))
    logging.info("Mean Absolute Error: {}".format(np.mean(errors)))
    print("Median Absolute Error: {}".format(np.median(errors)))
    print("Standard Deviation of Absolute Error: {}".format(np.std(errors)))

    mse = np.mean((outputs-targets)**2)
    rmse = np.sqrt(mse)
    print("RMSE: {}".format(rmse))
    logging.info("RMSE: {}".format(rmse))
    print("MSE: {}".format(mse))
    logging.info("MSE: {}".format(mse))

    pearson_corr, p_value = pearsonr(outputs, targets)

    print("Pearson Correlation Coefficient: {}".format(pearson_corr))
    print("P-value: {}".format(p_value))
    logging.info("Pearson Correlation Coefficient: {}".format(pearson_corr))
    logging.info("P-value: {}".format(p_value))


if __name__ == '__main__':
    test_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/test_pruned_efcamdat.csv')
    print('test_df shape', test_df.shape)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir="/cluster/work/sachan/abhinav/model/roberta/cache", resume_download=True)
    print('tokenizer loaded')
    test_dataset = RobertaEfcamdatDataset(
        test_df, tokenizer, MAX_LEN)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = RobertaNet(MODEL_NAME)
    checkpoint = torch.load(
        '/cluster/work/sachan/abhinav/model/roberta/efcamdat_run2/gru_20231129_162428.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("checkpoint model best val loss {}".format(
        checkpoint["loss"]))
    logging.info("checkpoint model best epoch {}".format(
        checkpoint["epoch"]))
    model.to(device)
    evaluate(model, test_loader)
