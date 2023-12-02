import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from text_complexity.model.trainable.gru import (EfcamdatDataset, GRUNet,
                                                 load_vectors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
HIDDEN_DIM = 256
folder_path = "/cluster/work/sachan/abhinav/model/gru/"
model_folder_name = 'efcamdat_run1'
model_folder = os.path.join(folder_path, model_folder_name)
log_path = model_folder + "/result_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"

logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
logging.info("device: " + str(device))
logging.info("Inferenece Result on Original Training")

# srun -n 1 -A es_sachan --cpus-per-task=4 --mem-per-cpu=12G --gpus=1 --gres=gpumem:24G --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --pty python3 text_complexity/model/gru_inference.py


def collate_fn(batch):
    label_list, word_list = [], []
    for _word, _label, _text in batch:
        label_list.append(_label)
        word_list.append(_word)
    word_padded = torch.nn.utils.rnn.pad_sequence(word_list, batch_first=True)
    return word_padded, torch.tensor(label_list, device=device)


def evaluate(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for i, data in enumerate(tqdm(test_loader)):
        inputs, labels = data
        out = model(inputs.to(device).float())
        outputs.extend(out[0].squeeze().detach().cpu().numpy().tolist())
        targets.extend(labels.tolist())
    print("Evaluation Time: {}".format(str(time.perf_counter()-start_time)))
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
    # find pearson correlation coefficient and p-value of target and output
    pearson_corr, p_value = pearsonr(outputs, targets)

    print("Pearson Correlation Coefficient: {}".format(pearson_corr))
    print("P-value: {}".format(p_value))
    logging.info("Pearson Correlation Coefficient: {}".format(pearson_corr))
    logging.info("P-value: {}".format(p_value))


if __name__ == '__main__':
    test_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/test_pruned_efcamdat.csv')
    print('test_df shape', test_df.shape)
    print('testing data loaded')
    fasttext_model = '/cluster/work/sachan/abhinav/text_complexity/embedding/wiki-news-300d-1M.vec'
    fasttext = load_vectors(fasttext_model)

    test_dataset = EfcamdatDataset(test_df, fasttext)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_fn)

    input_dim = next(iter(test_dataloader))[0].shape[2]
    output_dim = 200
    n_layers = 2
    # Instantiating the models
    model = GRUNet(input_dim, HIDDEN_DIM, output_dim, n_layers)
    checkpoint = torch.load(
        '/cluster/work/sachan/abhinav/model/gru/efcamdat_run1/gru_20231129_161208.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    evaluate(model, test_dataloader)
