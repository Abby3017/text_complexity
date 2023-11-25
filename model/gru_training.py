import io
import logging
import os
import pdb
import pickle
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from text_complexity.model.trainable.gru import GRUNet

# srun -n 1 --cpus-per-task=4 --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --pty python3 text_complexity/model/gru_training.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "/cluster/work/sachan/abhinav/model/gru/"
model_folder_name = 'efcamdat_run1'
model_folder = os.path.join(folder_path, model_folder_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = model_folder + "/gru_" + \
    datetime.now.strftime("%Y%m%d_%H%M%S") + ".pt"

log_path = model_folder + "/run_" + \
    datetime.now.strftime("%Y%m%d_%H%M%S") + ".log"
tb_model_path = model_folder + "/tensorboard_" + \
    datetime.now.strftime("%Y%m%d_%H%M%S") + "/"

logging.basicConfig(filename=log_path, level=logging.INFO)
writer = SummaryWriter(log_dir=tb_model_path)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(x) for x in tokens[1:]]
    return data


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


def collate_fn(batch):
    label_list, word_list = [], []
    for _word, _label, _text in batch:
        label_list.append(_label)
        word_list.append(_word)
    word_padded = torch.nn.utils.rnn.pad_sequence(word_list, batch_first=True)
    return word_padded, torch.tensor(label_list, device=device)


def save_plots(train_loss, val_loss):
    plt.plot(train_loss, label='Training loss', color='green')
    plt.plot(val_loss, label='Validation loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(frameon=False)
    plt.savefig(model_folder + '/loss.png')


def train(train_loader, val_loader, learn_rate, batch_size=128, hidden_dim=256, EPOCHS=10):

    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 200
    n_layers = 2
    # Instantiating the models
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of trainable parameters: ", pytorch_total_params)
    logging.info("Total number of trainable parameters: " +
                 str(pytorch_total_params))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of GRU model")
    epoch_times = []
    val_loss_epochs = []
    training_loss_epochs = []
    # Start training loop
    for epoch in tqdm(range(1, EPOCHS+1), desc="Epochs Loop"):
        start_time = time.perf_counter()
        total_loss = 0.
        counter = 0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data
            counter += 1
            optimizer.zero_grad()

            out, _ = model(inputs.to(device).float())
            loss = criterion(out.squeeze(), labels.to(device).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(
                    epoch, counter, len(train_loader), total_loss/counter))
                logging.info("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(
                    epoch, counter, len(train_loader), total_loss/counter))
        current_time = time.perf_counter()
        writer.add_scalar('Loss/train', total_loss /
                          len(train_loader), epoch)
        training_loss_epochs.append(total_loss/len(train_loader))
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch,
              EPOCHS, total_loss/len(train_loader)))
        logging.info("Epoch {}/{} Done, Total Loss: {}".format(
            epoch, EPOCHS, total_loss/len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(
            str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
        val_losses = []
        val_loss_min = np.Inf
        model.eval()
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels = data
            output, _ = model(inputs.to(device).float())
            val_loss = criterion(output.squeeze(), labels.to(
                device).float())
            val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        writer.add_scalar('Loss/val', val_loss, epoch)
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            torch.save({'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'loss': val_loss,
                        'batch_size': batch_size,
                        'learn_rate': learn_rate,
                        }, model_path)
        val_loss_epochs.append(val_loss)
        print("Validation Loss: {}".format(val_loss))
        logging.info(
            "Epoch {}/{} Done,  Validation Loss: {}".format(epoch, EPOCHS, val_loss))
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    return model, training_loss_epochs, val_loss_epochs


def evaluate(model, test_loader):
    model.eval()
    outputs = []
    targets = []
    start_time = time.perf_counter()
    for i, data in enumerate(tqdm(test_loader)):
        inputs, labels = data
        out = model(inputs.to(device).float())
        outputs.extend(out.squeeze().cpu().detach().numpy().tolist())
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


if __name__ == '__main__':

    train_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/train_pruned_efcamdat.csv')
    print('train_df shape', train_df.shape)
    val_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/val_pruned_efcamdat.csv')
    print('val_df shape', val_df.shape)
    print('training and validation data loaded')
    fasttext_model = '/cluster/work/sachan/abhinav/text_complexity/embedding/wiki-news-300d-1M.vec'
    fasttext = load_vectors(fasttext_model)
    print('fasttext loaded')
    train_dataset = EfcamdatDataset(train_df, fasttext)
    train_dataloader = DataLoader(train_dataset, batch_size=256,
                                  shuffle=True, collate_fn=collate_fn)
    val_dataset = EfcamdatDataset(val_df, fasttext)
    val_dataloader = DataLoader(val_dataset, batch_size=256,
                                shuffle=True, collate_fn=collate_fn)

    model, training_loss_epochs, val_loss_epochs = train(
        train_dataloader, val_dataloader, learn_rate=0.0001, batch_size=256, hidden_dim=256, EPOCHS=40)
    save_plots(training_loss_epochs, val_loss_epochs)

    data = {'training_loss_epochs': training_loss_epochs,
            'val_loss_epochs': val_loss_epochs}
    with open(model_folder + '/losses.pickle', 'wb') as f:
        pickle.dump(data, f)
