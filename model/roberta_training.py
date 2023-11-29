import logging
import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from text_complexity.model.pretrained.roberta import (RobertaEfcamdatDataset,
                                                      RobertaNet)
from text_complexity.utils.plot_util import save_plots

MAX_LEN = 256
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
LEARNING_RATE = 1e-05
EPOCHS = 30
MODEL_NAME = 'roberta-base'

torch.cuda.init()
torch.cuda.empty_cache()

# after each login
# module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy hdf5/1.10.1
# srun -n 1 --cpus-per-task=4 --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --gpus=2 --gres=gpumem:8128 --pty python3 text_complexity/model/roberta_training.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = "/cluster/work/sachan/abhinav/model/roberta/"
model_folder_name = 'efcamdat_run1'
model_folder = os.path.join(folder_path, model_folder_name)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = model_folder + "/gru_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".pt"

log_path = model_folder + "/run_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
tb_model_path = model_folder + "/tensorboard_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + "/"

logging.basicConfig(filename=log_path, level=logging.INFO)
writer = SummaryWriter(log_dir=tb_model_path)
logging.info("device: " + str(device))
print("device: " + str(device))


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    # os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True


def train(model, train_loader, valid_loader, optimizer, criterion, tokenizer):
    train_loss = []
    valid_loss = []
    best_valid_loss = float('Inf')
    # for name, param in model.model.named_parameters():
    #     param.requires_grad = False
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of trainable parameters: ", pytorch_total_params)
    logging.info("Total number of trainable parameters: " +
                 str(pytorch_total_params))
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        epoch_valid_loss = 0
        start_time = time.perf_counter()
        for i, batch in enumerate(tqdm(train_loader)):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(
                device, dtype=torch.long)
            target = batch['labels'].to(device, dtype=torch.float)
            optimizer.zero_grad()
            output = model(ids, mask, token_type_ids)
            loss = criterion(output.squeeze(), target.to(device).float())
            epoch_train_loss += loss.item()
            if i % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Training Loss for Epoch: {}".format(
                    epoch, i+1, len(train_loader), epoch_train_loss/(i+1)))
                logging.info("Epoch {}......Step: {}/{}....... Average Training Loss for Epoch: {}".format(
                    epoch, i+1, len(train_loader), epoch_train_loss/(i+1)))
                writer.add_scalar(
                    'Training Loss', epoch_train_loss/200, epoch*len(train_loader)+(i+1))
            loss.backward()
            optimizer.step()
        current_time = time.perf_counter()
        print("Total Time Elapsed: {} seconds".format(
            str(current_time-start_time)))
        logging.info("Total Time Elapsed: {} seconds".format(
            str(current_time-start_time)))
        epoch_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(epoch_train_loss)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(valid_loader)):
                ids = batch['ids'].to(device, dtype=torch.long)
                mask = batch['mask'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(
                    device, dtype=torch.long)
                target = batch['labels'].to(device, dtype=torch.float)
                output = model(ids, mask, token_type_ids)
                loss = criterion(output.squeeze(), target.to(device))
                epoch_valid_loss += loss.item()
                if i % 200 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Validation Loss for Epoch: {}".format(
                        epoch, i, len(valid_loader), epoch_valid_loss/(i+1)))
                    logging.info("Epoch {}......Step: {}/{}....... Average Validation Loss for Epoch: {}".format(
                        epoch, i, len(valid_loader), epoch_valid_loss/(i+1)))
            epoch_valid_loss = epoch_valid_loss / len(valid_loader)
            valid_loss.append(epoch_valid_loss)
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                torch.save({'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict(),
                            'loss': epoch_valid_loss,
                            'batch_size': TRAIN_BATCH_SIZE,
                            }, model_path)
                tokenizer.save_vocabulary(model_folder)
        logging.info(f'\tTrain Loss: {epoch_train_loss:.3f}')
        logging.info(f'\t Val. Loss: {epoch_valid_loss:.3f}')
        print(f'\tTrain Loss: {epoch_train_loss:.3f}')
        print(f'\t Val. Loss: {epoch_valid_loss:.3f}')
    return train_loss, valid_loss


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


if __name__ == "__main__":
    set_random_seed(42)
    train_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/train_pruned_efcamdat.csv')
    print('train_df shape', train_df.shape)
    val_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/val_pruned_efcamdat.csv')
    print('val_df shape', val_df.shape)
    print('training and validation data loaded')
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir="/cluster/work/sachan/abhinav/model/roberta/cache", resume_download=True)
    print('tokenizer loaded')
    train_dataset = RobertaEfcamdatDataset(
        train_df, tokenizer, MAX_LEN)
    val_dataset = RobertaEfcamdatDataset(
        val_df, tokenizer, MAX_LEN)
    print('val_dataset & train_dataset loaded')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=4)
    print('train and val dataloader loaded')

    model = RobertaNet(MODEL_NAME)
    writer.add_graph(model, (next(iter(train_loader))['ids'], next(
        iter(train_loader))['mask'], next(iter(train_loader))['token_type_ids']))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    logging.info("Model Loaded & Training Started")
    train_loss, val_loss = train(model=model, train_loader=train_loader, valid_loader=val_loader,
                                 optimizer=optimizer, criterion=criterion, tokenizer=tokenizer)

    save_plots(train_loss, val_loss, model_folder)

    data = {'training_loss_epochs': train_loss,
            'val_loss_epochs': val_loss}
    with open(model_folder + '/losses.pickle', 'wb') as f:
        pickle.dump(data, f)
