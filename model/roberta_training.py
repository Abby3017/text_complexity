import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

MAX_LEN = 256
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 128
LEARNING_RATE = 1e-05
EPOCHS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "/cluster/work/sachan/abhinav/model/roberta/"
model_folder_name = 'efcamdat_run1'
model_folder = os.path.join(folder_path, model_folder_name)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = model_folder + "/gru.pt"
log_path = model_folder + "/run.log"

logging.basicConfig(filename=log_path, level=logging.INFO)


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
    model.train()
    for epoch in range(EPOCHS):
        epoch_train_loss = 0
        epoch_valid_loss = 0
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
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(
                    epoch, i, len(train_loader), epoch_train_loss/i))
            loss.backward()
            optimizer.step()
            scheduler.step()
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
                loss = criterion(output, target.unsqueeze(1))
                epoch_valid_loss += loss.item()
                if i % 200 == 0:
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(
                        epoch, i, len(valid_loader), epoch_valid_loss/i))
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
