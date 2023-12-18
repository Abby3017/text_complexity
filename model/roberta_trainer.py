import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          get_scheduler)

from text_complexity.model.pretrained.roberta_class_trainer import (
    RobertaEfcamdatDataset, RobertaNet)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 8
MODEL_NAME = 'roberta-base'
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LEARNING_RATE = 1e-05

# os.environ["WANDB_DISABLED"] = "true"
# srun -n 1 --cpus-per-task=4 --time=4:00:00 --job-name="learn1" --mem-per-cpu=16384 --gpus=1 --gres=gpumem:32G --pty python3 text_complexity/model/roberta_trainer.py

folder_path = "/cluster/work/sachan/abhinav/model/roberta/"
model_folder_name = 'efcamdat_trainer3'
model_folder = os.path.join(folder_path, model_folder_name)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_path = model_folder + "/roberta_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".pt"

log_path = model_folder + "/run_" + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        criterion = nn.MSELoss()
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = criterion(logits.squeeze(), labels.to(device).float())
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    predictions, labels = p
    errors = abs(predictions-labels)
    return {
        'mae': np.mean(errors),
        'mdae': np.median(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
    }


if __name__ == "__main__":

    train_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/train_pruned_efcamdat.csv')
    print('train_df shape', train_df.shape)
    val_df = pd.read_csv(
        '/cluster/work/sachan/abhinav/text_complexity/data/val_pruned_efcamdat.csv')
    print('val_df shape', val_df.shape)
    # train_df = train_df[:2000]
    # val_df = val_df[:200]
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

    training_args = TrainingArguments(
        "test-trainer", evaluation_strategy="epoch")

    training_args = TrainingArguments(
        output_dir=model_folder,         # output directory
        num_train_epochs=8,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=200,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=model_folder,            # directory for storing logs
        learning_rate=LEARNING_RATE,
        metric_for_best_model='eval_rmse',
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        label_names=["labels"],
        report_to="none",
    )

    num_training_steps = (EPOCHS * len(train_loader)) // TRAIN_BATCH_SIZE
    model = RobertaNet(MODEL_NAME)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    lr_scheduler = get_scheduler(
        "linear",  # can try "cosine" too
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    trainer = CustomTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(model_path)
