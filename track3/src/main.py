import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.integrations import *
import time
import torch
import numpy as np
from dataset import InputDataset
from model import BertForSeq
from utils import log_creater, seed_everything, quadratic_weighted_kappa
from transformers.utils.notebook import format_time
import argparse
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

os.environ["WANDB_DISABLED"] = "true"


def train(model, train_loader, val_loader, log):
    weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
    bias_params = [param for name, param in model.named_parameters() if "bias" in name]
    optimizer = AdamW([{'params': weight_params, 'weight_decay': 1e-5},
                       {'params': bias_params, 'weight_decay': 0}],
                      lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    best_model_path = args.save_path + f'_batch{args.batch_size}_lr{args.learning_rate}/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    log.info("Train batch size = {}".format(args.batch_size))
    log.info("Total steps = {}".format(total_steps))
    log.info("Training Start!")
    log.info('')

    best_score, step_count = -1, 0
    for epoch in range(args.epochs):
        total_train_loss = 0
        t0 = time.time()
        model.to(args.device)
        model.train()
        for step, batch in enumerate(train_loader):
            step_count += 1
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            token_type_ids = batch['token_type_ids'].to(args.device)
            labels = batch['label'].to(args.device)
            model.zero_grad()
            loss, output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 labels=labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        score_dict, track3_score = evaluate(model, val_loader)
        if track3_score > best_score: best_score = track3_score
        train_time = format_time(time.time() - t0)
        log.info('==== Epoch:[{}/{}] score_dict = {} | score = {:.5f} | best-score = {:.5f} ===='.format(
            epoch + 1, args.epochs, score_dict, track3_score, best_score))
        log.info('')

    log.info('   Training Completed!')
    log.info('==========================')
    return best_score


def evaluate(model, data_loader):
    model.eval()
    label_list, pred_result = [], []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        labels = batch['label'].to(args.device)

        with torch.no_grad():
            loss, output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        output = torch.argmax(output, dim=1)
        pred_result += output.data.tolist()
        label_list += labels.data.tolist()

    acc = accuracy_score(label_list, pred_result)
    f1 = f1_score(label_list, pred_result, average = "micro")
    qwk = quadratic_weighted_kappa(label_list, pred_result)
    track3_score = 0.5*f1+0.2*qwk+0.3*acc

    return {"acc": acc, "f1": f1, "qwk": qwk}, track3_score


def main():
    seed_everything(args.seed)

    log = log_creater(output_dir='../cache/logs/')
    log.info(args.model_path)
    log.info('EPOCH = {}; LR = {}'.format(args.epochs, args.learning_rate))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    tr_data = pd.read_csv('../data/train.csv')
    va_data = pd.read_csv('../data/val.csv')

    tr_dataset = InputDataset(tr_data, tokenizer, args.max_input_length)
    tr_data_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    va_dataset = InputDataset(va_data, tokenizer, args.max_input_length)
    va_data_loader = DataLoader(va_dataset, batch_size=args.batch_size, shuffle=False)

    model = BertForSeq(args, tokenizer)

    best_score = train(model, tr_data_loader, va_data_loader, log)
    log.info('best score: {:.5f}'.format(best_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, default = '../best_model/baseline')
    parser.add_argument('--model_path', type = str, default = "C:/PLMs/bert-base-chinese")
    parser.add_argument('--num_labels', type = int, default = 3)  # 类别数量
    parser.add_argument('--max_input_length', type = int, default = 256)
    parser.add_argument('--epochs', type = int, default = 20)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 4)
    parser.add_argument('--num_feature', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 2023)
    parser.add_argument('--device', type = str, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    main()














