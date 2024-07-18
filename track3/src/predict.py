import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers.integrations import *
import time
import torch
import numpy as np
from dataset import InputDataset, InputDatasetAddOpt
from model import BertForSeq
from utils import log_creater, seed_everything
from transformers.utils.notebook import format_time
import argparse
from sklearn.metrics import mean_squared_error
from feature_special_token import SPECIAL_TOKEN
from feature_special_token_add_css import SPECIAL_TOKEN_CSS

os.environ["WANDB_DISABLED"] = "true"


def train(model, fold, train_loader, val_loader, test_loader, log):
    weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
    bias_params = [param for name, param in model.named_parameters() if "bias" in name]
    optimizer = AdamW([{'params': weight_params, 'weight_decay': 1e-5},
                       {'params': bias_params, 'weight_decay': 0}],
                      lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    best_model_path = args.save_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    log.info("Train batch size = {}".format(args.batch_size))
    log.info("Total steps = {}".format(total_steps))
    log.info("Training Start!")
    log.info('')

    best_val_loss = float('inf')
    best_val_acc = -1
    for epoch in range(args.epochs):
        total_train_loss = 0
        t0 = time.time()
        model.to(args.device)
        model.train()
        for step, batch in enumerate(train_loader):
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
            if step % 10 == 0:
                print('step : {},   loss : {}'.format(step, loss.item()))
        avg_train_loss = total_train_loss / len(train_loader)
        train_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch + 1, args.epochs, avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')
        avg_val_loss, avg_val_acc, avg_adj1, avg_mse = evaluate(model, val_loader)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] Avg_Val_Loss={:.5f} ===='.format(epoch + 1, args.epochs, avg_val_loss))
        log.info('====Epoch:[{}/{}] Avg_Val_Acc = {:.5f} | Avg_Adj1_Acc = {:.5f} | Avg_MSE_Score={:.5f}===='.format(
            epoch + 1, args.epochs, avg_val_acc, avg_adj1, avg_mse))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')

        if avg_val_acc > best_val_acc:  # 验证集模型保存指标
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path + 'fold{}.pt'.format(fold))
            print('Model Saved!')
    log.info('   Training Completed!')
    log.info('')
    model.load_state_dict(torch.load(best_model_path + 'fold{}.pt'.format(fold)))
    _, test_acc, avg_adj1, avg_mse = evaluate(model, test_loader)
    log.info('avg_test_acc = {:.5f} ==== avg_adj1 = {:.5f} ==== avg_mse = {:.5f}'.format(test_acc, avg_adj1, avg_mse))
    log.info('')
    log.info('==========================')
    return test_acc, avg_adj1, avg_mse


def evaluate(model, data_loader):
    model.eval()
    predict_readability = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        token_type_ids = batch['token_type_ids'].to(args.device)
        labels = batch['label'].to(args.device)

        with torch.no_grad():
            loss, output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        output = torch.argmax(output, dim=1)
        predict_readability += output.tolist()

    return predict_readability


def main():
    seed_everything(args.seed)

    log = log_creater(output_dir='../cache/logs/')
    log.info(args.model_path)
    log.info('EPOCH = {}; LR = {}'.format(args.epochs, args.learning_rate))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, additional_special_tokens = SPECIAL_TOKEN_CSS)
    css_data = pd.read_csv(
        'D:/pythonProject/Chinese_Sentence_Readability/sentence-readability-for-chinese/feature_simplification_dataset/css_9features.csv')
    css_data = css_data.copy()
    mcts_dev_orig = css_data.loc[css_data['sentence_id'] == 'mcts.dev.orig'].reset_index(drop = True)  # 需要加上reset_index，不然会报数组越界
    mcts_dev_orig.insert(loc = 3, column = 'label', value = 0)
    mcts_dev_simp_0 = css_data.loc[css_data['sentence_id'] == 'mcts.dev.simp.0'].reset_index(drop = True)
    mcts_dev_simp_0.insert(loc = 3, column = 'label', value = 0)
    mcts_dev_simp_1 = css_data.loc[css_data['sentence_id'] == 'mcts.dev.simp.1'].reset_index(drop = True)
    mcts_dev_simp_1.insert(loc = 3, column = 'label', value = 0)
    mcts_dev_simp_2 = css_data.loc[css_data['sentence_id'] == 'mcts.dev.simp.2'].reset_index(drop = True)
    mcts_dev_simp_2.insert(loc = 3, column = 'label', value = 0)
    mcts_dev_simp_3 = css_data.loc[css_data['sentence_id'] == 'mcts.dev.simp.3'].reset_index(drop = True)
    mcts_dev_simp_3.insert(loc = 3, column = 'label', value = 0)
    mcts_dev_simp_4 = css_data.loc[css_data['sentence_id'] == 'mcts.dev.simp.4'].reset_index(drop = True)
    mcts_dev_simp_4.insert(loc = 3, column = 'label', value = 0)

    mcts_test_orig = css_data.loc[css_data['sentence_id'] == 'mcts.test.orig'].reset_index(
        drop = True)  # 需要加上reset_index，不然会报数组越界
    mcts_test_orig.insert(loc = 3, column = 'label', value = 0)
    mcts_test_simp_0 = css_data.loc[css_data['sentence_id'] == 'mcts.test.simp.0'].reset_index(drop = True)
    mcts_test_simp_0.insert(loc = 3, column = 'label', value = 0)
    mcts_test_simp_1 = css_data.loc[css_data['sentence_id'] == 'mcts.test.simp.1'].reset_index(drop = True)
    mcts_test_simp_1.insert(loc = 3, column = 'label', value = 0)
    mcts_test_simp_2 = css_data.loc[css_data['sentence_id'] == 'mcts.test.simp.2'].reset_index(drop = True)
    mcts_test_simp_2.insert(loc = 3, column = 'label', value = 0)
    mcts_test_simp_3 = css_data.loc[css_data['sentence_id'] == 'mcts.test.simp.3'].reset_index(drop = True)
    mcts_test_simp_3.insert(loc = 3, column = 'label', value = 0)
    mcts_test_simp_4 = css_data.loc[css_data['sentence_id'] == 'mcts.test.simp.4'].reset_index(drop = True)
    mcts_test_simp_4.insert(loc = 3, column = 'label', value = 0)


    add_orig_dataset = InputDataset(add_orig, tokenizer, args.max_input_length)
    add_orig_loader = DataLoader(add_orig_dataset, batch_size = args.batch_size, shuffle = True)

    add_simp_dataset = InputDataset(add_simp, tokenizer, args.max_input_length)
    add_simp_loader = DataLoader(add_simp_dataset, batch_size = args.batch_size, shuffle = True)
    test_orig_dataset = InputDataset(test_orig, tokenizer, args.max_input_length)
    test_orig_loader = DataLoader(test_orig_dataset, batch_size = args.batch_size, shuffle = True)
    test_simp_1_dataset = InputDataset(test_simp_1, tokenizer, args.max_input_length)
    test_simp_1_loader = DataLoader(test_simp_1_dataset, batch_size = args.batch_size, shuffle = True)
    test_simp_2_dataset = InputDataset(test_simp_2, tokenizer, args.max_input_length)
    test_simp_2_loader = DataLoader(test_simp_2_dataset, batch_size = args.batch_size, shuffle = True)

    model = BertForSeq(args, tokenizer)
    model.load_state_dict(torch.load('../best_model/for_css_test/fold3.pt'))
    model.to(args.device)
    add_orig_pred = evaluate(model, add_orig_loader)
    add_simp_pred = evaluate(model, add_simp_loader)
    test_orig_pred = evaluate(model, test_orig_loader)
    test_simp_1_pred = evaluate(model, test_simp_1_loader)
    test_simp_2_pred = evaluate(model, test_simp_2_loader)

    pred_writer = open('../../readability_predition_visual/css/css_readbility_predition.jsonlines', 'w')
    pred_writer.write(json.dumps({"add_orig_pred": add_orig_pred}) + '\n')
    pred_writer.write(json.dumps({"add_simp_pred": add_simp_pred}) + '\n')
    pred_writer.write(json.dumps({"test_orig_pred": test_orig_pred}) + '\n')
    pred_writer.write(json.dumps({"test_simp_1_pred": test_simp_1_pred}) + '\n')
    pred_writer.write(json.dumps({"test_simp_2_pred": test_simp_2_pred}) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type = str, default = '../best_model/for_css_test/')
    parser.add_argument('--model_path', type = str, default = 'C:/PLMs/bert-base-chinese')
    parser.add_argument('--num_labels', type = int, default = 10)  # 类别数量
    parser.add_argument('--max_input_length', type = int, default = 128)
    parser.add_argument('--epochs', type = int, default = 8)
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_feature', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 2023)
    parser.add_argument('--device', type = str, default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    main()














