import torch
from transformers import BertModel, AutoModel
from torch import nn


class BertForSeq(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.num_labels = args.num_labels
        self.bert = AutoModel.from_pretrained(args.model_path)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.linear_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, args.hidden_size),
            nn.Dropout(args.dropout),
        )

        self.bert_head = nn.Sequential(
            nn.Linear(args.hidden_size, args.num_labels),
        )

        self.bert_feature_head = nn.Sequential(
            nn.Linear(args.hidden_size + args.num_feature, args.num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, feature=None, labels=None):
        loss = None
        if not feature:
            output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = self.bert_head(self.linear_head(output[1]))
        else:
            output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = self.bert_feature_head(torch.cat((self.linear_head(output[1]), feature), 1))
        if labels is not None:
            loss_fc = nn.CrossEntropyLoss()
            loss = loss_fc(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits




