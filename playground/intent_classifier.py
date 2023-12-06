#!/usr/bin/env python3

from torch import nn
from transformers import BertModel


class INTENT_CLASSIFIER(nn.Module):
    def __init__(self, freeze_bert=True):
        super(INTENT_CLASSIFIER, self).__init__()

        self.bert_layers = BertModel.from_pretrained(
            "bert-base-multilingual-cased", return_dict=False
        )
        self.linear1 = nn.Linear(768, 300)
        self.linear11 = nn.Linear(300, 8)
        self.linear2 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.5)

        if freeze_bert:
            for param in self.bert_layers.parameters():
                param.requires_grad = False

    def forward(self, token_ids, atten_mask):
        """Both argument are of shape: batch_size, max_seq_len"""
        _, CLS = self.bert_layers(token_ids, attention_mask=atten_mask)
        logits = self.dropout(self.linear1(CLS))
        logits = self.dropout(self.linear11(logits))
        logits = self.linear2(logits)

        return logits
