# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """Hyperparrameters configuration"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # train dataset
        self.dev_path = dataset + '/data/dev.txt'                                    # validation dataset
        self.test_path = dataset + '/data/test.txt'                                  # test dataset
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # class list
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # model save path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # training device

        self.require_improvement = 1000                                 # early stop iterations
        self.num_classes = len(self.class_list)                         # number of classes
        self.num_epochs = 10                                             # epoch number
        self.batch_size = 1024                                           # mini-batch大小, 1024 batch size utilize around 40G VRAM.
        self.pad_size = 32                                              # the padding size for each sentence.
        self.learning_rate = 5e-5                                       # learning rate
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # input sentence
        mask = x[2]  # mask part of padding，the same size with the sentence，padding with Zeros，for example:[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
