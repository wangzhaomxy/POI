# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'poi_data'  # the dataset folder name under the same document with this python script.

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset) # See and change the config details in bert.py under the models
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # To guarantee the same results in duplicate.

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
