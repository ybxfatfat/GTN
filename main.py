# coding: utf-8
import sys
sys.path.insert(0, './models')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from train import train, get_time_dif
import importlib
import argparse
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # device_cnt = torch.cuda.device_count()
    # default_device = ','.join(map(str, range(device_cnt)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='choice one: MPNRec, wide_deep, DIN, NRMS, MEIRec')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--sample_size', default=30, type=int)
    parser.add_argument('--history_len', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--neg_pos_ratio', default=5, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--batches_per_check', default=500, type=int)
    parser.add_argument('--require_improvement', default=3000, type=int)
    parser.add_argument('--text_len', default=128, type=int)
    parser.add_argument('--agg_method', default='pooling', type=str) # pooling or self_attention
    parser.add_argument('--text_encoding', default='bert', type=str)
    parser.add_argument('--use_pretrain', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--device', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--reinit_epoch', action='store_true')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 需要放在所有访问GPU代码之前，否则不生效

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    module = importlib.import_module(args.model + '.{}_model'.format(args.text_encoding))
    config = module.Config(args)
    logging = config.logging

    if args.model in ['wide_deep', 'DIN', 'NRMS']:
        util_module = importlib.import_module('common_utils')
    else:
        util_module = importlib.import_module(args.model + '.utils')
    dataset = util_module.MyDataset(config)

    model = module.Model(config)
    if config.multi_gpu:
        model = nn.DataParallel(model)
    logging.info(model)
    model.to(config.device)

    # train
    train(config, model, dataset)
    logging.info('All Time usage: %s', get_time_dif(start_time))