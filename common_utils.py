# coding: utf-8
import torch
from tqdm import tqdm
import time
import datetime
from datetime import timedelta
import random
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from prefetch_generator import BackgroundGenerator


NEWS_PAD = 'NewsPAD'
CATEGORY_PAD = 'Cate_PAD'
CATEGORY_UNK = 'Cate_UNK'
USER_PAD = 'UID_PAD'
USER_UNK = 'UID_UNK'
TOPIC_PAD_ID = 200
WORDPAD = 'PAD'
WORDUNK = 'UNK'


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
def collate_fn(batch_data, device, multi_gpu=True):
    L = len(batch_data[0])
    tensors = [torch.as_tensor([x[i] for x in batch_data]) for i in range(L)]
    if not multi_gpu:
        tensors = [tensor.to(device) for tensor in tensors]
    return tensors

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def load_json_object(json_path):
    with open(json_path, 'r') as f:
        obj = json.load(f)
    return obj

def load_df(behavior_path, debug=False):
    columns = ['uid', 'impr_time', 'week', 'hour', 'relative_days', 'relative_hours', 'relative_seconds', 'history', 'cur_impr']
    df = pd.read_csv(behavior_path, sep='\t', names=columns)
    if debug:
        df = df.sample(10000, random_state=1)
    df.fillna('', inplace=True)
    df['impr_date'] = df.impr_time.apply(lambda x: x.split()[0])
    df['relative_days'] = df['relative_days'].apply(lambda x: x / 6.0)
    df['relative_hours'] = df['relative_hours'].apply(lambda x: x / (6.0 * 24))
    df['relative_seconds'] = df['relative_seconds'].apply(lambda x: x / (6.0 * 24 * 3600))
    return df   

def load_news_dict(news_path, news2idx, cate2idx, vocab, text_len):
    news_dict = {}
    columns = ['news_id', 'category', 'subcategory', 'topic', 'title_lower', 'abstract_lower', 'clean_title', 'clean_abstract']
    df = pd.read_csv(news_path, sep='\t', names=columns)
    df['text'] = df['clean_title'] + [' '] * df.shape[0] + df['clean_abstract']
    for news, cate, topic, text in zip(df.news_id, df.subcategory, df.topic, df.text):
        nid, cate_id, topic_id = news2idx[news], cate2idx[cate], int(topic)
        words = [vocab[w] if w in vocab else vocab[WORDUNK] for w in text.split()]
        if len(words) < text_len:
            words_len = len(words)
            words.extend([vocab[WORDPAD]] * (text_len - len(words)))
        else:
            words = words[:text_len]
            words_len = text_len
        news_dict[nid] = (cate_id, topic_id, words, words_len)
    defalut_words = [vocab[WORDPAD]] * text_len
    news_dict[news2idx[NEWS_PAD]] = (cate2idx[CATEGORY_PAD], TOPIC_PAD_ID, defalut_words, 1)
    return news_dict
        

class MINDDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
        self.length = len(datas) 
    
    def __getitem__(self, index):
        return self.datas[index]
    
    def __len__(self):
        return self.length    

class MyDataset(object):
    def __init__(self, config):
        self.config = config
        self.news2idx = load_json_object(config.news2idx_path)
        self.uid2idx = load_json_object(config.uid2idx_path)
        self.cate2idx = load_json_object(config.cate2idx_path)
        config.uid_num, config.nid_num, config.cate_num = len(self.uid2idx), len(self.news2idx), len(self.cate2idx)
        self.train_df = load_df(config.train_behavior_path, config.debug)
        self.val_df = load_df(config.val_behavior_path, config.debug)
        self.logging = config.logging
        # with open(config.new_user_path, 'r') as f:
        #     new_user = f.readline()
        # self.config.new_user_set = {self.uid2idx[x] for x in new_user.split()}
        with open(config.new_news_path, 'r') as f:
            new_news = f.readline()
        self.config.new_news_set = {self.news2idx[x] for x in new_news.split()}
        self.vocab = load_json_object(config.vocab_path)
        self.news_dict = load_news_dict(config.news_path, self.news2idx, self.cate2idx, self.vocab, config.text_len)
        config.vocab_size = len(self.vocab)
        self.has_init = False
        self.logging.info('news size %s, user size %s, cate size %s, vocab size %s', len(self.news2idx), len(self.uid2idx), len(self.cate2idx), len(self.vocab))
        
    def init_per_epoch(self):
        if self.has_init:
            return
        self.train_iter = self.build_train_iter()
        self.val_iter, self.test_iter = self.build_val_test_iter()
        self.has_init = True
    
    def process_one_record(self, record, mode='train'):
        user, history, cur_impr, week, hour, relative_days, relative_hours, relative_seconds = record
        ret = []
        uid = self.uid2idx[user] if user in self.uid2idx else self.uid2idx[USER_UNK]
        history_nids = [self.news2idx[news] for news in history.split()]  # news must in news2idx
        news_pad_idx = self.news2idx[NEWS_PAD]
        user_pad_idx = self.uid2idx[USER_PAD]
        history_len = min(len(history_nids), self.config.history_len)
        history_len = max(1, history_len)
        if len(history_nids) < self.config.history_len:
            history_nids.extend([news_pad_idx] * (self.config.history_len - len(history_nids)))
        else:
            history_nids = history_nids[:self.config.history_len]
        
        category_ids = [self.news_dict[nid][0] for nid in history_nids]
        topic_ids = [self.news_dict[nid][1] for nid in history_nids]
        
        imprs = cur_impr.split()
        pos_news = list(filter(lambda x: x[-1] == '1', imprs))
        pos_nids = [self.news2idx[x[:-2]] for x in pos_news]
        neg_news = list(filter(lambda x: x[-1] == '0', imprs))
        neg_nids = [self.news2idx[x[:-2]] for x in neg_news]

        expected_neg_num = max(1, len(pos_nids)) * self.config.neg_pos_ratio
        if len(neg_nids) > expected_neg_num and mode == 'train':  # 训练模式才需要负采样
            neg_nids = np.random.choice(neg_nids, size=expected_neg_num, replace=False).tolist()
        
        labels = [1] * len(pos_nids) + [0] * len(neg_nids)
        impr_nids = pos_nids + neg_nids
        impr_category_ids = [self.news_dict[nid][0] for nid in impr_nids]
        impr_topic_ids = [self.news_dict[nid][1] for nid in impr_nids]
        
        ## 使用bert编码或者w2v编码
        if self.config.text_encoding == 'w2v':
            history_nids_words = [self.news_dict[nid][2] for nid in history_nids]
            history_words_len = [self.news_dict[nid][3] for nid in history_nids]
            impr_nids_words = [self.news_dict[nid][2] for nid in impr_nids]
            impr_words_len = [self.news_dict[nid][3] for nid in impr_nids]
            
            for impr_nid, impr_nid_words, impr_nid_words_len, impr_category_id, impr_topic_id, label in zip(impr_nids, impr_nids_words, impr_words_len, impr_category_ids, impr_topic_ids, labels):
                ret.append((
                    uid, impr_nid, history_nids, history_nids_words, history_words_len, history_len, category_ids, topic_ids, 
                    impr_nid_words, impr_nid_words_len, impr_category_id, impr_topic_id, week, hour, relative_days, relative_hours, relative_seconds, label
                ))
        else:   
            for impr_nid, impr_category_id, impr_topic_id, label in zip(impr_nids, impr_category_ids, impr_topic_ids, labels):
                ret.append(
                    (
                        uid, impr_nid, history_nids, history_len, category_ids, topic_ids, 
                        impr_category_id, impr_topic_id, week, hour, relative_days, relative_hours, relative_seconds, label
                    )
                )
        return ret
        
    def build_train_iter(self):
        self.logging.info('start build_train_iter...')
        seed = int(time.time())
        random.seed(seed)
        np.random.seed(seed)
        mode = 'train'
        start_time = time.time()
        df = self.train_df
        records = [record for record in zip(df.uid, df.history, df.cur_impr, df.week, df.hour, df.relative_days, df.relative_hours, df.relative_seconds)]
        datas = []
        for record in tqdm(records):
            ret = self.process_one_record(record, mode)
            datas.extend(ret)
        self.logging.info('build_train_iter done. train data size {}. time usage {}'.format(len(datas), get_time_dif(start_time)))
        random.shuffle(datas)
        dataset = MINDDataset(datas)
        data_iter = DataLoaderX(
            dataset, self.config.batch_size, shuffle=True, num_workers=0, pin_memory=False,
            collate_fn=lambda x: collate_fn(x, self.config.device, self.config.multi_gpu)
            )
        return data_iter
        
    def build_val_test_iter(self):
        self.logging.info('start build_val_test_iter....')
        mode = 'val'
        start_time = time.time()
        df = self.val_df
        records = [record for record in zip(df.uid, df.history, df.cur_impr, df.week, df.hour, df.relative_days, df.relative_hours, df.relative_seconds)]
        datas = []
        for record in tqdm(records):
            ret = self.process_one_record(record, mode)
            datas.extend(ret)

        # 划分验证集和测试集
        random.seed(2021)
        random.shuffle(datas)
        val_len = int(0.3 * len(datas))
        val_datas = datas[:val_len]
        test_datas = datas[val_len:]
        val_dataset = MINDDataset(val_datas)
        val_iter = DataLoaderX(
            val_dataset, self.config.batch_size, shuffle=False, num_workers=0, pin_memory=False,
            collate_fn=lambda x: collate_fn(x, self.config.device, self.config.multi_gpu)
            )
        test_dataset = MINDDataset(test_datas)
        test_iter = DataLoaderX(
            test_dataset, self.config.batch_size, shuffle=False, num_workers=0, pin_memory=False,
            collate_fn=lambda x: collate_fn(x, self.config.device, self.config.multi_gpu)
            )
        # 记录new_user and new news idx
        self.config.new_user_index = {i for i, x in enumerate(test_datas) if x[0] == self.uid2idx[USER_UNK]}
        self.config.new_news_index = {i for i, x in enumerate(test_datas) if x[1] in self.config.new_news_set}
        self.config.test_user_id = [x[0] for x in test_datas] # 用来计算gauc
        labels = [x[-1] for x in test_datas]
        pos_num, test_size = sum(labels), len(labels)
        pos_rate = float(pos_num) / test_size
        self.config.entropy = -pos_rate * np.log(pos_rate) - (1 - pos_rate) * np.log(1 - pos_rate)
        self.logging.info('build_val_test_iter done. val data size {}. test data size {}. pos_rate {:.3f}. time usage {}'.format(len(val_datas), len(test_datas), pos_rate, get_time_dif(start_time)))
        self.logging.info('new_user_index len %s', len(self.config.new_user_index))
        self.logging.info('new_news_index len %s', len(self.config.new_news_index))
        return val_iter, test_iter