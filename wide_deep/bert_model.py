import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import logging
import sys


class Config(object):
    def __init__(self, args):
        self.model_name = 'wide_deep'
        self.train_behavior_path = './data/behavior_sample_train.tsv'
        self.val_behavior_path = './data/behavior_sample_val.tsv'
        self.news_path = './data/all_news.tsv'
        self.news2idx_path = './data/news2idx.json'
        self.uid2idx_path = './data/uid2idx.json'
        self.cate2idx_path = './data/cate2idx.json'
        self.vocab_path = './data/vocab.json'
        self.uid_embedding_deepwalk_path = './data/uid_embeddings_deepwalk.npy'
        self.news_embeddings_deepwalk_path = './data/news_embeddings_deepwalk.npy'
        self.news_bert_embedding_path = './data/news_bert_embedding.npy'
        self.w2v_embedding_path = './data/w2v_embedding.npy'
        self.cate2news_path = './data/cate2news.json'
        self.topic2news_path = './data/topic2news.json'
        self.news2uid_path = './data/news2uid.json'
        self.new_user_path = './data/new_user.csv'
        self.new_news_path = './data/new_news.csv'
        self.text_encoding = args.text_encoding
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug = args.debug
        self.device_count = torch.cuda.device_count()
        self.multi_gpu = self.device_count > 1 and args.multi_gpu
        self.text_len = args.text_len # default 512
        self.learning_rate = args.lr
        self.history_len = args.history_len
        self.batch_size = args.batch_size
        self.neg_pos_ratio = args.neg_pos_ratio
        self.num_epochs = args.num_epochs
        self.batches_per_check = args.batches_per_check
        self.require_improvement = args.require_improvement
        self.use_pretrain = args.use_pretrain
        self.agg_method = args.agg_method # pooling or self-attention
        self.embedding_dim = 128
        self.context_emb_dim = 16
        self.uid_num = -1
        self.nid_num = -1
        self.cate_num = -1
        self.topic_num = 201
        self.vocab_size = -1
        trial_name = 'debug_{}_prefix_{}_text_encoding_{}_history_len_{}_pretrain_{}_agg_method_{}_lr_{}_neg_pos_ratio_{}_epochs_{}_batches_per_check_{}_text_len_{}'.format(
            self.debug,
            args.prefix,
            self.text_encoding,
            self.history_len,
            self.use_pretrain,
            self.agg_method,
            self.learning_rate,
            self.neg_pos_ratio,
            self.num_epochs,
            self.batches_per_check,
            self.text_len
        )
        self.summary_dir = './models/{}/summary/{}'.format(self.model_name, trial_name)
        self.save_path = './models/{}/checkpoint/{}.ckpt'.format(self.model_name, trial_name)
        self.log_path = './models/{}/logs/{}.log'.format(self.model_name, trial_name)
        self.pred_label_save_path = './models/{}/predict/{}.json'.format(self.model_name, trial_name)
        self.class_list = ['0', '1']
        if self.debug:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)-15s %(message)s')
        else:
            logging.basicConfig(level=logging.INFO, filename=self.log_path, filemode='w', format='%(asctime)-15s %(message)s')
        self.logging = logging
    
    def get_parameters(self):
        return {
            'learning_rate': self.learning_rate,
            'history_len': self.history_len,
            'batch_size': self.batch_size,
            'neg_pos_ratio': self.neg_pos_ratio,
            'num_epochs': self.num_epochs,
            'batches_per_check': self.batches_per_check,
            'require_improvement': self.require_improvement,
            'text_encoding': self.text_encoding,
            'text_len': self.text_len,
            'agg_method': self.agg_method,
            'use_pretrain': self.use_pretrain
        }

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.L_tensor = nn.Parameter(torch.arange(config.history_len), requires_grad=False)
        if config.use_pretrain:
            self.load_pretrain_embeddings(config)
        else:
            self.uid_embedding = nn.Embedding(config.uid_num, config.embedding_dim)
            self.nid_embedding_structual = nn.Embedding(config.nid_num, config.embedding_dim)
        self.nid_embedding_semantic_pre = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(config.news_bert_embedding_path)), freeze=True)  
        self.category_embedding = nn.Embedding(config.cate_num, config.embedding_dim)
        self.topic_embedding = nn.Embedding(config.topic_num, config.embedding_dim)
        self.week_embedding = nn.Embedding(7, config.context_emb_dim)
        self.hour_embedding = nn.Embedding(24, config.context_emb_dim)
        self.nid_proj = nn.Linear(768, config.embedding_dim)  # bert embedding 投影  
        input_dim = 7 * config.embedding_dim + 2 * config.context_emb_dim + 3
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), # 仅适合于2D and 3D
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2)
        )
        self.init_network()
        
    # 权重初始化，默认xavier
    def init_network(self, method='xavier', exclude='embedding', seed=123):
        for name, w in self.named_parameters():
            if exclude not in name:
                if len(w.size()) < 2:
                    continue
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass
    
    def load_pretrain_embeddings(self, config):
        self.uid_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(config.uid_embedding_deepwalk_path)), freeze=False)
        self.nid_embedding_structual = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(config.news_embeddings_deepwalk_path)), freeze=False)

    def forward(self, batch_datas):
        """@params:
        uid: [B]
        impr_nid: [B]
        history_nids: [B, config.history_len]
        history_len: [B]
        category_ids: [B, config.history_len]
        topic_ids: [B, config.history_len]
        impr_category_id: [B]
        impr_topic_id: [B]
        week: [B]
        hour: [B]
        relative_days: [B]
        relative_hours: [B]
        relative_seconds: [B]
        """
        uid, impr_nid, history_nids, history_len, category_ids, topic_ids, \
            impr_category_id, impr_topic_id, week, hour, relative_days, relative_hours, relative_seconds, label = batch_datas
        
        uid_embedding = self.uid_embedding(uid)
        history_nids_embedding = self.nid_embedding_structual(history_nids) + self.nid_proj(self.nid_embedding_semantic_pre(history_nids))
        category_embedding = self.category_embedding(category_ids)
        topic_embedding = self.topic_embedding(topic_ids)
        impr_nid_embedding = self.nid_embedding_structual(impr_nid) + self.nid_proj(self.nid_embedding_semantic_pre(impr_nid))
        impr_cate_embedding = self.category_embedding(impr_category_id)
        impr_topic_embedding = self.topic_embedding(impr_topic_id)
        week_embedding = self.week_embedding(week)
        hour_embedding = self.hour_embedding(hour)
        
        B, L = history_nids.size()
        history_embedding = torch.cat([history_nids_embedding, category_embedding, topic_embedding], dim=-1) # B, L, 3D
        mask = (self.L_tensor.view(1, -1) >= history_len.view(-1, 1)).view(B, L, 1)
        history_embedding = history_embedding.masked_fill(mask, 0)
        history_embedding = torch.sum(history_embedding, dim=1) # B, 3D
        
        impr_embedding = torch.cat([impr_nid_embedding, impr_cate_embedding, impr_topic_embedding], dim=-1) # B, 3D
        
        combine_input = torch.cat([
            uid_embedding,
            history_embedding,
            impr_embedding,
            week_embedding,
            hour_embedding,
            relative_days.view(-1, 1),
            relative_hours.view(-1, 1),
            relative_seconds.view(-1, 1)
        ], dim=-1)
        
        out = self.fc(combine_input)
        loss = F.cross_entropy(out, label)
        return out, loss
        
