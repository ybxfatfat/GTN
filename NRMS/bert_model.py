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
        self.model_name = 'NRMS'
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
        self.T_tensor = nn.Parameter(torch.arange(config.text_len), requires_grad=False)
        self.nid_embedding_semantic_pre = nn.Embedding.from_pretrained(torch.FloatTensor(np.load(config.news_bert_embedding_path)), freeze=True)  
        hist_len, d_input, d_model, n_head, text_len, p = config.history_len, config.embedding_dim, 256, 4, config.text_len, 0.2
        self.nid_proj = nn.Linear(768, d_model)  # bert embedding 投影
        self.news_encoder = NewsEncoder(d_model, d_model, n_head, hist_len)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 4, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2)
        )

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
        
        history_nids_embedding = self.nid_proj(self.nid_embedding_semantic_pre(history_nids))
        user_embedding = self.news_encoder(history_nids_embedding, history_len)
        news_embedding = self.nid_proj(self.nid_embedding_semantic_pre(impr_nid))
        
        combine_input = torch.cat([
            user_embedding,
            news_embedding,
            user_embedding + news_embedding,
            user_embedding * news_embedding
        ], dim=-1)
        
        out = self.fc(combine_input)
        loss = F.cross_entropy(out, label)
        return out, loss
    
class NewsEncoder(nn.Module):
    def __init__(self, d_input, d_model, n_head, text_len, p=0.2):
         super(NewsEncoder, self).__init__()
         self.T_tensor = nn.Parameter(torch.arange(text_len), requires_grad=False)
         self.pe = PositionalEncoding(d_input, p, text_len)
         self.self_att = MultiHeadedAttention(d_input, d_model, n_head, text_len)
         self.att = nn.Sequential(
             nn.Linear(d_model, 256),
             nn.Tanh(),
             nn.Linear(256, 1)
         )
         
    def forward(self, x, input_mask):
        """
        x: [B, T, D]
        input_mask: [B]
        """
        B, T, D = x.size()
        x = self.pe(x)
        x = self.self_att(x, input_mask)
        att_raw_score = self.att(x)
        mask = (self.T_tensor.view(1, -1) >= input_mask.view(-1, 1)).view(B, T, 1)
        att_raw_score = att_raw_score.masked_fill(mask, -1e9)
        att_score = F.softmax(att_raw_score, dim=1)
        out = torch.sum(att_score * x, dim=1)
        return out
   
 
class UserEncoder(nn.Module):
    def __init__(self, hist_len, d_input, d_model, n_head, text_len, p=0.2):
        super(UserEncoder, self).__init__()
        self.news_encoder = NewsEncoder(d_input, d_model, n_head, text_len, p)
        self.user_encoder = NewsEncoder(d_model, d_model, n_head, hist_len, p)
        
    def forward(self, x, history_words_len, history_len):
        """
        x: [B, L, T, D]
        history_words_len: [B, L]
        history_len: [B]
        """
        B, L, T, D = x.size()
        x = self.news_encoder(x.view(-1, T, D), history_words_len.view(-1)) # [B*L, d_model]
        x = x.view(B, L, -1)
        B, L, D = x.size()
        x = self.user_encoder(x, history_len) # [B, D]
        return x
    

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_input, d_model, n_head, text_len):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.query = nn.Linear(d_input, d_model)
        self.key = nn.Linear(d_input, d_model)
        self.value = nn.Linear(d_input, d_model)
        self.T_tensor = nn.Parameter(torch.arange(text_len), requires_grad=False)
        
    def forward(self, x, input_mask):
        """
        x: [B, T, D]
        input_mask: [B]
        """
        B, L, D = x.size()  
        query = self.query(x).view(B, L, self.n_head, -1).transpose(1, 2)
        key = self.key(x).view(B, L, self.n_head, -1).permute(0, 2, 3, 1)
        value = self.value(x).view(B, L, self.n_head, -1).transpose(1, 2)
        
        self_att_raw_score = torch.matmul(query, key) / np.sqrt(self.n_head)
        mask = (self.T_tensor.view(1, 1, 1, -1) >= input_mask.view(B, 1, 1, 1))
        self_att_raw_score = self_att_raw_score.masked_fill(mask, -1e9)
        self_att_score = F.softmax(self_att_raw_score, dim=-1)  # [B, N, L, L]
        out = torch.matmul(self_att_score, value).transpose(1, 2).contiguous().view(B, L, -1)  # [B, L, D]
        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_input, dropout, text_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(text_len, d_input)
        position = torch.arange(0, text_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_input, 2) *
                             -(np.log(10000.0) / d_input)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=False)
        
    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)