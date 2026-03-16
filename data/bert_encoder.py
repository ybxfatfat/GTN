import pandas as pd
import numpy as np
import json
from bert_serving.client import BertClient
bc = BertClient()

# train_path = '/home/luxiaoling/wukun/MIND/data/large/train/preprocessed_news.tsv'
# val_path = '/home/luxiaoling/wukun/MIND/data/large/dev/preprocessed_news.tsv'

# train_df = pd.read_csv(train_path, sep='\t', names=['news_id', 'category', 'subcategory', 'topic', 'title_lower', 'abstract_lower', 'clean_title', 'clean_abstract'])
# val_df = pd.read_csv(val_path, sep='\t', names=['news_id', 'category', 'subcategory', 'topic', 'title_lower', 'abstract_lower', 'clean_title', 'clean_abstract'])

# train_df['text'] = train_df['title_lower'] + ['. '] * train_df.shape[0] + train_df['abstract_lower']
# val_df['text'] = val_df['title_lower'] + ['. '] * val_df.shape[0] + val_df['abstract_lower']

# news_text_df = pd.concat([train_df[['news_id', 'text']], val_df[['news_id', 'text']]], axis=0)
# news_text_df.drop_duplicates(['news_id'], inplace=True)

# del train_df
# del val_df

news_path = '/home/luxiaoling/wukun/MIND/data/all_news.tsv'
news_df = pd.read_csv(news_path, sep='\t', names=['news_id', 'category', 'subcategory', 'topic', 'title_lower', 'abstract_lower', 'clean_title', 'clean_abstract'])
news_df['text'] = news_df['title_lower'] + ['. '] * news_df.shape[0] + news_df['abstract_lower']

news2idx_path = '/home/luxiaoling/wukun/MIND/data/news2idx.json'
with open(news2idx_path, 'r') as f:
    news2idx = json.load(f)

texts = list(news_df['text'].values)
# import pdb; pdb.set_trace()
ret = []
for i in range(0, len(texts), 1024):
    texts_slice = texts[i: i+1024]
    ret.append(bc.encode(texts_slice))
    
try:
    ret = np.concatenate(ret, axis=0)
    embeddings = np.zeros(shape=(len(news2idx), 768))

    for news_id, emb in zip(news_df['news_id'], ret):
        idx = news2idx[news_id]
        embeddings[idx] = emb

    save_path = '/home/luxiaoling/wukun/MIND/data/news_bert_embedding.npy'
    np.save(save_path, embeddings)
except Exception as e:
    import pdb; pdb.set_trace()
