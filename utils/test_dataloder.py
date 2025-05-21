#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os, random, pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.task_utils import batchify
from models.template_embedding import TemplateEmbedding
from models.melody_embedding import MelodyEmbedding
from utils.indexed_datasets import IndexedDataset
from keys import *


# In[16]:


binary_dir = '/home/qihao/CS6207/binary'
words_dir = '/home/qihao/CS6207/binary/words'
hparams = {
    'batch_size': 8,
    'word_data_dir': '/home/qihao/CS6207/binary/words',
    'sentence_maxlen': 512,
    'hidden_size': 768,
}


# In[17]:


class L2MDataset (Dataset):
    def __init__(self, split, event2word_dict, hparams, shuffle=True):
        super().__init__()
        self.split = split
        self.hparams = hparams
        self.batch_size = hparams['batch_size']
        self.event2word_dict = event2word_dict
        
        self.data_dir = f"{hparams['word_data_dir']}"
        self.data_path = f'{hparams["word_data_dir"]}/{self.split}_words.npy'
        self.ds_name = split ## name of dataset
        
        self.data = np.load(open(self.data_path, 'rb'), allow_pickle= True)
        self.size = np.load(open(f'{hparams["word_data_dir"]}/{self.split}_words_length.npy', 'rb'), allow_pickle= True)
        self.shuffle = shuffle
        self.sent_maxlen = self.hparams['sentence_maxlen'] ## 512
        self.indexed_ds = None ## indexed dataset
        self.indices = [] ## indices to data samples
        
        if shuffle:
            self.indices = list(range(len(self.size)))  ## viz. number of data samples
            random.shuffle(self.indices)
            # print(f"indices: {self.indices}")
        else:
            self.indices = list(range(len(self.size)))
    
    def shuffle_indices(self):
        random.shuffle(self.indices)
    
    def ordered_indices(self):
        return self.indices

    def __len__(self):
        return self.size

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.ds_name}')
        return self.indexed_ds[index]
    
    def __getitem__(self, idx):
        # obtain one sentence segment according to the index
        item = self._get_item(idx)

        # input and output
        src_words = item['src_words']
        tgt_words = item['tgt_words']
        
        # to long tensors
        for k in src_keys:
            item[f'src_{k}'] = torch.LongTensor([word[k] for word in src_words])
        for k in tgt_keys:
            item[f'tgt_{k}'] = torch.LongTensor([word[k] for word in tgt_words])
            
        return item
    
    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS',10))  
    
    def collater(self, samples):
        # print(samples)
        if len(samples) == 0:
            return {}

        batch = {}
        for k in src_keys:
            batch[f'src_{k}'] = batchify([s[f'src_{k}'] for s in samples], pad_idx=0)
        for k in tgt_keys:
            batch[f'tgt_{k}'] = batchify([s[f'tgt_{k}'] for s in samples], pad_idx=0)
        
        # batch['n_src_tokens'] = sum([len(s['src_meter']) for s in samples])
        # batch['n_tgt_tokens'] = sum([len(s['tgt_word']) for s in samples])
        # batch['n_tokens'] = torch.LongTensor([s['n_tokens'] for s in samples])
        batch['input_path'] = [s['input_path'] for s in samples]
        batch['item_name'] = [s['item_name'] for s in samples]
        return batch


# In[18]:


def build_dataloader(dataset, shuffle, batch_size=10, endless=False):
    def shuffle_batches(batches):
        np.random.shuffle(batches)  # shuffle： 随机打乱数据
        # print(batches)
        return batches

    # batch sample and endless
    indices = dataset.ordered_indices()

    batch_sampler = []
    for i in range(0, len(indices), batch_size):
        batch_sampler.append(indices[i:i + batch_size])  # batch size [0:20],

    if shuffle:
        batches = shuffle_batches(list(batch_sampler))
        if endless:
            batches = [b for _ in range(20) for b in shuffle_batches(list(batch_sampler))]
    else:
        batches = batch_sampler
        if endless:
            batches = [b for _ in range(20) for b in batches]
    
    num_workers = dataset.num_workers
    return torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collater, num_workers=num_workers,
        batch_sampler=batches, pin_memory=False)


# In[19]:


def main():
    event2word_dict, word2event_dict = pickle.load(open(f"{binary_dir}/music_dict.pkl", 'rb'))
    batch_size = hparams['batch_size']

    test_dataset = L2MDataset('test', event2word_dict, hparams, shuffle=True)
    test_dataloader = build_dataloader(dataset=test_dataset, shuffle=True, batch_size=hparams['batch_size'], endless=True)

    print("length of train_dataloader", len(test_dataloader))
    
    # Test embedding
    for idx, item in enumerate(tqdm(test_dataloader)):
        enc_inputs = {k: item[f'src_{k}'] for k in src_keys}
        dec_inputs = {k: item[f'tgt_{k}'] for k in tgt_keys}
        template_embed = TemplateEmbedding(event2word_dict=event2word_dict, d_embed=hparams['hidden_size'])
        melody_embed = MelodyEmbedding(event2word_dict=event2word_dict, d_embed=hparams['hidden_size'])
        enc_emb = template_embed(**enc_inputs)
        dec_emb = melody_embed(**dec_inputs)
        print(enc_emb.shape, dec_emb.shape)


# In[20]:


# main()


# In[ ]:




