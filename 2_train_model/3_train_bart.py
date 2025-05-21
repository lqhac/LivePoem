#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os, sys
os.environ['PYTHONPATH'] = '/home/qihao/CS6207'
sys.path.append('/home/qihao/CS6207')
import pickle
import random
import subprocess
import torch.cuda
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.earlystopping.protocols import EarlyStopping
from utils.test_dataloder import *
import datetime
from utils.get_time import get_time
import gc
from tqdm import tqdm
from utils.warmup import *
import torch.nn.functional as F
from models.bartmodel import Bart
from transformers import get_linear_schedule_with_warmup

# In[10]:


src_keys = ['strength', 'length', 'phrase']
tgt_keys = ['bar', 'pos', 'token', 'dur', 'phrase']

binary_dir = '/home/qihao/CS6207/binary'
words_dir = '/home/qihao/CS6207/binary/words'
hparams = {
    'batch_size': 4,
    'word_data_dir': '/home/qihao/CS6207/binary/words',
    'sentence_maxlen': 512,
    'hidden_size': 768,
    'n_layers': 6,
    'n_head': 8,
    # 'pretrain': '/home/qihao/CS6207/checkpoints/checkpoint_20240331:181343_lr_5e-05/best.pt',
    # 'pretrain': '/home/qihao/CS6207/checkpoints/checkpoint_20240331:191608_lr_5e-05/best.pt',
    # 'pretrain': '/home/qihao/CS6207/checkpoints/checkpoint_20240403:194534_lr_5e-05/best.pt',
    'pretrain': '/home/qihao/CS6207/checkpoints/checkpoint_20240405:195400_lr_5e-05/best.pt',
    # 'pretrain': '',
    'optimizer_adam_beta1': 0.9,
    'optimizer_adam_beta2': 0.98,
    'weight_decay': 0.1,
    'patience': 10,
    'warmup': 3000,
    'lr': 5.0e-5,
    'checkpoint_dir': '/home/qihao/CS6207/checkpoints',
    'drop_prob': 0.2,
    'total_epoch': 1000,
    'num_epochs': 30,
}


# In[11]:


def set_seed(seed=1234):  # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[12]:


def xe_loss(outputs, targets):
    outputs = outputs.transpose(1, 2)
    return F.cross_entropy(outputs, targets, ignore_index=0, reduction='mean')


# In[21]:


def train(train_loader, model, optimizer, scheduler, epoch, total_epoch):
    # define the format of tqdm
    with tqdm(total=len(train_loader), ncols=150, position=0, leave=True) as _tqdm:  # 总长度是data的长度
        _tqdm.set_description('training epoch: {}/{}'.format(epoch + 1, total_epoch))  # 设置前缀更新信息

        # Model Train
        model.train()
        running_loss = 0.0
        train_loss = []

        for idx, data in enumerate(train_loader):
            # prompt_index = list(data[f'tgt_word'].numpy()).index(50268)
            enc_inputs = {k: data[f'src_{k}'].to(device) for k in src_keys}
            dec_inputs = {k: data[f'tgt_{k}'].to(device) for k in tgt_keys}
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            out = model(enc_inputs, dec_inputs)
            
            tgt = (data['tgt_token'].to(device))[:, 1:]
            loss = xe_loss(out[:, :-1], tgt)
            
            # 3) total loss
            total_loss = loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(total_loss.item())
            running_loss += total_loss.item()
            

            _tqdm.set_postfix(
                loss="{:.3f}".format(total_loss))
            
            _tqdm.update(2)

    train_loss_avg = np.mean(train_loss)
    
    return train_loss_avg


# In[22]:


def valid(valid_loader, model, epoch, total_epoch):
    # define the format of tqdm
    with tqdm(total=len(valid_loader), ncols=150) as _tqdm:  # 总长度是data的长度
        _tqdm.set_description('validation epoch: {}/{}'.format(epoch + 1, total_epoch))  # 设置前缀更新信息

        model.eval()  # switch to valid mode
        running_loss = 0.0
        val_loss = []
        
        with torch.no_grad():
            for idx, data in enumerate((valid_loader)):
                try:
                    enc_inputs = {k: data[f'src_{k}'].to(device) for k in src_keys}
                    dec_inputs = {k: data[f'tgt_{k}'].to(device) for k in tgt_keys}

                    out = model(enc_inputs, dec_inputs)

                    tgt = (data['tgt_token'].to(device))[:, 1:]
                    loss = xe_loss(out[:, :-1], tgt)


                    # 3) total loss
                    total_loss = loss
                    val_loss.append(total_loss.item())
                    running_loss += total_loss.item()

                    _tqdm.set_postfix(
                        loss="{:.3f}".format(total_loss))

                    _tqdm.update(2)
                    
                except Exception as e:
                    print(data)
                    print("Bad Data Item!")
                    print(e)
                    break
            
    val_loss_avg = np.mean(val_loss)

    return val_loss_avg


# In[23]:


def train_l2m():
    ## train melody to lyric generation
    gc.collect()
    torch.cuda.empty_cache()
    
    global device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # args
    set_seed()
    # set_hparams()
    event2word_dict, word2event_dict = pickle.load(open(f"{binary_dir}/music_dict.pkl", 'rb'))

    # tensorboard logger
    cur_time = get_time()
    # tensorboard_dir = hparams['tensorboard']
    # train_log_dir = f'{tensorboard_dir}/{cur_time}/train'
    # valid_log_dir = f'{tensorboard_dir}/{cur_time}/valid'
    # train_writer = SummaryWriter(log_dir=train_log_dir)
    # valid_writer = SummaryWriter(log_dir=valid_log_dir)

    # ------------
    # train
    # ------------
    # load data
    train_dataset = L2MDataset('train', event2word_dict, hparams, shuffle=True)
    valid_dataset = L2MDataset('test', event2word_dict, hparams, shuffle=False)

    train_loader = build_dataloader(dataset=train_dataset, shuffle=True, batch_size=hparams['batch_size'], endless=False)
    val_loader = build_dataloader(dataset=valid_dataset, shuffle=False, batch_size=hparams['batch_size'], endless=False)
    
    print(len(train_loader))
    
    
    def tensor_check_fn(key, param, input_param, error_msgs):
        if param.shape != input_param.shape:
            return False
        return True
    
    model = Bart(event2word_dict=event2word_dict, 
                 word2event_dict=word2event_dict, 
                 model_pth='',
                 hidden_size=hparams['hidden_size'], 
                 num_layers=hparams['n_layers'], 
                 num_heads=hparams['n_head'], 
                 dropout=hparams['drop_prob'],).to(device)
    
    pre_trained_path = hparams['pretrain']
    if pre_trained_path != '':
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(pre_trained_path)
        new_state_dict={k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        # model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(torch.load(pre_trained_path), strict=False, tensor_check_fn=tensor_check_fn)
        model.load_state_dict(new_state_dict, strict=False)
        print(">>> Load pretrained model successfully")
        
    ## warm up
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
        weight_decay=hparams['weight_decay'])

    num_training_steps = int(len(train_loader) * hparams['num_epochs'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=hparams['warmup'], num_training_steps=num_training_steps
    )

    """
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    """

    # training conditions (for naming the ckpt)
    lr = hparams['lr']

    # early stop: initialize the early_stopping object
    # checkpointpath = f"{hparams['checkpoint_dir']}/Cond_{cond}_GPT2_{cur_time}_lr{lr}"
    checkpointpath = f"{hparams['checkpoint_dir']}/checkpoint_{cur_time}_lr_{lr}"
    if not os.path.exists(checkpointpath):
        os.mkdir(checkpointpath)
    early_stopping = EarlyStopping(patience=hparams['patience'], verbose=True,
                                   path=f"{checkpointpath}/early_stopping_checkpoint.pt")
    

    # -------- Train & Validation -------- #
    min_valid_running_loss = 1000000  # inf
    total_epoch = hparams['total_epoch']
    with tqdm(total=total_epoch) as _tqdm:
        for epoch in range(total_epoch):
            # Train
            train_running_loss = train(train_loader, model, optimizer, scheduler, epoch, total_epoch)
            # train_writer.add_scalars("train_epoch_loss", {"running": train_running_loss, 'word': train_word_loss}, epoch)

            # validation  
            valid_running_loss = valid(val_loader, model, epoch, total_epoch)
            # valid_writer.add_scalars("valid_epoch_loss", {"running": valid_running_loss, 'word': valid_word_loss}, epoch)

            # early stopping Check
            early_stopping(valid_running_loss, model, epoch)
            if early_stopping.early_stop == True:
                print("Validation Loss convergence， Train over")
                break

            # save the best checkpoint
            if valid_running_loss < min_valid_running_loss:
                min_valid_running_loss = valid_running_loss
                torch.save(model.state_dict(), f"{checkpointpath}/best.pt")
            print(f"Training Runinng Loss = {train_running_loss}, Validation Running Loss = {min_valid_running_loss}")  
            _tqdm.update(2)


# In[24]:


train_l2m()


# In[ ]:




