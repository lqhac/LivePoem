#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pickle
import random
import subprocess
import torch.cuda
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.earlystopping.protocols import EarlyStopping
from test_dataloder import *
import datetime
from utils.get_time import get_time
import gc
from tqdm import tqdm
from utils.warmup import *
import torch.nn.functional as F
from transformermodel import *
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
    'hidden_size': 256,
    'n_layers': 6,
    'n_head': 8,
    'pretrain': '',
    'lr': 1.0e-5,
    'optimizer_adam_beta1': 0.9,
    'optimizer_adam_beta2': 0.98,
    'weight_decay': 0.1,
    'patience': 5,
    'warmup': 2500,
    'lr': 1.0e-5,
    'checkpoint_dir': '/home/qihao/CS6207/checkpoints',
    'drop_prob': 0.2,
    'total_epoch': 1000,
    'infer_batch_size': 1,
    'temperature': 1.3,
    'topk': 5,
    'prompt_step': 1,
    'infer_max_step': 1024,
    'output_dir': "/home/qihao/CS6207/output_melody",
    'num_heads': 4,
    'enc_layers': 4, 
    'dec_layers': 4, 
    'enc_ffn_kernel_size': 1,
    'dec_ffn_kernel_size': 1,
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


# In[13]:


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

# In[15]:


def train_l2m():
    ## train melody to lyric generation
    gc.collect()
    torch.cuda.empty_cache()
    
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    valid_dataset = L2MDataset('valid', event2word_dict, hparams, shuffle=False)

    train_loader = build_dataloader(dataset=train_dataset, shuffle=True, batch_size=hparams['batch_size'], endless=False)
    val_loader = build_dataloader(dataset=valid_dataset, shuffle=False, batch_size=hparams['batch_size'], endless=False)
    
    print(len(train_loader))
    
    # print(f"foundation model pth: {hparams['custom_model_dir']}")
    
    def tensor_check_fn(key, param, input_param, error_msgs):
        if param.shape != input_param.shape:
            return False
        return True
    
    model = MusicTransformer(event2word_dict=event2word_dict, 
                             word2event_dict=word2event_dict, 
                             hidden_size=hparams['hidden_size'], 
                             num_heads=hparams['num_heads'],
                             enc_layers=hparams['enc_layers'], 
                             dec_layers=hparams['dec_layers'], 
                             dropout=hparams['drop_prob'], 
                             enc_ffn_kernel_size=hparams['enc_ffn_kernel_size'],
                             dec_ffn_kernel_size=hparams['dec_ffn_kernel_size'],
                            ).to(device)
    
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

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=hparams['warmup'], num_training_steps=-1
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


# In[16]:


train_l2m()


# In[ ]:





# In[ ]:




