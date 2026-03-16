# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import time
from datetime import timedelta
from collections import defaultdict
import json
import os
import traceback


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, dataset):
    logging = config.logging
    start_time = time.time()
    parameters = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
    total_batch = 0  
    val_best_auc = -np.inf
    val_best_loss = np.inf
    last_improve = 0  
    flag1 = False
    no_improve_cnt = 0
    model.train()
    writer = SummaryWriter(log_dir=config.summary_dir)
    for epoch in range(config.num_epochs):
        logging.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        logging.info('start init data...')
        dataset.init_per_epoch()
        logging.info('init data done.')
        for batch_datas in dataset.train_iter:
            batch_Y = batch_datas[-1]
            out, loss = model(batch_datas)
            if config.multi_gpu:
                loss = loss.sum()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            if total_batch > 0 and total_batch % config.batches_per_check == 0:
                logging.info('step: %s, start eval...', total_batch)
                true = batch_Y.data.cpu()
                predict = torch.max(out.data, 1)[1].cpu()
                train_loss = loss.item()
                train_acc = metrics.accuracy_score(true, predict)
                val_acc, val_auc, val_loss = evaluate(config, model, dataset.val_iter)
                if val_best_auc < val_auc:
                    val_best_auc = val_auc
                    save_checkpoint(model, config)
                    # torch.save(model.state_dict(), config.save_path)
                    improve = '***'
                    last_improve = total_batch
                    no_improve_cnt = 0
                else:
                    improve = ''
                    no_improve_cnt += 1
                time_dif = get_time_dif(start_time)
                lr = optimizer.param_groups[0]['lr']
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.2%},  Val Auc: {5:>6.4}, Lr: {6:>8.6}, Time: {7} {8}'
                msg = msg.format(total_batch, train_loss, train_acc, val_loss, val_acc, val_auc, lr, time_dif, improve)
                logging.info(msg)
                model.train()
                writer.add_scalar('train_loss', train_loss, total_batch)
                writer.add_scalar('train_acc', train_acc, total_batch)
                writer.add_scalar('val_loss', val_loss, total_batch)
                writer.add_scalar('val_auc', val_auc, total_batch)
                writer.add_scalar('val_acc', val_acc, total_batch)
                writer.add_scalar('lr', lr, total_batch)
                if no_improve_cnt > 0 and no_improve_cnt % 3 == 0 and lr > 3e-5:
                    lr_scheduler.step()
            if total_batch - last_improve > config.require_improvement:
                logging.info("No optimization for a long time, auto-stopping...")
                flag1 = True
                break
            total_batch += 1
        if flag1:
            break
        test(config, model, dataset.test_iter)  # 每个epoch eval一次
    writer.close()
    test(config, model, dataset.test_iter)
    
def save_checkpoint(model, config):
    ckpt_dict = {
        'model': model.module.state_dict() if config.multi_gpu else model.state_dict(),
        'config': config.get_parameters()
    }
    torch.save(ckpt_dict, config.save_path)

def load_checkpoint(model, config):
    ckpt_dict = torch.load(config.save_path)
    if config.multi_gpu:
        model.module.load_state_dict(ckpt_dict['model'])
    else:
        model.load_state_dict(ckpt_dict['model'])

def test(config, model, test_iter):
    logging = config.logging
    # test
    # model.load_state_dict(torch.load(config.save_path))
    if not os.path.exists(config.save_path):
        logging.info('model path does not exist...')
        return
    load_checkpoint(model, config)
    model.eval()
    start_time = time.time()
    test_acc, test_auc, loss_total, test_loss, test_report, test_confusion, pred_prob_all, labels_all = evaluate(config, model, test_iter, test=True)
    # compute new_user/old_user/new_news/old_news auc
    try:
        new_user_pred = [x for i, x in enumerate(pred_prob_all) if i in config.new_user_index]
        new_user_label = [x for i, x in enumerate(labels_all) if i in config.new_user_index]
        new_user_auc = metrics.roc_auc_score(new_user_label, new_user_pred)

        old_user_pred = [x for i, x in enumerate(pred_prob_all) if i not in config.new_user_index]
        old_user_label = [x for i, x in enumerate(labels_all) if i not in config.new_user_index]
        old_user_auc = metrics.roc_auc_score(old_user_label, old_user_pred)
        
        new_news_pred = [x for i, x in enumerate(pred_prob_all) if i in config.new_news_index]
        new_news_label = [x for i, x in enumerate(labels_all) if i in config.new_news_index]
        new_news_auc = metrics.roc_auc_score(new_news_label, new_news_pred)
        
        old_news_pred = [x for i, x in enumerate(pred_prob_all) if i not in config.new_news_index]
        old_news_label = [x for i, x in enumerate(labels_all) if i not in config.new_news_index]
        old_news_auc = metrics.roc_auc_score(old_news_label, old_news_pred)   
        
        rig = 1 - loss_total / config.entropy  # 信息增益
        gauc = cal_gauc(config.test_user_id, pred_prob_all, labels_all)
        
        pred_label = {
            'pred': [str(round(x, 3)) for x in pred_prob_all],
            'label': list(map(str, labels_all))
        }
        msg = 'Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}, Test Auc: {2:>6.4}, new_user Auc: {3:>6.4}, old_user Auc: {4:>6.4}, new_news Auc: {5:>6.4}, old_news Auc: {6:>6.4}, total_loss: {7:>7.4}, rig: {8:>8.4}, gauc: {9:>9.4}'
        msg = msg.format(test_loss, test_acc, test_auc, new_user_auc, old_user_auc, new_news_auc, old_news_auc, loss_total, rig, gauc)
        logging.info(msg.format())
        logging.info("Precision, Recall and F1-Score...")
        logging.info(test_report)
        logging.info("Confusion Matrix...")
        logging.info(test_confusion)
        time_dif = get_time_dif(start_time)
        logging.info("Time usage: %s", time_dif)
        model.train()
        with open(config.pred_label_save_path, 'w') as f:
            json.dump(pred_label, f)
    except Exception as e:
        traceback.print_exc()


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    pred_prob_all = []
    with torch.no_grad():
        for batch_datas in data_iter:
            batch_Y = batch_datas[-1]
            out, loss = model(batch_datas)
            if config.multi_gpu:
                loss = loss.sum()
            # loss = F.cross_entropy(out, batch_Y)
            loss_total += loss.item()
            labels = batch_Y.data.cpu().numpy()
            predict = torch.max(out.data, 1)[1].cpu().numpy()
            predict_all.append(predict)
            labels_all.append(labels)
            pred_prob = F.softmax(out, dim=1)[:, 1].data.cpu().numpy()
            pred_prob_all.append(pred_prob)
    predict_all = np.concatenate(predict_all)
    labels_all = np.concatenate(labels_all)
    pred_prob_all = np.concatenate(pred_prob_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    try:
        auc = metrics.roc_auc_score(labels_all, pred_prob_all)
    except Exception as e:
        import pdb; pdb.set_trace()
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, auc, loss_total, loss_total / len(data_iter), report, confusion, pred_prob_all, labels_all
    return acc, auc, loss_total / len(data_iter)

def cal_gauc(user_ids, preds, labels):
    assert len(user_ids) == len(preds) and len(preds) == len(labels)
    uid_pred_dict = defaultdict(list)
    uid_label_dict = defaultdict(list)
    for uid, pred, label in zip(user_ids, preds, labels):
        uid_pred_dict[uid].append(pred)
        uid_label_dict[uid].append(label)
    all_auc = 0
    total = 0
    for uid in uid_pred_dict.keys():
        if len(set(uid_label_dict[uid])) < 2:
            continue
        u_auc = metrics.roc_auc_score(uid_label_dict[uid], uid_pred_dict[uid])
        n_u = len(uid_label_dict[uid])
        all_auc += u_auc * n_u
        total += n_u  
    return all_auc / total

