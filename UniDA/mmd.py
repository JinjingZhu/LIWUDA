#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import ot
def cosine_distance( source_hidden_features, target_hidden_features):
    n_s = source_hidden_features.shape[0]
    n_t = target_hidden_features.shape[0]
    temp_matrix = torch.mm(source_hidden_features, target_hidden_features.t())
    for i in range(n_s):
        vec = source_hidden_features[i]
        temp_matrix[i] /= torch.norm(vec, p=2)
    for j in range(n_t):
        vec = target_hidden_features[j]
        temp_matrix[:, j] /= torch.norm(vec, p=2)
    return temp_matrix

def sa(source, target, s_label, t_label, source_weight,target_weight):
    import torch
    batch_size = source.size()[0]
    guassian = cosine_distance(source,target)
    cc = (torch.ones(batch_size,batch_size).cuda()-guassian)/2
    c = (torch.ones(batch_size,batch_size).cuda()+guassian)/2
    source_weight = 0.9999 * (source_weight / torch.sum(source_weight)).squeeze()
    target_weight = 0.9999 * (target_weight / torch.sum(target_weight)).squeeze()
    gama = torch.Tensor(ot.partial.partial_wasserstein(source_weight.detach().cpu().numpy(), target_weight.detach().cpu().numpy(),
                                                       cc.detach().cpu().numpy())).cuda()
    loss = torch.mul(gama, cc).sum()
    known_matric = torch.mul((1-0.5*(1+torch.sign(cc-loss))),gama)
    unknown_matric = (gama-known_matric)
    for i in range(batch_size):
        for j in range(batch_size):
            if unknown_matric[i,j]==0:
                pass
            else:
                unknown_matric[i,j] = torch.exp(-unknown_matric[i,j])
    loss_all = 5*torch.mul(known_matric,cc).sum()/torch.sum(known_matric)+torch.mul(unknown_matric,c).sum()/torch.sum(unknown_matric)
    target_mean_loss = torch.mul(gama, cc).sum(dim=0)/target_weight
    source_mean_loss = torch.mul(gama,cc).sum(dim=1)/source_weight
    weight_loss = torch.sum(torch.mul((torch.mul(gama, cc).sum(dim=0) / target_weight).detach(),target_weight))\
                  + torch.sum(torch.mul((torch.mul(gama, cc).sum(dim=1) / source_weight.detach()).detach(), source_weight))
    return weight_loss,loss_all,source_mean_loss,target_mean_loss

def align_source(source, target, s_label):
    batch_size = source.size()[0]
    guassian = cosine_distance(source,target)
    distance = (torch.ones(batch_size,batch_size).cuda() - guassian) / 2
    distance2 = (torch.ones(batch_size,batch_size).cuda() + guassian) / 2
    same_matric = torch.ones((batch_size,batch_size)).cuda()
    dif_matric = torch.ones((batch_size,batch_size)).cuda()
    for i in range(batch_size):
        for j in range(batch_size):
            if s_label[i]==s_label[j]:
                dif_matric[i,j]=0
            else:
                same_matric[i,j]=0
    source_distance = torch.mul(distance,same_matric).sum()/torch.sum(same_matric)+torch.mul(distance2,dif_matric).sum()/torch.sum(dif_matric)
    return source_distance

def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x : i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x : i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix

def predict(y_true, y_pred,weight_target):
    for i in range(len(y_pred)):
        if weight_target[i] > 0.5:
            pass
        else:
            y_pred[i] = 15
    m = extended_confusion_matrix(y_true, y_pred, true_labels=list(range(15)) + list(range(15, 65)),
                                  pred_labels=range(16))
    cm = m
    cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(10)]) / 10
    acc_os = (acc_os_star * 10 + sum([cm[i][15] for i in range(15, 65)]) / 50) / 11
    unkn = sum([cm[i][15] for i in range(15, 65)]) / 50
    hos = (2 * acc_os_star * unkn) / (acc_os_star + unkn)
    return hos,acc_os_star