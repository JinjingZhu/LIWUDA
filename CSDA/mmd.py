#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import ot
def convert_to_onehot(sca_label, class_num=65):
    return np.eye(class_num)[sca_label]

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

def lwot(source, target,source_weight,target_weight):
    batch_size = source.size()[0]
    guassian = cosine_distance(source,target)
    cc = (torch.ones(batch_size,batch_size).cuda()-guassian)/2
    b = 0.9999 * source_weight
    a = 0.9999* target_weight
    gama = torch.Tensor(ot.partial.partial_wasserstein(a, b,
                                                       cc.detach().cpu().numpy())).cuda()
    loss = torch.mul(gama, cc).sum()
    return loss

def cal_loss(source,target, s_label, t_label, class_num=65):
    batch_size = s_label.size()[0]
    s_sca_label = s_label.cpu().data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum
    t_sca_label = t_label.cpu().data.max(1)[1].numpy()
    t_vec_label = t_label.cpu().data.numpy()
    t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum
    loss =0
    set_s = set(s_sca_label)
    set_t = set(t_sca_label)
    count = 0
    for i in range(class_num):
        if i in set_s and i in set_t:
            s_tvec = s_vec_label[:, i]
            t_tvec = t_vec_label[:, i]
            loss += lwot(source,target,s_tvec,t_tvec)
            count += 1
    length = count
    if length != 0:
        loss = loss/length
    else:
        loss=0
    return loss




