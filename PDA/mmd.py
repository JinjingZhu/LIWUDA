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

def sa(source, target, s_label, t_label, weight):
    import torch
    batch_size = source.size()[0]
    guassian = cosine_distance(source,target)
    cc = (torch.ones(batch_size,batch_size).cuda()-guassian)/2
    c = (torch.ones(batch_size,batch_size).cuda()+guassian)/2
    mean_weight = (weight/torch.sum(weight)).squeeze()
    b = 0.9999 * (weight/torch.sum(weight)).squeeze()
    a = ((1 - 0.0001) / batch_size) * torch.ones(batch_size).cuda()
    gama = torch.Tensor(ot.partial.partial_wasserstein(a.detach().cpu().numpy(), b.detach().cpu().numpy(),
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
    mean_loss = torch.mul(gama, cc).sum(dim=0)/b
    weight_loss = torch.sum(torch.mul((torch.mul(gama, cc).sum(dim=0) / b).detach(),mean_weight))
    return weight_loss,loss_all,mean_loss

def iot(source, target, weight):
    import torch
    batch_size = source.size()[0]
    guassian = cosine_distance(source, target)
    cc = (torch.ones(batch_size, batch_size).cuda() - guassian) / 2
    b = 0.9999 * (weight / torch.sum(weight)).squeeze()
    a = ((1 - 0.0001) / batch_size) * torch.ones(batch_size).cuda()
    gama = torch.Tensor(ot.partial.partial_wasserstein(a.detach().cpu().numpy(), b.detach().cpu().numpy(),
                                                       cc.detach().cpu().numpy())).cuda()
    loss= torch.sum(torch.mul(((torch.mul(gama, cc).sum(dim=0)).detach()/ b.detach()), b))
    return loss

def align_source(source, s_label):
    batch_size = source.size()[0]
    guassian = cosine_distance(source,source)
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


