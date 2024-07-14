#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import ot


### calculate cosine_distance
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
## seperate and align
def sa(source, target, weight):###the input is the extracted features of source and target domain. another input "weight" is the weight of data, which is obtained by weight net
    import torch
    batch_size = source.size()[0]
    guassian = cosine_distance(source,target)
    cc = (torch.ones(batch_size,batch_size).cuda()-guassian)/2
    c = (torch.ones(batch_size,batch_size).cuda()+guassian)/2
    mean_weight = (weight/torch.sum(weight)).squeeze()
    b = 0.9999 * (weight/torch.sum(weight)).squeeze()
    a = ((1 - 0.0001) / batch_size) * torch.ones(batch_size).cuda()
	### calculate the loss of partial optimal transport
    gama = torch.Tensor(ot.partial.partial_wasserstein(a.detach().cpu().numpy(), b.detach().cpu().numpy(),
                                                       cc.detach().cpu().numpy())).cuda()
    loss = torch.mul(gama, cc).sum()
    known_matric = torch.mul((1-0.5*(1+torch.sign(cc-loss))),gama)##discrimate to align or seperate
    unknown_matric = (gama-known_matric)
    for i in range(batch_size):
        for j in range(batch_size):
            if unknown_matric[i,j]==0:
                pass
            else:
                unknown_matric[i,j] = torch.exp(-unknown_matric[i,j])
    loss_all = 5*torch.mul(known_matric,cc).sum()/torch.sum(known_matric)+torch.mul(unknown_matric,c).sum()/torch.sum(unknown_matric)##sa loss
    mean_loss = torch.mul(gama, cc).sum(dim=0)/b ## the cost of data transport from source to target
    weight_loss = torch.sum(torch.mul((torch.mul(gama, cc).sum(dim=0) / b).detach(),mean_weight))### the total loss from source to target
    return weight_loss,loss_all,mean_loss

#intro domain optimal transport
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

	### seperate data with different classes and align data with same class
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
#### label the data as shared or unshared based on the value of weight, then predict the shared data.
def predict(y_true,y_pred,weight_target):
    for i in range(len(y_pred)):
        if weight_target[i] > 0.5: ##the threshold for discrimating unshared class or shared class
            pass
        else:
            y_pred[i] = 25   ### the label of unshared class
    m = extended_confusion_matrix(y_true, y_pred, true_labels=list(range(25)) + list(range(25, 65)),
                                  pred_labels=range(26))
    cm = m
    cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
    acc_os_star = sum([cm[i][i] for i in range(25)]) / 25
    acc_os = (acc_os_star * 10 + sum([cm[i][25] for i in range(25, 65)]) / 26) / 26
    unkn = sum([cm[i][25] for i in range(25, 65)]) / 26
    hos = (2 * acc_os_star * unkn) / (acc_os_star + unkn)
    return hos, unkn,acc_os_star
#
#
