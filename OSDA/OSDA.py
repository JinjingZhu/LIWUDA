from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import ResNet as modelss
import time
import torch
import torch.nn as nn

import torch.nn.functional as F
import math
from torch.autograd import Variable
import random
from data import *
import matplotlib.pyplot as pltpython
import numpy as np
import os
import mmd
from data_provider import load_images


def MSE_Loss(source, target, label, y, model):
    with torch.no_grad():
        _, _, _, y_hat = model(source, target, label)
    return (F.mse_loss(y_hat.softmax(1), y.softmax(1).detach(), reduction='none').mean(1))

def bclass_loss(in_pred, in_label, weight, seen=False):
    if seen:
        size = len(in_pred)
        loss = F.cross_entropy(in_pred, in_label.long(), reduction='none', ignore_index=-1)
        weight = weight
        loss = torch.reshape(loss, (size, 1))
        weight_loss = torch.mul(loss, weight)
        cls_loss = torch.sum(weight_loss) / torch.sum(weight)
        return cls_loss
    else:
        loss = F.cross_entropy(in_pred, in_label.long(), reduction='none', ignore_index=-1)
        weight = 1 - weight
        size = len(in_pred)
        loss = torch.reshape(loss, (size, 1))
        weight_loss = torch.mul(loss, weight)
        cls_loss = torch.sum(weight_loss) / torch.sum(weight)
        return cls_loss


def class_loss(in_pred, weight):
    softmax = F.softmax(in_pred, dim=1)
    log_softmax = -F.log_softmax(in_pred, dim=1)
    entropy = softmax * log_softmax
    entropy_class = entropy.sum(dim=1)
    entropy_class = (torch.reshape(entropy_class, (batch_size, 1))).detach()
    cls_loss = torch.sum(torch.mul(entropy_class, weight)) / torch.sum(weight)
    return cls_loss


def weight_entropy(weight):
    weight_p = torch.mul(weight, torch.log(weight))
    weight_1_p = torch.mul((1 - weight), torch.log(1 - weight))
    loss = -torch.sum(weight_p) - torch.sum(weight_1_p)
    return loss

def entropy(logit):
    soft_logit = F.softmax(logit, dim=1)
    return -torch.sum(soft_logit* F.log_softmax(logit,dim=1), dim=1)

def train(epoch, model, wnet):
    LEARNING_RATE = 0.01 / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)  ###0.0001
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    optimizer = torch.optim.SGD([
        {'params': model.feature_layers.parameters()},
        {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=1e-3)  # e-3
    optimizer_wnet = torch.optim.Adam(wnet.parameters(), lr=1e-3 / math.pow((1 + 10 * (epoch - 1) / epochs),
                                                                            0.75))
    iter_source = iter(source_train_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    len_target = int(len(target_train_loader) / 2)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
    for i in range(0, num_iter):
        data_source, label_source = iter_source.next()
        data_source, label_source = Variable(data_source).cuda(), Variable(label_source).cuda()
        if i % len_target == 0:
            iter_target = iter(target_train_loader)
        data_target, label_target = iter_target.next()
        data_target, label_target = Variable(data_target).cuda(), Variable(
            label_target).cuda()

        model.train()
        wnet.train()
        source_feature, target_feature, source_pred, target_pred = model(data_source, data_target, label_source)
        weight_target = wnet(target_feature)
        weight_source = wnet(source_feature)

        loss_cls_model = ce_loss(source_pred, label_source)
        sa_loss, target_total_loss, target_mean_loss = mmd.sa(source_feature, target_feature, weight_target)
        iot_loss = mmd.iot(source_feature,source_feature,weight_source)
        source_align_feature = mmd.align_source(source_feature, source_feature, label_source)
        target_entropy=0
        target_weight_loss1=0
        j = 0
        for i in range(batch_size):
            max_model, index_model = torch.max(torch.nn.functional.softmax(target_pred, dim=1)[i], 0)
            if max_model > 0.9 and weight_target[i]>threhold:
                target_entropy = wnet(target_feature[i])*ce_loss(target_pred[i].reshape(1, class_num), (
                            index_model * torch.ones(1).cuda()).long()) + target_entropy
                target_weight_loss1 = (1 - weight_target[i]) ** 2 + target_weight_loss1
                j+=1
        if j == 0:
            target_loss = 0
            target_weight_loss1 = 0
        else:
            target_loss= target_entropy / j
            target_weight_loss1 = target_weight_loss1 / j
###     make the weight of data as zero or one based on the value of entropy.
        first_ten = torch.sort(target_mean_loss.detach(), dim=0)[1]
        first_ten_feature = torch.index_select(target_feature, 0, first_ten.view(batch_size))
        first_weight = wnet(first_ten_feature)

        target_entropy = entropy(target_pred)
        first_two1 = torch.sort(target_entropy.detach(), dim=0)[1]
        first_ten_feature1 = torch.index_select(target_feature, 0, first_two1.view(batch_size))
        first_weight1 = wnet(first_ten_feature1)
####    make the weight of data from zero to one according the rank of the mean loss
        weight_loss = (torch.sum(torch.mul((fix_weight - first_weight), (fix_weight - first_weight))) +
                       torch.sum(torch.mul((fix_weight - first_weight1), (fix_weight - first_weight1)))) / 2
        loss = 2 * loss_cls_model# + lambd * (0.1 * target_total_loss + 0.05 * weight_loss + 0.2 * source_align_feature + 0.05 * target_weight_loss1) + target_loss#+lambd*(0.3 * sa_loss+0.05*iot_loss)
        optimizer.zero_grad()
        optimizer_wnet.zero_grad()
        loss.backward()
        optimizer_wnet.step()
        optimizer.step()


def test(model, wnet):
    with torch.no_grad():
        model.eval()
        wnet.eval()
        i = 0
        for im, label in target_test_loader:
            data = Variable(im, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
            target_feature1, target_feature2, label_target_pred1, label_target_pred2 = model(data, data, label)
            label = label.cpu().data.numpy()
            predict_index = np.argmax(label_target_pred1.cpu().data.numpy(), axis=-1).reshape(-1, 1)
            weight_pred = (wnet(target_feature2)).cpu().data.numpy()
            if i == 0:
                true_label = label
                predict = predict_index
                weight = weight_pred
            else:
                true_label = np.append(true_label, label)
                predict = np.append(predict, predict_index)
                weight = np.append(weight, weight_pred)
            i += 1
    y_true = true_label.flatten()
    y_pred = predict.flatten()
    weight_target = weight.flatten()
    hos_best,acc_star_best,unkn_known = mmd.predict(y_true,y_pred,weight_target)
    return unkn_known, acc_star_best, hos_best


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    cuda = torch.cuda.is_available()
    batch_size = 128
    threhold=0.3
    class_num = 25
    epochs = 500
    fix_weight = torch.zeros((batch_size, 1)).cuda()
    for i in range(batch_size):
        fix_weight[i, :] = 1 - (i / (batch_size - 1))
    model = modelss.LIWUDA(num_classes=class_num)### resnet-50
    wnet = modelss.WNET(256, 1).cuda()###weight net
    unknown_best = acc_os_star_best = hos_best = 0
    if cuda:
        model.cuda()
    time_start = time.time()
    for epoch in range(1, epochs + 1):
        source_train_loader = load_images("data/office_home/clipart_0-24_train_all.txt",
                                          batch_size=batch_size, is_cen=False)
        target_train_loader = load_images('data/office_home/RealWorld_list.txt',
                                          batch_size=batch_size, is_cen=False)
        target_test_loader = load_images('data/office_home/RealWorld_list.txt',
                                         batch_size=batch_size, is_train=False)
        len_source_loader = len(source_train_loader)
        len_target_loader = len(target_train_loader)
        len_source_dataset = len(source_train_loader.dataset)
        len_target_dataset = len(target_test_loader.dataset)
        train(epoch, model, wnet)
        unkn_known, acc_os_star, hos = test(model, wnet)
        if hos > hos_best:
            hos_best = hos
            unknown_best=unkn_known
            acc_os_star_best = acc_os_star
            # torch.save(model, 'model.pkl')
        end_time = time.time()
        print('hos{: .2f} unknown: {} acc_os_star{: .2f}\n'.format(hos_best, unknown_best, acc_os_star_best))