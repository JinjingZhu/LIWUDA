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
def MSE_Loss(source,target, label, y, model):
        with torch.no_grad():
            _,_,_,y_hat= model(source,target,label)
        return (F.mse_loss(y_hat.softmax(1),y.softmax(1).detach(), reduction='none').mean(1))
def entropy(logit):
    soft_logit = F.softmax(logit, dim=1)
    return -torch.sum(soft_logit* F.log_softmax(logit,dim=1), dim=1)
def train(epoch, model,wnet,unshare_list):
    LEARNING_RATE = 0.01 / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE))
    optimizer = torch.optim.SGD([
        {'params': model.feature_layers.parameters()},
        {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
    ], lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=5e-4)
    optimizer_wnet = torch.optim.Adam(wnet.parameters(),lr=1e-3/ math.pow((1 + 10 * (epoch - 1) / epochs), 0.75))
    iter_source= iter(source_train_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    len_target = int(len(target_train_loader)/2)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
    for i in range(0,num_iter):
        data_source, label_source = iter_source.next()
        data_source, label_source= Variable(data_source).cuda(), Variable(label_source).cuda()
        if i % len_target == 0:
            iter_target = iter(target_train_loader)
        data_target, label_target= iter_target.next()
        data_target, label_target = Variable(data_target).cuda(), Variable(
            label_target).cuda()

        model.train()
        wnet.train()
        source_feature, target_feature,source_pred,  target_pred = model(data_source, data_target, label_source)### obtain extracted features and predictions from resnet50
        weight_target = wnet(target_feature)
        weight_source = wnet(source_feature)
        loss_cls_model = ce_loss(source_pred,label_source)
        sa_loss, total_loss, source_mean_loss,target_mean_loss= mmd.sa(source_feature, target_feature, label_source,
                           torch.nn.functional.softmax(target_pred, dim=1), weight_source ,weight_target)
        source_align_feature = mmd.align_source(source_feature, source_feature,label_source)
        first_ten = torch.sort(target_mean_loss.detach(), dim=0)[1]
        first_ten_feature = torch.index_select(target_feature, 0, first_ten.view(batch_size))
        first_weight = wnet(first_ten_feature)

        target_entropy = entropy(target_pred)
        first_two1 = torch.sort(target_entropy.detach(), dim=0)[1]
        first_ten_feature1 = torch.index_select(target_feature, 0, first_two1.view(batch_size))
        first_weight1 = wnet(first_ten_feature1)

        first_ten_source = torch.sort(source_mean_loss.detach(), dim=0)[1]
        first_ten_feature_source = torch.index_select(source_feature, 0, first_ten_source.view(batch_size))
        first_weight_source = wnet(first_ten_feature_source)

        FL_weight_loss = (torch.sum(torch.mul((fix_weight - first_weight), (fix_weight - first_weight))) +
                          torch.sum(torch.mul((fix_weight - first_weight1), (fix_weight - first_weight1))) +
                          torch.sum(
                              torch.mul((fix_weight - first_weight_source), (fix_weight - first_weight_source)))) / 3
        target_entropy=0
        target_weight_loss1 = 0
        j = 0
        for i in range(batch_size):
            max_model, index_model = torch.max(torch.nn.functional.softmax(target_pred, dim=1)[i], 0)
            if max_model > 0.9 and index_model not in unshare_list:
                target_entropy = ce_loss(target_pred[i].reshape(1, class_num), (
                            index_model * torch.ones(1).cuda()).long()) + target_entropy
                target_weight_loss1 = (1 - weight_target[i]) ** 2 + target_weight_loss1
                j+=1
        if j == 0:
            target_loss = 0
            target_weight_loss1 = 0
        else:
            target_loss= target_entropy / j
            target_weight_loss1 = target_weight_loss1 / j

        loss = 2 * loss_cls_model + lambd * (0.1 * total_loss + 0.3 * sa_loss + 0.05 * FL_weight_loss
                                             + 0.2 * source_align_feature +0.05*target_weight_loss1) + target_loss
        optimizer.zero_grad()
        optimizer_wnet.zero_grad()
        loss.backward()
        optimizer_wnet.step()
        optimizer.step()

def test(model,wnet):
    model.eval()
    wnet.eval()
    with torch.no_grad():
        ###source
        i = 0
        for im1, label1 in source_test_loader:
            data1 = Variable(im1, volatile=True).cuda()
            label1 = Variable(label1, volatile=True).cuda()
            source_feature, _, _,_ = model(data1, data1, label1)
            label1 = label1.cpu().data.numpy()
            weight_pred1 = wnet(source_feature)
            if i == 0:
                true_label1 = label1
                weight1 = weight_pred1.cpu().data.numpy()
            else:
                true_label1 = np.append(true_label1, label1)
                weight1 = np.append(weight1, weight_pred1.cpu().data.numpy())
            i += 1
        weight_target1 = weight1.flatten()
        source_true_label = true_label1.flatten()

        n = 0
        unshare_list=[]
        share_list=[]
        list_source = list(source_true_label)
        for i in range(class_num):
            mean_weight = np.mean(weight_target1[n:n + list_source.count(i)])
            if mean_weight<0:
                unshare_list.append(i)
            else:
                share_list.append(i)
            n += list_source.count(i)
        l_target=0
        ##target
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            target_feature,_,s_output,_ = model(data, data,target)
            pred = s_output.data.max(1)[1]
            pred = pred.cpu().data.numpy()
            label = target.cpu().data.numpy()
            weight_pred = (wnet(target_feature)).cpu().data.numpy()
            if l_target==0:
                true_label = label
                for i in range(pred.shape[0]):
                    if pred[i] in unshare_list:
                        label_index = torch.sort(s_output[i].detach(), dim=0, descending=True)[1]
                        label_sort = torch.index_select(class_label, 0, label_index.view(class_num))
                        print(label_sort)
                        for j in range(class_num):
                            if label_sort[j] in share_list:
                                pred[i] = label_sort[j]
                                break
                predict = pred
                weight = weight_pred
            else:
                true_label = np.append(true_label, label)
                for i in range(pred.shape[0]):
                    if pred[i] in unshare_list:
                        label_index = torch.sort(s_output[i].detach(), dim=0, descending=True)[1]
                        label_sort = torch.index_select(class_label, 0, label_index.view(class_num))
                        print(label_sort)
                        for j in range(class_num):
                            if label_sort[j] in share_list:
                                pred[i] = label_sort[j]
                                break
                predict = np.append(predict, pred)
                weight = np.append(weight, weight_pred)
            l_target+=1
        y_true = true_label.flatten()
        y_pred = predict.flatten()
        weight_target = weight.flatten()
        hos_best,acc_star_best = mmd.predict(y_true,y_pred,weight_target)
        return acc_star_best, hos_best,unshare_list

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    cuda = torch.cuda.is_available()
    batch_size = 128
    fix_weight = torch.zeros((batch_size, 1)).cuda()
    for i in range(batch_size):
        fix_weight[i, :] = 1 - (i / (batch_size - 1))
    n_classes = class_num = 15
    n_classes_target = 16
    epochs = 500
    hos_best_acc = 0
    acc_os_star_acc =0
    model = modelss.LIWUDA(num_classes=class_num)
    wnet = modelss.WNET(256, 1).cuda()
    class_label = torch.ones(class_num).cuda()
    for i in range(class_num):
        class_label[i]=i
    acc_os= 0
    acc_os_star = 0
    if cuda:
        model.cuda()
    time_start=time.time()
    unshare_list=[]
    for epoch in range(1, epochs + 1):
        source_train_loader = load_images("data/office_home/source_RealWorld_unida.txt",
                                          batch_size=batch_size, is_cen=False)
        target_train_loader = load_images('data/office_home/target_Art_unida.txt',
                                          batch_size=batch_size, is_cen=False)
        target_test_loader = load_images('data/office_home/target_Art_unida.txt',
                                         batch_size=batch_size, is_train=False)
        source_test_loader = load_images("data/office_home/source_RealWorld_unida.txt",
                                         batch_size=batch_size, is_train=False)
        len_source_loader = len(source_train_loader)
        len_target_loader = len(target_train_loader)
        len_source_dataset = len(source_train_loader.dataset)
        len_target_dataset = len(target_test_loader.dataset)
        train(epoch, model,wnet,unshare_list)
        t_acc_os_star, hos,unshare_list = test(model, wnet)
        if hos > hos_best_acc:
            hos_best_acc = hos
            acc_os_star_acc = t_acc_os_star
            # torch.save(model, 'model.pkl')
        end_time = time.time()
        print('hos{: .2f} acc_os_star{: .2f}\n'.format(hos_best_acc, acc_os_star_acc))
        print('cost time:', end_time - time_start)
