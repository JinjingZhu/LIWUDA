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

def bclass_loss(in_pred,in_label, weight,seen=False):
    if seen:
        size = len(in_pred)
        loss = F.cross_entropy(in_pred, in_label.long(), reduction='none', ignore_index=-1)
        weight = weight
        loss = torch.reshape(loss,(size,1))
        weight_loss = torch.mul(loss,weight)
        cls_loss = torch.sum(weight_loss)/torch.sum(weight)
        return cls_loss
    else:
        loss = F.cross_entropy(in_pred, in_label.long(), reduction='none', ignore_index=-1)
        weight = 1-weight
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
    entropy_class = (torch.reshape(entropy_class,(batch_size,1))).detach()
    cls_loss = torch.sum(torch.mul(entropy_class,weight))/torch.sum(weight)
    return cls_loss

def weight_entropy(weight):
    weight_p = torch.mul(weight,torch.log(weight))
    weight_1_p = torch.mul((1-weight),torch.log(1-weight))
    loss = -torch.sum(weight_p)-torch.sum(weight_1_p)
    return loss

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
    len_target = len(target_train_loader)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
    for i in range(0,num_iter):
        _,_,data_source, label_source = iter_source.next()
        data_source, label_source= Variable(data_source).cuda(), Variable(label_source).cuda()
        if i % len_target == 0:
            iter_target = iter(target_train_loader)
        _,_,data_target, label_target= iter_target.next()
        data_target, label_target = Variable(data_target).cuda(), Variable(
            label_target).cuda()

        model.train()
        wnet.train()
        source_feature, target_feature,source_pred,  target_pred = model(data_source, data_target, label_source)
        weight_source = wnet(source_feature)
        weight_target = wnet(target_feature)
        loss_cls_model = ce_loss(source_pred,label_source)
        sa_loss, source_total_loss, source_mean_loss= mmd.sa(target_feature, source_feature, label_source,
                           torch.nn.functional.softmax(target_pred, dim=1), weight_source)
        source_align_feature = mmd.align_source(source_feature,label_source)
        iot_loss = mmd.iot(target_feature, target_feature, weight_target)
        target_entropy = 0
        j = 0
        for i in range(batch_size):
            max_model, index_model = torch.max(torch.nn.functional.softmax(target_pred, dim=1)[i], 0)
            if max_model > 0.9 and index_model not in unshare_list:
                target_entropy = ce_loss(target_pred[i].reshape(1, class_num), (
                        index_model * torch.ones(1).cuda()).long()) + target_entropy
                j += 1
        if j == 0:
            target_loss = 0
        else:
            target_loss = target_entropy / j


        first_ten_source = torch.sort(source_mean_loss.detach(), dim=0)[1]
        first_ten_feature_source = torch.index_select(source_feature, 0, first_ten_source.view(batch_size))
        first_weight_source = wnet(first_ten_feature_source)
        FL_weight_loss = torch.sum(torch.mul((fix_weight - first_weight_source), (fix_weight - first_weight_source)))
        loss = 2 * loss_cls_model+ lambd * (0.1 * source_total_loss + 0.3 * sa_loss+0.05*iot_loss
                                    + 0.05 * FL_weight_loss + 0.2 * source_align_feature) + target_loss
        optimizer.zero_grad()
        optimizer_wnet.zero_grad()
        loss.backward()
        optimizer_wnet.step()
        optimizer.step()
def test(model):
    model.eval()
    wnet.eval()
    correct=0
    test_loss=0
    pred_list = []
    with torch.no_grad():
        i = 0
        for im1, label1,_,_ in source_test_loader:
            data1 = Variable(im1, volatile=True).cuda()
            label1 = Variable(label1, volatile=True).cuda()
            target_feature3, target_feature4, label_target_pred3, label_target_pred4 = model(data1, data1, label1)
            label1 = label1.cpu().data.numpy()
            weight_pred1 = wnet(target_feature3)
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
        weight_list = []
        list_source = list(source_true_label)
        for i in range(class_num):
            mean_weight = np.mean(weight_target1[n:n + list_source.count(i)])
            weight_list.append(mean_weight)
            if mean_weight<0:
                unshare_list.append(i)
            else:
                share_list.append(i)
            n += list_source.count(i)
        for data, target,_,_ in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            _,_,s_output,_ = model(data, data,target)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target).item()
            pred = s_output.data.max(1)[1]
            for i in range(pred.shape[0]):
                if pred[i] in unshare_list:
                    label_index = torch.sort(s_output[i].detach(), dim=0, descending=True)[1]
                    label_sort = torch.index_select(class_label, 0, label_index.view(class_num))
                    print(label_sort)
                    for j in range(class_num):
                        if label_sort[j] in share_list:
                            pred[i] = label_sort[j]
                            break
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred_list.append(list(pred.cpu().data.numpy()))
        test_loss /= len_target_dataset
    return correct,unshare_list


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cuda = torch.cuda.is_available()
    batch_size = 128
    class_num = 65
    fix_weight = torch.zeros((batch_size, 1)).cuda()
    for i in range(batch_size):
        fix_weight[i, :] = 1 - (i / (batch_size - 1))
    class_label = torch.ones(class_num).cuda()
    for i in range(class_num):
        class_label[i]=i
    epochs = 500
    model = modelss.LIWUDA(num_classes=class_num)
    wnet = modelss.WNET(256, 1).cuda()
    acc_os= 0
    source_name = "webcam"  #
    target_name = "amazon"
    if cuda:
        model.cuda()
    time_start=time.time()
    unshare_list=[]
    source_name = "./data/office_home/Clipart_list.txt"
    target_name = './data/office_home/Art_25_list.txt'
    print(source_name,target_name)
    for epoch in range(1, epochs + 1):
        source_train_loader = load_images(source_name,
                                          batch_size=batch_size, is_cen=False)
        target_train_loader = load_images(target_name,
                                          batch_size=batch_size, is_cen=False)
        target_test_loader = load_images(target_name,
                                         batch_size=batch_size, is_train=False)
        source_test_loader = load_images(source_name,
                                          batch_size=batch_size, is_train=False)
        len_source_loader = len(source_train_loader)
        len_target_loader = len(target_train_loader)
        len_source_dataset = len(source_train_loader.dataset)
        len_target_dataset = len(target_test_loader.dataset)
        train(epoch, model,wnet,unshare_list)
        t_acc_os,unshare_list = test(model)
        if t_acc_os > acc_os:
            acc_os = t_acc_os
            torch.save(model, 'model.pkl')
        end_time = time.time()
        print("best_accuracy:",acc_os/ len_target_dataset)
        print('cost time:', end_time - time_start)