from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
from Config import *
import time
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
cuda = not no_cuda and torch.cuda.is_available()


def train(epoch, model):
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    if bottle_neck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    model.train()

    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    num_iter = len_source_loader
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(1, num_iter):
        data_source, label_source = iter_source.next()
        data_target, jing_label = iter_target.next()
        
        if i % (len_target_loader-1) == 0:
            iter_target = iter(target_train_loader)
        if cuda:
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)
        a = time.time()
        optimizer.zero_grad()
        label_source_pred, target_pred,loss_dis = model(data_source, data_target, label_source)
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        loss = loss_cls +lambd * loss_dis
        print("loss",loss_dis)
        loss.backward()
        optimizer.step()

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            s_output,_, t_output= model(data, data, target)
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target).item() # sum up batch loss
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len_target_dataset

    return correct


if __name__ == '__main__':
    from preprocess.data_provider import load_images
    model = models.LIWUDA(num_classes=class_num)
    correct = 0
    if cuda:
        model.cuda()
    time_start = time.time()
    source_loader = load_images('./data/RealWorld_list.txt', batch_size=128, is_cen=False)
    target_train_loader = load_images('./data/Product_list.txt', batch_size=128, is_cen=False)
    target_test_loader = load_images('./data/Product_list.txt', batch_size=128, is_train=False)

    len_source_dataset = len(source_loader.dataset)
    len_target_dataset = len(target_test_loader.dataset)
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct = test(model)
        if t_correct > correct:
            correct = t_correct
        print('max correct: {} max accuracy{: .2f}%\n'.format(
            correct, 100. * correct / len_target_dataset))
        end_time = time.time()
        print('cost time:', end_time - time_start)
        







