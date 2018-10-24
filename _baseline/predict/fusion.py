#coding=utf-8

############################################################################
#
# Copyright (c) 2018 ICT MCG Group, Inc. All Rights Reserved
#
###########################################################################
"""
Brief:
任务：单微博的谣言检测
方法：实现文本模态和视觉模态的融合
1. baseline:拼接
2. 每个post动态训练权重

Authors: qipeng(@ict.ac.cn)
Date:    2018/09/19 18:41:48
File:    fusion.py
"""
import sys

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, LSTM, Merge, Lambda, Input, merge, Flatten, Bidirectional, TimeDistributed, multiply
from keras.layers.core import Reshape, RepeatVector, Permute
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from attention_layer import AttentionLayer3
import tensorflow as tf
from keras import backend as K
import keras
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold,GroupKFold,StratifiedKFold

import time
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

import pickle

## PATH
model_path = './data/model_weights_'+ time.strftime("%y-%m-%d-%H-%M-%S", time.localtime()) +'.h5'
history_path = './data/train_logs_' + time.strftime("%y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
weights_path = './data/weights_' + time.strftime("%y-%m-%d-%H-%M-%S", time.localtime()) +'.txt'
score_path = './data/score_' + time.strftime("%y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'

## WEIGHTS
maxlen = 80
batch_size = 256
epoch_size = 1
embedding_dim = 32
hidden_num = 16

## FEATURES，注意两种特征要在一个scale上
# text feature
test_neg = np.load('./numpy_data/test_neg.npy')
test_pos = np.load('./numpy_data/test_pos.npy')
train_neg = np.load('./numpy_data/train_neg.npy')
train_pos = np.load('./numpy_data/train_pos.npy')

# visual feature
auxi_test_neg_vis = np.load('./features/vis_simplemodel/test_rumor_vis.npy')
auxi_test_pos_vis = np.load('./features/vis_simplemodel/test_nonrumor_vis.npy')
auxi_train_neg_vis = np.load('./features/vis_simplemodel/train_rumor_vis.npy')
auxi_train_pos_vis = np.load('./features/vis_simplemodel/train_nonrumor_vis.npy')

X_train = np.concatenate((train_pos, train_neg))
X_train_auxi_vis = np.concatenate((auxi_train_pos_vis, auxi_train_neg_vis))
train_label = np.concatenate((np.zeros(train_pos.shape[0]),np.ones(train_neg.shape[0])))
y_train = pd.get_dummies(train_label).as_matrix()

X_test = np.concatenate((test_pos, test_neg))
X_test_auxi_vis = np.concatenate((auxi_test_pos_vis, auxi_test_neg_vis))
test_label = np.concatenate((np.zeros(test_pos.shape[0]),np.ones(test_neg.shape[0])))
y_test = pd.get_dummies(test_label).as_matrix() # 单位矩阵

print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)
print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)

# output得到的是两个类别的概率，需要比较大小确定最终类别
# 把谣言类当做主类
def convert_label(label):
    res = []
    label = label.tolist()
    for ele in label:
        if ele[0] > ele[1]:
            res.append(0)
        else:
            res.append(1)
    return res

#把非谣言当做主类
def convert_label_nonrumor(label):
    res = []
    label = label.tolist()
    for ele in label:
        if ele[0] > ele[1]:
            res.append(1)
        else:
            res.append(0)
    return res


def build_base_model():

    ## text modality
    input_shape = (maxlen, embedding_dim)
    input_data = Input(input_shape, name = 'main_input')

    bilstm_out = Bidirectional(LSTM(hidden_num, return_sequences = True), merge_mode = 'concat')(input_data)
    bilstm_out = Bidirectional(LSTM(hidden_num, return_sequences = True), merge_mode = 'concat')(bilstm_out)

    # attention
    att_out = AttentionLayer3(name='attentionLayer')(bilstm_out)
    # [None, 20, 1]
    att_rep = Lambda(lambda x : K.repeat_elements(x, hidden_num * 2, axis=2))(att_out)
    merge_out = multiply([att_rep, bilstm_out])
    sum_out = Lambda(lambda x: K.sum(x, axis=1))(merge_out)

    ## visual modality
    auxi_input_vis = Input(shape = (auxi_test_pos_vis.shape[1],),name = 'auxi_input_vis')
    x_vis = Dense(hidden_num*2, activation = 'relu', kernel_regularizer = l2(0.005))(auxi_input_vis)

    x = keras.layers.concatenate([sum_out, x_vis]) # fusion way

    # x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu', kernel_regularizer = l2(0.005))(x)

    main_out = Dense(2, activation = 'softmax', name = 'main_output')(x)

    net = Model(input = [input_data, auxi_input_vis], output =  main_out)
    return net

def build_fusion_model():

    ########### text modality ##########
    # shape(None, maxlen, embedding_dim)
    input_text = Input(shape = (maxlen, embedding_dim), name = 'input_text')
    # (None, max_len, hidden_num * 2)
    # Q: 输出是None, None, hidden_num*2，可能是因为具体长度无法确定，数据预处理的时候做补齐和截断了吗
    bilstm_out1 = Bidirectional(LSTM(hidden_num, return_sequences = True), merge_mode = 'concat')(input_text)
    bilstm_out2 = Bidirectional(LSTM(hidden_num, return_sequences = True), merge_mode = 'concat')(bilstm_out1)
    # attention (None, max_len, 1)
    att_out = AttentionLayer3(name='attentionLayer')(bilstm_out2)
    # (None, max_len, hidden_num * 2)
    att_rep = Lambda(lambda x : K.repeat_elements(x, hidden_num * 2, axis=2))(att_out)
    # (None, max_len, hidden_num * 2)
    merge_out = multiply([att_rep, bilstm_out2])
    # (None, hidden_num * 2)
    x_text = Lambda(lambda x: K.sum(x, axis=1))(merge_out)

    ########### visiual modality ##########
    input_vis = Input(shape = (auxi_test_pos_vis.shape[1],),name = 'input_vis')
    # (None, hidden_num * 2)
    # x_vis = Dense(hidden_num*2, activation = 'relu', kernal_regularizer = l2(0.03))(input_vis)
    x_vis = Dense(hidden_num*2, activation = 'relu', kernel_regularizer = l2(0.01))(input_vis)

    ########### fusion ##########
    # (None, hidden_num * 4)
    x = keras.layers.concatenate([x_text, x_vis])

    # (None, 2)
    weights = Dense(2, activation='softmax', name='weights', kernel_regularizer = l2(0.01))(x)
    # (None, 2, 1)
    my_expand_dims = Lambda(lambda xx: K.expand_dims(xx, axis=2))
    weights1 = my_expand_dims(weights)
    # (None, 2, hidden_num *2)
    # weights = K.repeat_elements(weights, hidden_num*2, axis=1) 在函数式模型中调用后端函数需要用lambda进行封装
    my_weights = Lambda(lambda xx: K.repeat_elements(xx, hidden_num*2, axis=2))
    weights2 = my_weights(weights1)
    # print weights2.shape
    # (None, hidden_num *4, 1)
    weights3 = keras.layers.core.Reshape((hidden_num * 4, 1),name='weights_e')(weights2)
    # print weights3.shape
    # (None, hidden_num *4)
    my_sum = Lambda(lambda x: K.sum(x, axis=-1))
    weights4 = my_sum(weights3)
    # print weights4.shape

    # (None, hidden_num*4)
    x_fusion = keras.layers.multiply([x, weights4])

    # (None, 64)
    #x_fusion1 = Dense(64, activation = 'relu')(x_fusion)
    # (None, 32)
    x_fusion2 = Dense(32, activation = 'relu', kernel_regularizer = l2(0.01))(x_fusion)

    # my_dropout = Dropout(0.5)
    # x_fusion2 = my_dropout(x_fusion2)

    # main_out = Dense(2, activation = 'softmax', kernel_regularizer = l2(0.03), name = 'main_output')(x_fusion2)
    main_out = Dense(2, activation = 'softmax', name = 'main_output', kernel_regularizer = l2(0.01))(x_fusion2)
    #print main_out.shape

    model = Model(input = [input_text, input_vis], output =  main_out)
    return model

def compile_model(model):
    print "Training model..."

    print(model.summary())
    adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
    model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    # history = model.fit([X_train, X_train_auxi_vis],y_train, nb_epoch = epoch_size, batch_size = batch_size, validation_data = ([X_test,  X_test_auxi_vis], y_test))
    history = model.fit([X_train, X_train_auxi_vis],y_train, nb_epoch = epoch_size, batch_size =
            batch_size, shuffle = True, validation_split = 0.3)

    hist_logs = history.history
    print "max_val_acc: %f" % max(hist_logs['val_acc'])
    print "Saving history logs..."
    with open(history_path, 'wb') as f:
        pickle.dump(hist_logs, f)

    print "Saving model..."
    model.save(model_path)


def get_weights(model):
    weights_model = Model(inputs=model.input, outputs=model.get_layer('weights').output)
    weights_output = weights_model.predict([X_test, X_test_auxi_vis])
    print "Saving weights..."
    np.savetxt(weights_path, weights_output)


def predict_model(model):
    pre_label = model.predict([X_test, X_test_auxi_vis])

    print('#################')
    print('## NonRumor')
    print('#################')
    pre_label_c = convert_label_nonrumor(pre_label)
    test_label_c = convert_label_nonrumor(y_test)
    acc = metrics.accuracy_score(test_label_c, pre_label_c)
    precision = metrics.precision_score(test_label_c, pre_label_c)
    recall = metrics.recall_score(test_label_c, pre_label_c)
    f1 = metrics.f1_score(test_label_c, pre_label_c)
    print 'Accuracy: ' + str(acc)
    print 'Precision: ' + str(precision)
    print 'Recall: ' + str(recall)
    print 'F1: ' + str(f1)

    print('#################')
    print('## Average')
    print('#################')
    acc = metrics.accuracy_score(test_label_c, pre_label_c)
    precision = metrics.precision_score(test_label_c, pre_label_c, average = 'macro')
    recall = metrics.recall_score(test_label_c, pre_label_c, average = 'macro')
    f1 = metrics.f1_score(test_label_c, pre_label_c, average = 'macro')
    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

    print('#################')
    print('## Rumor')
    print('#################')
    pre_label_c = convert_label(pre_label)
    test_label_c = convert_label(y_test)
    acc = metrics.accuracy_score(test_label_c, pre_label_c)
    precision = metrics.precision_score(test_label_c, pre_label_c)
    recall = metrics.recall_score(test_label_c, pre_label_c)
    f1 = metrics.f1_score(test_label_c, pre_label_c)
    print('Accuracy: ' + str(acc))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

    fpr, tpr, thresholds = metrics.roc_curve(test_label_c, pre_label_c)
    auc = metrics.auc(fpr, tpr)
    print ('AUC: ' + str(auc))




def main():
    if (os.path.exists(model_path)):
        print "Model exists. Loading model..."
        model = load_model(model_path)
    else:
        print "Model doesn't exist. Building model..."
        model = build_fusion_model()
        # model = build_base_model()

    start = time.time()
    compile_model(model)
    get_weights(model)
    predict_model(model)
    end = time.time()
    print "time consumed: %.2f s" % (end - start)

if __name__ == '__main__':
    main()
# vim: set expandtab ts=4 sw=4 sts=4 tw=100:
