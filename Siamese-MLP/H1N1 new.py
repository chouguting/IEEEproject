# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:26:04 2019

@author: yu-hao
"""
import random
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import numpy
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing
from sklearn import svm, metrics
# numpy.random.seed(10)
import sys
import all_22277_list

SAME_SYMPTOMS = 1
DIFF_SYMPTOMS = 0
POSITIVE = 1
NEGATIVE = 0
is_suv = 2

# 366*22279
all_df = pd.read_csv("rma_H1N1_leave_001_out.csv", sep='\t', encoding='utf-8')

# 11*22279
standar_df = pd.read_csv("H1N1_health_no_001.txt", sep='\t', encoding='utf-8')

# 16*22279
test_df = pd.read_csv("rma_H1N1_001.txt", sep='\t', encoding='utf-8')

# 北個DATA代表的意思(Title)
cols = all_22277_list.cols
# print((cols))


# 把DATA利用cols當作列來排好
all_df = all_df[cols]
standar_df = standar_df[cols]
test_df = test_df[cols]

# 生成366個(all_df的樣本數) 隨機數字 並且取<0.8的作為True ,其他是false
msk = numpy.random.rand(len(all_df)) < 0.8
# train_df = all_df[160:]
# test_df = all_df[:16]

# 從剛才的msk中 True的編號從all_df中提取當作訓練用(大約8成)，其餘用來測試用
train_df = all_df[msk]
val_df = all_df[~msk]

'''print('total:',len(all_df),
      'train:',len(train_df),
      'val:',len(val_df),
      'test:',len(test_df))'''


def PreprocessData(raw_df):
    df = raw_df.drop(['ID'], axis=1)  # 移除name欄位
    ndarray = df.values  # dataframe轉換為array
    Features = ndarray[:, 1:]
    Label = ndarray[:, 0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, Label


train_Features, train_Label = PreprocessData(train_df)
# print(len(train_Features))
val_Features, val_Label = PreprocessData(val_df)
test_Features, test_Label = PreprocessData(test_df)
standar_Features, standar_Label = PreprocessData(standar_df)


# 歐式距離
def euclid_dis(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


# loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, train_symptoms):
    n = min([len(train_symptoms[NEGATIVE]), len(train_symptoms[POSITIVE])])
    random_pairs = []
    random_labels = []
    negative_negative_pairs = []
    negative_positive_pairs = []
    positive_positive_pairs = []
    positive_negative_pairs = []
    negative_negative_pairs = create_negative_negative_pairs(x, train_symptoms)
    negative_positive_pairs = create_negative_positive_pairs(x, train_symptoms)
    positive_positive_pairs = create_positive_positive_pairs(x, train_symptoms)
    positive_negative_pairs = create_positive_negative_pairs(x, train_symptoms)
    for i in range(n * n // 4):
        random_pairs += [negative_negative_pairs[i], negative_positive_pairs[i]]
        random_labels += [SAME_SYMPTOMS, DIFF_SYMPTOMS]
        random_pairs += [positive_positive_pairs[i], positive_negative_pairs[i]]
        random_labels += [SAME_SYMPTOMS, DIFF_SYMPTOMS]
    # print(numpy.array(random_pairs).shape)
    # print(numpy.array(random_labels).shape)
    return numpy.array(random_pairs), numpy.array(random_labels)


def create_negative_negative_pairs(x, train_symptoms):
    negative_negative_pairs = []
    random_negative_negative_pairs = []
    n = min([len(train_symptoms[NEGATIVE]), len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[NEGATIVE])):
        for j in range(len(train_symptoms[NEGATIVE])):
            z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[NEGATIVE][j]
            negative_negative_pairs += [[x[z1], x[z2]]]
    # print("negative_negative_pairs:",len(negative_negative_pairs))
    negative_negative_pairs_index = random.sample(range(0, len(negative_negative_pairs)), n * n // 4)
    for i in range(len(negative_negative_pairs_index)):
        random_negative_negative_pairs += [negative_negative_pairs[negative_negative_pairs_index[i]]]
    # print("negative_negative_pairs_index:",len(negative_negative_pairs_index))
    # print("random_negative_negative_pairs:",len(random_negative_negative_pairs))
    return random_negative_negative_pairs


def create_negative_positive_pairs(x, train_symptoms):
    negative_positive_pairs = []
    random_negative_positive_pairs = []
    k = 0
    n = min([len(train_symptoms[NEGATIVE]), len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[NEGATIVE])):
        for j in range(len(train_symptoms[NEGATIVE])):
            if (k < len(train_symptoms[POSITIVE])):
                z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][k]
                negative_positive_pairs += [[x[z1], x[z2]]]
                k += 1
            else:
                k = 0
                z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][k]
                negative_positive_pairs += [[x[z1], x[z2]]]
    # print("negative_positive_pairs:",len(negative_positive_pairs))
    negative_positive_pairs_index = random.sample(range(0, len(negative_positive_pairs)), n * n // 4)
    for i in range(len(negative_positive_pairs_index)):
        random_negative_positive_pairs += [negative_positive_pairs[negative_positive_pairs_index[i]]]
    # print("negative_positive_pairs_index:",len(negative_positive_pairs_index))
    # print("random_negative_positive_pairs:",len(random_negative_positive_pairs))
    return random_negative_positive_pairs


def create_positive_positive_pairs(x, train_symptoms):
    positive_positive_pairs = []
    random_positive_positive_pairs = []
    n = min([len(train_symptoms[NEGATIVE]), len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[POSITIVE])):
        for j in range(len(train_symptoms[POSITIVE])):
            z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[POSITIVE][j]
            positive_positive_pairs += [[x[z1], x[z2]]]
    # print("positive_positive_pairs:",len(positive_positive_pairs))
    positive_positive_pairs_index = random.sample(range(0, len(positive_positive_pairs)), n * n // 4)
    for i in range(len(positive_positive_pairs_index)):
        random_positive_positive_pairs += [positive_positive_pairs[positive_positive_pairs_index[i]]]
    # print("positive_positive_pairs_index:",len(positive_positive_pairs_index))
    # print("random_positive_positive_pairs:",len(random_positive_positive_pairs))
    return random_positive_positive_pairs


def create_positive_negative_pairs(x, train_symptoms):
    positive_negative_pairs = []
    random_positive_negative_pairs = []
    k = 0
    n = min([len(train_symptoms[NEGATIVE]), len(train_symptoms[POSITIVE])])
    for i in range(len(train_symptoms[POSITIVE])):
        for j in range(len(train_symptoms[POSITIVE])):
            if (k < len(train_symptoms[NEGATIVE])):
                z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][k]
                positive_negative_pairs += [[x[z1], x[z2]]]
                k += 1
            else:
                k = 0
                z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][k]
                positive_negative_pairs += [[x[z1], x[z2]]]
                k = 0
    # print("positive_negative_pairs:",len(positive_negative_pairs))
    positive_negative_pairs_index = random.sample(range(0, len(positive_negative_pairs)), n * n // 4)
    for i in range(len(positive_negative_pairs_index)):
        random_positive_negative_pairs += [positive_negative_pairs[positive_negative_pairs_index[i]]]
    # print("positive_negative_pairs_index:",len(positive_negative_pairs_index))
    # print("random_positive_negative_pairs:",len(random_positive_negative_pairs))
    return random_positive_negative_pairs


'''
def create_pairs(x, train_symptoms):
    pairs = []
    labels = []
    random_pairs = []
    random_labels = []
    n=min([len(train_symptoms[NEGATIVE]),len(train_symptoms[POSITIVE])])
    final_label0_index = len(train_symptoms[NEGATIVE])*len(train_symptoms[NEGATIVE])*2
    k = 0
    print('n:',n)
    for i in range(len(train_symptoms[NEGATIVE])):
        for j in range(len(train_symptoms[NEGATIVE])):
            z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[NEGATIVE][j]
            pairs += [[x[z1], x[z2]]]
            if(k < len(train_symptoms[POSITIVE])):
                z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][k]
                pairs += [[x[z1], x[z2]]]
                k += 1
            else:
                k = 0
                z1, z2 = train_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][k]
                pairs += [[x[z1], x[z2]]]
            labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    k = 0
    for i in range(len(train_symptoms[POSITIVE])):
        for j in range(len(train_symptoms[POSITIVE])):
            z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[POSITIVE][j]
            pairs += [[x[z1], x[z2]]]
            if(k < len(train_symptoms[NEGATIVE])):
                z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][k]
                pairs += [[x[z1], x[z2]]]
                k += 1
            else:
                k = 0
                z1, z2 = train_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][k]
                pairs += [[x[z1], x[z2]]]
            labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    print("pairs:",len(pairs),"labels:",len(labels))
    zero_zero_pair_index = random.sample(range(0,final_label0_index,2), n*n)
    zero_one_pair_index = random.sample(range(1,final_label0_index,2), n*n)
    one_one_pair_index = random.sample(range(final_label0_index,len(pairs),2), n*n)
    one_zero_pair_index = random.sample(range(final_label0_index+1,len(pairs),2), n*n)
    for i in range(n*n//8):
        random_pairs += [pairs[zero_zero_pair_index[i]],pairs[zero_one_pair_index[i]]]
        random_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
        random_pairs += [pairs[one_one_pair_index[i]],pairs[one_zero_pair_index[i]]]
        random_labels += [SAME_SYMPTOMS,DIFF_SYMPTOMS]
    print("zero_zero_pair_index:",len(zero_zero_pair_index))
    print("zero_one_pair_index:",len(zero_one_pair_index))
    print("one_one_pair_index:",len(one_one_pair_index))
    print("one_zero_pair_index:",len(one_zero_pair_index))
    print("random_labels:",len(random_labels))
    print(numpy.array(pairs).shape)
    print(numpy.array(random_pairs).shape)
    return numpy.array(random_pairs), numpy.array(random_labels)
'''


def create_test_pairs(x, train_symptoms, y, test_symptoms):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    positive_random_index = []
    negative_random_index = []
    test_pairs = []
    test_labels = []
    if (len(test_symptoms[NEGATIVE]) > 0):
        positive_random_index = random.sample(range(0, len(train_symptoms[POSITIVE])), len(test_symptoms[NEGATIVE]))
        negative_random_index = random.sample(range(0, len(train_symptoms[NEGATIVE])), len(test_symptoms[NEGATIVE]))
        # print('positive_random_index:',positive_random_index)
        # print('negative_random_index:',negative_random_index)
        for i in range(len(test_symptoms[NEGATIVE])):
            z1, z2 = test_symptoms[NEGATIVE][i], train_symptoms[NEGATIVE][negative_random_index[i]]
            test_pairs += [[y[z1], x[z2]]]
            z1, z2 = test_symptoms[NEGATIVE][i], train_symptoms[POSITIVE][positive_random_index[i]]
            test_pairs += [[y[z1], x[z2]]]
            test_labels += [SAME_SYMPTOMS, DIFF_SYMPTOMS]
    if (len(test_symptoms[NEGATIVE]) > 0):
        positive_random_index = random.sample(range(0, len(train_symptoms[POSITIVE])), len(test_symptoms[NEGATIVE]))
        negative_random_index = random.sample(range(0, len(train_symptoms[NEGATIVE])), len(test_symptoms[NEGATIVE]))
        # print('positive_random_index:',positive_random_index)
        # print('negative_random_index:',negative_random_index)
        for i in range(len(test_symptoms[NEGATIVE])):
            z1, z2 = train_symptoms[NEGATIVE][negative_random_index[i]], test_symptoms[NEGATIVE][i]
            test_pairs += [[x[z1], y[z2]]]
            z1, z2 = train_symptoms[POSITIVE][positive_random_index[i]], test_symptoms[NEGATIVE][i]
            test_pairs += [[x[z1], y[z2]]]
            test_labels += [SAME_SYMPTOMS, DIFF_SYMPTOMS]
    if (len(test_symptoms[POSITIVE]) > 0):
        positive_random_index = random.sample(range(0, len(train_symptoms[POSITIVE])), len(test_symptoms[POSITIVE]))
        negative_random_index = random.sample(range(0, len(train_symptoms[NEGATIVE])), len(test_symptoms[POSITIVE]))
        # print('positive_random_index:',positive_random_index)
        # print('negative_random_index:',negative_random_index)
        for i in range(len(test_symptoms[POSITIVE])):
            z1, z2 = test_symptoms[POSITIVE][i], train_symptoms[POSITIVE][positive_random_index[i]]
            test_pairs += [[y[z1], x[z2]]]
            z1, z2 = test_symptoms[POSITIVE][i], train_symptoms[NEGATIVE][negative_random_index[i]]
            test_pairs += [[y[z1], x[z2]]]
            test_labels += [SAME_SYMPTOMS, DIFF_SYMPTOMS]
    if (len(test_symptoms[POSITIVE]) > 0):
        positive_random_index = random.sample(range(0, len(train_symptoms[POSITIVE])), len(test_symptoms[POSITIVE]))
        negative_random_index = random.sample(range(0, len(train_symptoms[NEGATIVE])), len(test_symptoms[POSITIVE]))
        # print('positive_random_index:',positive_random_index)
        # print('negative_random_index:',negative_random_index)
        for i in range(len(test_symptoms[POSITIVE])):
            z1, z2 = train_symptoms[POSITIVE][positive_random_index[i]], test_symptoms[POSITIVE][i]
            test_pairs += [[x[z1], y[z2]]]
            z1, z2 = train_symptoms[NEGATIVE][negative_random_index[i]], test_symptoms[POSITIVE][i]
            test_pairs += [[x[z1], y[z2]]]
            test_labels += [SAME_SYMPTOMS, DIFF_SYMPTOMS]
    # print('len(test_pairs):',len(test_pairs))
    # print(numpy.array(test_pairs).shape)
    return numpy.array(test_pairs), numpy.array(test_labels)


def create_standar_test_pairs(x, standar_symptoms, y, test_symptoms):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    test_pairs = []
    test_labels = []
    if (len(test_symptoms[NEGATIVE]) > 0):
        for i in range(len(standar_symptoms[NEGATIVE])):
            for j in range(len(test_symptoms[NEGATIVE])):
                z1, z2 = standar_symptoms[NEGATIVE][i], test_symptoms[NEGATIVE][j]
                test_pairs += [[x[z1], y[z2]]]
                test_labels += [SAME_SYMPTOMS]
    if (len(test_symptoms[POSITIVE]) > 0):
        for i in range(len(standar_symptoms[NEGATIVE])):
            for j in range(len(test_symptoms[POSITIVE])):
                z1, z2 = standar_symptoms[NEGATIVE][i], test_symptoms[POSITIVE][j]
                test_pairs += [[x[z1], y[z2]]]
                test_labels += [DIFF_SYMPTOMS]

    # print('len(test_pairs):',len(test_pairs))
    # print(numpy.array(test_pairs).shape)
    return numpy.array(test_pairs), numpy.array(test_labels)


def create_base_net(input_shape):
    '''
    input = Input(shape = input_shape)
   #x = Flatten()(input)
#####################################################
    x = Dense(100, activation='relu')(input)
    x = Dropout(0.1)(x)
    #x = Dropout(0.2)(x)
    x = Dense(100, activation='relu')(x)
    #x = Dropout(0.1)(x)
    x = Dense(100, activation='relu')(x)
    #x = Dropout(0.2)(x)
    x = Dense(100, activation='relu')(x)
    #x = Dense(50, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input, x)
    '''
    # model.summary()
    model = Sequential()
    model.add(Dense(units=100, input_dim=22277, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    return model


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return numpy.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# the data, split between train and test sets
input_shape = train_Features.shape[1:]
# print(input_shape[0])
standar_symptoms = [numpy.where(standar_Label == i)[0] for i in range(is_suv)]
# create training+test positive and negative pairs
train_symptoms = [numpy.where(train_Label == i)[0] for i in range(is_suv)]
a = [numpy.where(train_Label == 0)[0], numpy.where(train_Label == 1)[0]]
# print("len(a[0]),len(a[1])",len(a[0]),len(a[1]))
# print(len(train_symptoms[0]),len(train_symptoms[1]))
train_pairs, train_y = create_pairs(train_Features, train_symptoms)
# print("len(train_pairs),len(train_y)",len(train_pairs),len(train_y))
# plot_images_labels_prediction(x_train[0])
val_symptoms = [numpy.where(val_Label == i)[0] for i in range(is_suv)]
val_pairs, val_y = create_test_pairs(train_Features, train_symptoms, val_Features, val_symptoms)
# print("len(val_pairs),len(val_y)",len(val_pairs),len(val_y))

test_symptoms = [numpy.where(test_Label == i)[0] for i in range(is_suv)]
test_pairs, test_y = create_standar_test_pairs(standar_Features, standar_symptoms, test_Features, test_symptoms)
# print("len(test_pairs),len(test_y)",len(test_pairs),len(test_y))
# network definition
base_network = create_base_net(input_shape)
base_network.save("base_network.h5")
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches


# TODO: WHAT IS THIS???
# ??????????????????
processed_a = base_network(input_a)
processed_b = base_network(input_b)

print(type(base_network))
print(type(input_a))

distance = Lambda(euclid_dis,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

print(type(processed_a))


# 跑20次

model = Model([input_a, input_b], distance)
# print("start")
import tensorflow.keras

rms = RMSprop()
opt = tensorflow.keras.optimizers.Adam(lr=0.00001)


model.compile(loss=contrastive_loss, optimizer=opt, metrics=[accuracy])
train_history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], tf.cast(train_y, tf.float32),
                          batch_size=200,
                          epochs=25,
                          validation_data=([val_pairs[:, 0], val_pairs[:, 1]], tf.cast(val_y, tf.float32)))

# compute final accuracy on training and test sets
y_pred_train = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
train_acc = compute_accuracy(train_y, y_pred_train)
y_pred_val = model.predict([val_pairs[:, 0], val_pairs[:, 1]])
val_acc = compute_accuracy(val_y, y_pred_val)
test_pred = []
y_pred_test = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
test_acc = compute_accuracy(test_y, y_pred_test)
for i in y_pred_test:
    if (i > 0.5):
        # print(i)
        test_pred.append(0)
    else:
        # print(i)
        test_pred.append(1)

# roc
# fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pred_test, pos_label=0)
# print("fpr:",fpr)
# print("tpr:",tpr)
# auc_roc = metrics.auc(fpr, tpr)
# pr
# precision, recall, thresholds = precision_recall_curve(test_y, y_pred_test, pos_label=0)
# auc_pr = metrics.auc(recall, precision)

test_TP = 0
test_TN = 0
test_FP = 0
test_FN = 0
# print(test_pred)
# print(test_y)
for j in range(len(test_pred)):
    if (test_pred[j] == SAME_SYMPTOMS and test_pred[j] == test_y[j]):
        test_TP = test_TP + 1
    else:
        test_TP = test_TP + 0

    if (test_pred[j] == DIFF_SYMPTOMS and test_pred[j] == test_y[j]):
        test_TN = test_TN + 1
    else:
        test_TN = test_TN + 0

    if (test_y[j] == DIFF_SYMPTOMS and test_pred[j] == SAME_SYMPTOMS):
        test_FP = test_FP + 1
    else:
        test_FP = test_FP + 0

    if (test_y[j] == SAME_SYMPTOMS and test_pred[j] == DIFF_SYMPTOMS):
        test_FN = test_FN + 1
    else:
        test_FN = test_FN + 0
print("test_TP:", test_TP)
print("test_TN:", test_TN)
print("test_FP:", test_FP)
print("test_FN:", test_FN)

print('* Accuracy on training set: %0.2f%%' % (100 * train_acc))
print('* Accuracy on validation set: %0.2f%%' % (100 * val_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * test_acc))
print("==========================================\n\n")


