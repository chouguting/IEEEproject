# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna
"""

import pandas as pd
from sklearn import preprocessing
import sys

import all_22277_list

all_df = pd.read_csv("./"+sys.argv[1]+".txt",sep='\t',encoding='utf-8')
#all_df = pd.read_csv("./"+"H1N1Data"+".txt",sep='\t',encoding='utf-8')

cols=all_22277_list.cols
all_df=all_df[cols]

def PreprocessData(raw_df):
   
    df=raw_df.drop(['ID'], axis=1)
    ndarray = df.values
    Features = ndarray[:,1:] 
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
test_Features=all_Features[:2]
test_Label=all_Label[:2]
train_Features=all_Features[2:]
train_Label=all_Label[2:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(units=100, input_dim=22277, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, y=train_Label, validation_split=0.1, epochs=150, batch_size=200,verbose=0)
scores=model.evaluate(x=test_Features, y=test_Label)

all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns), 'probability',all_probability)
predict=all_probability[:2]
predict_score=all_probability_score[:2]
predict=predict.tolist()
predict_score=predict_score.tolist()

f1 = open("./"+"result"+".txt", 'a', encoding = 'UTF-8')
f1.write(str(int(test_Label[0]))+"\t"+str(predict[0])+"\t"+str(predict[1])+"\t"+str(predict_score[0])+"\t"+str(predict_score[1])+"\n")
f1.close
