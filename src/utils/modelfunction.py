
import numpy as np
import time
import random
import tensorflow as tf
from keras import backend as K
from keras import callbacks

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from cfgs import *
from semantic_segmentation.dataset.transformer_ImageDataGenerator import dataagument,val_data_generator#augment_data
from semantic_segmentation.dataset import transformer
from semantic_segmentation.dataset.transformer import  Compose,RandomRotation,RandomContrast,RandomScale,RandomHorizontalFlip,RandomVerticalFlip
from keras.utils import to_categorical
from RF_DL.model.models import *
# from model.models import singlebranch
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling1D
from RF_DL.model.loss import *
from RF_DL.plot import *
import logging
import inspect

from tensorflow.keras import layers
import keras
from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply,Dense,Activation,Add,Flatten,Lambda ,Concatenate, Conv1D, Conv2D
from keras import backend as K
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, BatchNormalization, Conv1D, Conv2D, Activation
from tensorflow.keras.layers import Dense, Softmax, Flatten,Lambda,Concatenate
# from models import cnn3dattention 
from tensorflow import expand_dims
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras import layers
from keras.utils import to_categorical
from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply,Dense,Activation,Add,Flatten,Lambda ,Concatenate, Conv1D, Conv2D, Activation
from keras import backend as K
from keras.layers import Input, SeparableConv2D, Conv2D,DepthwiseConv2D
import math
import time
import os





def channel_attention_lowhigh(inputslow,inputshigh,ratio=8):
    
    # 通道维度上的平均池化
    # avg_pool= layers.TimeDistributed(layers.GlobalAveragePooling2D())(input_feature)
    inputs=layers.concatenate([inputslow,inputshigh],axis=-1)
    avg_pool= layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    avg_pool=Reshape((1,1,avg_pool.shape[-1]))(avg_pool)
    max_pool = Reshape((1,1,max_pool.shape[-1]))(max_pool)
    
    # avg_pool = Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(avggeo_pool)
    # max_pool= Lambda(lambda x: K.reshape(x, (K.shape(x)[0], 1, 1, K.shape(x)[-1])))(maxgeo_pool)
    # print('avg_pool',avg_pool.shape)
    channel =  avg_pool.shape[-1]  # 获取通道维度
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool=shared_layer_one(avg_pool
                               )
    avg_pool=shared_layer_two(avg_pool)
    print('avg2', avg_pool.shape) 
   
 
    max_pool= shared_layer_one(max_pool)
    max_pool= shared_layer_two(max_pool)


    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    inputshigh=Conv2D(filters=cbam_feature.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(inputshigh)
    inputshigh=BatchNormalization()(inputshigh)
    return multiply([inputshigh, cbam_feature])
def spatial_attention_lowhigh(low,high):
    # 空间注意力的计算，这里可以根据需要修改
    kernel_size = 5
    #两者都没有在网络图中显示
    # inputs=layers.concatenate([low,high],axis=-1)
    input1s=Concatenate(axis=-1)([low,high])
    print('inputconctat',input1s.shape)
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input1s)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input1s)
    print('concatspatial',avg_pool.shape,max_pool.shape) #(None, 256, 256, 1) (None, 256, 256, 1)
    concat = Concatenate(axis=-1)([avg_pool, max_pool]) #(None, 256, 256, 1) (None, 256, 256, 2)

    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    # print("spation",cbam_feature.shape) #(None, 7,7 1)
    low=Conv2D(filters=high.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(low)
    print('low',low.shape)
    # low=BatchNormalization()(low)
    return multiply([low, cbam_feature])
def deepwise2d(inputs,k,ratio=8,sequentialcbrm=True,cbrmlowhigh=False,coord=False,name='one'):
    x1=layers.Conv2D(inputs.shape[-1]*k,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs) 
    x1=layers.BatchNormalization()(x1)
    x1= layers.ReLU(max_value=6)(x1)
    x=DepthwiseConv2D(3,padding='same',activation='relu')(x1)
    # x=DepthwiseConv2D(3,padding='same',activation='relu')(x)
    # x=coordinate(x,ratio=8)
    x=layers.Conv2D(inputs.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x=Add()([inputs,x])
    print('dpwise',x.shape)
    if sequentialcbrm:
        if cbrmlowhigh:
            x=channel_attention_lowhigh(inputs,x,ratio=ratio)
            x=spatial_attention_lowhigh(inputs,x)
        else:
            x=channel_attention(x,ratio=ratio)
            x=spatial_attention(x)
       
    if coord:
        x=coordinate(x,ratio=32)
    # x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.add([inputs,x])
    return x

def depthwiseattention(inputs,channelratio=8,depthwisecbrm=False,cbrmlowhigh=False,depthwisecorrd=False):
    print('input',inputs.shape)
  
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(inputs)   
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    print('none1',x.shape)
    x = keras.layers.BatchNormalization()(x)
    x1= keras.layers.ReLU()(x)
    print('none2',x.shape)
    x=deepwise2d(x1,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=False,name='zero') 
    # x=layers.concatenate([inputs,x],axis=-1) 
    x2=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x2=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x2,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=False,name='one')
    # x=layers.concatenate([inputs,x],axis=-1) 
    x3=layers.Conv2D(64,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)  
    # x3=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x3,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='two')
    # x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.Conv2D(96,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    # x4=layers.MaxPooling2D((2,2))(x)
    x=deepwise2d(x,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='three')
    # x=layers.concatenate([inputs,x],axis=-1) 
    x=layers.Conv2D(128,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x) 
    x=deepwise2d(x,k=6,ratio=channelratio,sequentialcbrm=False,cbrmlowhigh=cbrmlowhigh,coord=depthwisecorrd,name='four')
    return x



def depthsepar2d(input,filters):
    x=DepthwiseConv2D(3,padding='same',activation='relu')(input)
    x=layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
  
    x1=layers.Conv2D(x.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(input)
    x1 = keras.layers.BatchNormalization()(x1)
    x=Add()([x1,x])
    x = layers.ReLU()(x)
    return x
def depthsepar2dspatialattention(input,filters):
    x=DepthwiseConv2D(3,padding='same',activation='relu')(input)
    x=layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    
    x = keras.layers.BatchNormalization()(x)
    x=spatial_attention_lowhigh(input,x)
    
    input=layers.Conv2D(x.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(input)
    # input= keras.layers.BatchNormalization()(input)
    x=Add()([input,x])
    x = layers.ReLU()(x)
    return x

def multscaledepthsepar2dspatialattention(input,filters):
    x3=DepthwiseConv2D(3,padding='same',activation='relu')(input)
    x5=DepthwiseConv2D(5,padding='same',activation='relu')(input)
    x35=Concatenate()([x3,x5])
    
    x=layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x35)
    x = keras.layers.BatchNormalization()(x)

    x=spatial_attention_lowhigh(input,x)
    
    input=layers.Conv2D(x.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(input)
    input= keras.layers.BatchNormalization()(input)
    x=Add()([input,x])
    x = layers.ReLU()(x)
    return x

def depthsepar2dchannelspatialattention(input,filters):
    x=DepthwiseConv2D(3,padding='same',activation='relu')(input)
    x=layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    # x = keras.layers.BatchNormalization()(x)
    x=channel_attention_lowhigh(input,x,8)

    x=spatial_attention_lowhigh(input,x)
    
    input=layers.Conv2D(x.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(input)
    # input= keras.layers.BatchNormalization()(input)
    x=Add()([input,x])
    x = layers.ReLU()(x)
    return x




def dualsparableCnn2d(inputshape,numfilters,sattention111,sattention011, multscalesattetion,multscalesattetion001,csattention,noattention,concatdense,concatcnntrue1d,dropout=0,L2=0):


    inputs= keras.Input(shape=inputshape) #(None, 2, 11, 11, 15)
  
    inputheight,inputwidth,inputschannels=inputs.shape[1],inputs.shape[2],inputs.shape[-1]
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x1)
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    if sattention111==True:
        if numfilters==3:
            x=depthsepar2dspatialattention(x,64)
            x=depthsepar2dspatialattention(x,128)
            x=depthsepar2dspatialattention(x,256)
    if multscalesattetion==True:
        x=multscaledepthsepar2dspatialattention(x,64)
        x=multscaledepthsepar2dspatialattention(x,128)
        x=multscaledepthsepar2dspatialattention(x,256)
    if multscalesattetion001==True:
        if numfilters==3:           
            x=depthsepar2d(x,64)
            x=depthsepar2d(x,128)
            x=multscaledepthsepar2dspatialattention(x,256)
    if noattention==True:
        # x=ResBlock(x,64)
        # x=ResBlock(x,128)
        # x=ResBlock(x,256)
        x=depthsepar2d(x,64)
        x=depthsepar2d(x,128)
        x=depthsepar2d(x,256)
    x=layers.GlobalAveragePooling2D()(x)
    x=Flatten()(x)
    x=Dense(256,kernel_regularizer=reg)(x)
    # intergrate pixel and patch
    if concatcnntrue1d==True:
        xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputschannels])(x1)
        xcenter=Reshape((xcenter.shape[-1],1))(xcenter) # 15,1
        xcenter=layers.Conv1D(32,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)
        xcenter=layers.Conv1D(64,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)
        xcenter=layers.Conv1D(128,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)
        xcenter=layers.Conv1D(256,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)
        print('xcenter',xcenter.shape)
        xcenter=layers.GlobalAveragePooling1D(name='average')(xcenter)
        
        xcenter=Reshape((xcenter.shape[-1],))(xcenter)#256,
    
        xc=layers.concatenate([xcenter,x],axis=-1)
        #eca和dense区别不大，dense更好;dense中ratio的确定;
        # x=eca_block(x, 1, 2)
        c=xc.shape[-1] 
        x=Dense(c/8,kernel_regularizer=reg)(xc)
        x=Dense(c,kernel_regularizer=reg)(x)
        xattention=Activation('sigmoid')(x)
        x=Multiply()([xattention,xc])
    

    x= layers.Dropout(dropout)(x)
    x=Dense(64,kernel_regularizer=reg)(x)
    x= layers.Dropout(dropout)(x)
    print('x',x.shape)
    output_layer=Dense(2,kernel_regularizer=reg,activation='softmax')(x) #,activation='sigmoid'
    # output_layer=Dense(1, activation='sigmoid')(x)

    return Model(inputs,output_layer)

#%%   
inputshape=(11,11,15)
dropout,L2=0,0
inputtag=0
# m= dualsparableCnn2d(inputtag,inputshape,3,sattention111=0,sattention011=1, multscalesattetion=0,multscalesattetion001=0,csattention=0,noattention=0,concatdense=1,concatcnntrue1d=1,dropout=0,L2=0)
# m.summary()
