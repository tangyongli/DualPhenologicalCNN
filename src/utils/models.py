import numpy as np
import time
import random
import tensorflow as tf
from keras import backend as K
from keras import callbacks
import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Reshape, LayerNormalization, multiply, Dense, Activation, Add, Flatten, Lambda, Concatenate, Conv1D, Conv2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, Activation,DepthwiseConv2D
from tensorflow.keras.layers import Dense, Softmax, Flatten, Lambda, Concatenate
from tensorflow import expand_dims
from keras.models import Model
import math
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
def spatial_attention_lowhigh(low,high,kernel_size=7):
  
    inputs=Concatenate(axis=-1)([low,high])
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    print('concatspatial',avg_pool.shape,max_pool.shape) #(None, 256, 256, 1) (None, 256, 256, 1)
    concat = Concatenate(axis=-1)([avg_pool, max_pool]) #(None, 256, 256, 1) (None, 256, 256, 2)
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    # print("spation",cbam_feature.shape) #(None, 7,7 1)
    low=Conv2D(filters=high.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(low)
    low=BatchNormalization()(low)
    return multiply([low, cbam_feature])
def attentionf(low,high):
    x=channel_attention_lowhigh(low,high,ratio=8)
    x=spatial_attention_lowhigh(low,x)
    return x

def channel_attention(inputs,ratio=8):
    
    # 通道维度上的平均池化
    # avg_pool= layers.TimeDistributed(layers.GlobalAveragePooling2D())(input_feature)
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
    # inputshigh=Conv2D(filters=cbam_feature.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(inputs)
   
    return multiply([inputs, cbam_feature])
def spatial_attention(inputs,kernel_size=7):
  
   
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    print('concatspatial',avg_pool.shape,max_pool.shape) #(None, 256, 256, 1) (None, 256, 256, 1)
    concat = Concatenate(axis=-1)([avg_pool, max_pool]) #(None, 256, 256, 1) (None, 256, 256, 2)
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    # print("spation",cbam_feature.shape) #(None, 7,7 1)
    # low=Conv2D(filters=high.shape[-1], kernel_size=1, strides=1, padding='same',  kernel_initializer='he_normal', use_bias=False)(low)
    # low=BatchNormalization()(low)
    return multiply([inputs, cbam_feature])
def attentionf(low,high):
    x=channel_attention_lowhigh(low,high,ratio=8)
    x=spatial_attention_lowhigh(low,x)
    return x


def conv2dbnblock(inputs,kernel_size,num_filters):
    x = layers.Conv2D(num_filters, kernel_size, strides=1,padding="same",kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization()(x) 
  
    return x
def ResBlock(inputs,kernel_size,num_filters,two,attention):
        if two==False:
            if attention==0:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
            elif attention==1:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=spatial_attention_lowhigh(inputs,x)
            elif attention==2:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=channel_attention_lowhigh(inputs,x)
            elif attention==3:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=attentionf(inputs,x)
              
        else:
            if attention==0:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=layers.ReLU()(x)
                x=conv2dbnblock(x,kernel_size,num_filters)
            elif attention==1:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=layers.ReLU()(x)
                x=conv2dbnblock(x,kernel_size,num_filters)
                x=spatial_attention_lowhigh(inputs,x)
            elif attention==2:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=layers.ReLU()(x)
                x=conv2dbnblock(x,kernel_size,num_filters)
                x=channel_attention_lowhigh(inputs,x,8)
            elif attention==3:
                x=conv2dbnblock(inputs,kernel_size,num_filters)
                x=layers.ReLU()(x)
                x=conv2dbnblock(x,kernel_size,num_filters)
                x=attentionf(inputs,x)
        inputs=conv2dbnblock(inputs,kernel_size=1,num_filters=x.shape[-1])
        x=layers.Add()([inputs, x])
        x=layers.ReLU()(x)
        return x
def resnetattention(inputshape,cnn2dfilters,cnn1dfilters,two=False,attention=0,dropout=0):
    inputs= keras.Input(shape=inputshape)
    x= conv2dbnblock(inputs,kernel_size=3,num_filters=cnn2dfilters[0])
    x=layers.ReLU()(x)
    x= conv2dbnblock(x,kernel_size=3,num_filters=cnn2dfilters[1])
    x=layers.ReLU()(x)
    print('x',x.shape)
    # repeat the resnet block 3 times
    x=ResBlock(x,cnn2dfilters[2],3,two=two,attention=attention)
    print(x.shape)
    x=ResBlock(x,cnn2dfilters[3],3,two=two,attention=attention)
    x=ResBlock(x,cnn2dfilters[4],3,two=two,attention=attention)
    x2d=GlobalAveragePooling2D()(x)
    x2d=Dense(256)(x2d)
    x1d= cnn1d(inputs,cnn1dfilters)

    x=featurefusion(cnn2df=x2d,cnn1df=x1d,ratio=8)

    x= layers.Dropout(dropout)(x)
    x=Dense(64)(x)
    x= layers.Dropout(dropout)(x)
    output_layer=Dense(2,activation='softmax')(x) #,activation='sigmoid'
    # output_layer=Dense(1, activation='sigmoid')(x)

    return Model(inputs,output_layer)
 

def simplecnn2d(inputshape,num_filters,dropratio):
    inputs= keras.Input(shape=inputshape) # 11,11,121
    x=layers.Conv2D(num_filters[0], 3, strides=1,padding="same", activation='relu')(inputs)
    x=layers.Conv2D(num_filters[1],3,strides=1,padding='same', activation='relu')(x) # 
   
    # x=ResBlock(x,num_filters[1]) 
    x=layers.Conv2D(num_filters[2],3,strides=1,padding='same', activation='relu')(x) #
    x=layers.Conv2D(num_filters[3],3,strides=1,padding='same', activation='relu')(x) # 
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(num_filters[4],3,strides=1,padding='same', activation='relu')(x) 
    x=GlobalAveragePooling2D()(x)
    x=Dense(512)(x)
    x= layers.Dropout(dropratio)(x)
    output_layer=Dense(2,activation='softmax')(x)
    return Model(inputs, output_layer)


def depthsepar2dattention(input,filters,multiscale=False,attention=0):
    '''
    attention:
    0,no attention
    1spatial attention
    2,channel attention
    3,channel and spatial attention low high
    4,channel and spatial attention,common
    '''
    if multiscale:
        x1=DepthwiseConv2D(1,padding='same')(input)
        # x1 = BatchNormalization()(x1) 
        x3=DepthwiseConv2D(3,padding='same')(input)
        # x3 = BatchNormalization()(x3) 
        x5=DepthwiseConv2D(5,padding='same')(input)
        # x5 = BatchNormalization()(x5) 
        x=Concatenate()([x1,x3,x5])
        x=BatchNormalization()(x)
        x=layers.ReLU()(x)
    else:
        x=DepthwiseConv2D(3,padding='same')(input)
        x = BatchNormalization()(x) 
        x=layers.ReLU()(x)
    x1=layers.Conv2D(filters,(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x1 = keras.layers.BatchNormalization()(x1)
    if attention==0:
        x=x1
    elif attention==1:
        x=spatial_attention_lowhigh(input,x1)
    elif attention==2:
        x=channel_attention_lowhigh(input,x1,8)
    elif attention==3:
        x=attentionf(low=input,high=x1)
    elif attention==4:
        x=channel_attention(x1,ratio=8)
        # x3=layers.Add()([x1, x2])
        x=spatial_attention(x)
        # x=layers.Add()([x3, x])

    input=layers.Conv2D(x.shape[-1],(1,1),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(input)
    input= keras.layers.BatchNormalization()(input)
    x=Add()([input,x])
    x = layers.ReLU()(x)
    print('x',x.shape)
    return x

# 输入和输出都是none,channels
def eca_block(inputs, b=1, gama=2):
    print('inputs',inputs.shape) #(None, 256)
    # 输入特征图的通道数
    in_channel = inputs.shape[-1]
 
    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(256, 2) + b) / gama))
 
    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size
 
    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1
 
    # [h,w,c]==>[None,c] 全局平均池化

 
    # [None,c]==>[c,1]
    x = layers.Reshape((in_channel, 1))(inputs) # (None, 256, 1)
    # print('xreshape',x.shape)
    # [c,1]==>[c,1] 1D卷积输入3纬
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
 
    # sigmoid激活
    x = tf.nn.sigmoid(x) # 
    print('xsigmoid',x.shape) # xsigmoid (None, 256, 1)
 
    # [c,1]==>[1,1,c]
    x = layers.Reshape((in_channel,))(x) #none,256
    print('x',x.shape)

    
    '''
    import tensorflow as tf

# Define the tensors
A = tf.constant([[1, 2]])  # Shape: (1, 2)
B = tf.constant([[[3, 4]]])  # Shape: (1, 1, 2)

# Multiply the tensors
result = tf.multiply(A, B)

# Print the result
print(result.shape)  # Output: (1, 1, 2)

    
    
    
    '''
    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])
    print('outputs',outputs.shape)
    outputs = Add()([inputs,outputs])
    return outputs
def cnn1d(inputs,num_filters,cnn1dattention):
    inputheight,inputwidth,inputchannels=inputs.shape[1],inputs.shape[2],inputs.shape[-1]
    xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputchannels])(inputs)
    xcenter=Reshape((xcenter.shape[-1],1))(xcenter) # 15,1
    for i in range(len(num_filters)):
        
        xcenter=layers.Conv1D(num_filters[i],(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)
        # xcenter=BatchNormalization()(xcenter) 
        xcenter=layers.ReLU()(xcenter)
    print('xcenter1',xcenter.shape) # xcenter1 (None, 1, 256)
    xcenter=Reshape((xcenter.shape[-1],))(xcenter)  # none,256   
    if cnn1dattention:
        xcenter=eca_block(xcenter, b=1, gama=2)
        # xcenter=Reshape((xcenter.shape[-1],))(xcenter) # none,256
        # print('xcenter',xcenter.shape)
        # c=xcenter.shape[-1] 
        # x=Dense(c/8)(xcenter)
        # x=Dense(c)(x)
        # xattention=Activation('sigmoid')(x)
        # xcenter=multiply([xattention,xcenter])
    # xcenter=layers.Conv1D(256,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)\
    # xcenter=layers.GlobalAveragePooling1D(name='average')(xcenter)
    # print('xcenter2',xcenter.shape) # none,256

    print('xcenter2',xcenter.shape) #none,256
    # xcenter=layers.Dense(256)(xcenter)

    return xcenter
def cnn1dsingle(inputshape,num_filters,cnn1dattention,dropout): # 1,1,16
    input=keras.Input(shape=inputshape) #1,1,16
    xcenter=Reshape((input.shape[-1],1))(input) # 16,1
    for i in range(len(num_filters)):
        
        xcenter=layers.Conv1D(num_filters[i],(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(xcenter)
        xcenter=BatchNormalization()(xcenter) 
        xcenter=layers.ReLU()(xcenter)
    print('xcenter',xcenter.shape) # none,1,256
    xcenter=layers.GlobalAveragePooling1D(name='average')(xcenter)
    xcenter=Reshape((xcenter.shape[-1],1))(xcenter)
    if cnn1dattention:
        xcenter=eca_block(xcenter, b=1, gama=2)
        xcenter=Reshape((xcenter.shape[-1],))(xcenter) # none,256
        print('xcenter',xcenter.shape)
    x= layers.Dropout(dropout)(xcenter)
    x=Dense(64)(x)
    x= layers.Dropout(dropout)(x)
    output_layer=Dense(2,activation='softmax')(x) #,activation='sigmoid'
    # output_layer=Dense(1, activation='sigmoid')(x)

    return Model(input,output_layer)

 
def featurefusion(cnn2df,cnn1df,ratio,eca):
    # cnn2df=layers.Conv1D(128,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(cnn2df)
    # cnn1df=layers.Conv1D(128,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(cnn1df)
    # cnn2df=layers.GlobalAveragePooling1D(name='average2df')(cnn2df)
    # cnn1df=layers.GlobalAveragePooling1D(name='average1df')(cnn1df)
 
    xc=layers.concatenate([cnn2df,cnn1df],axis=-1)
    print('xc',xc.shape)
    #eca和dense区别不大，dense更好;dense中ratio的确定;
    if eca==1:
        x=eca_block(xc, 1, 2)
    elif eca==0:
        # xc=layers.concatenate([cnn2df,cnn1df],axis=-1)
        c=xc.shape[-1] 
        x=Dense(c/ratio)(xc)
        x=Dense(c)(x)
        xattention=Activation('sigmoid')(x)
        x=multiply([xattention,xc])
        # whether spilt by cnn2d and cnn1d?
        x=Add()([xc,x])
    return x

def dualsparableseparatenorCnn2dcnn1d(inputshape2d,inputshape1d,cnn2dfilters,cnn1dfilters,multiscales,attentions,cnn1dattention=1,eca=1,fusionattention=0,dropout=0):
    '''
    attention:
    0,no attention
    1spatial attention
    2,channel attention
    3,channel and spatial attention
    '''
    '''
    cnn2dfilters,cnn1dfilters,multiscales,attentions are list
    
    '''
    input2d=keras.Input(shape=inputshape2d)
    input1d=keras.Input(shape=inputshape1d)
    x=layers.Conv2D(16,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(input2d)
    x=BatchNormalization()(x) 
    x=layers.ReLU()(x)
    x=layers.Conv2D(32,(3,3),strides=(1,1),padding='same',kernel_initializer='he_normal', use_bias=False)(x)
    x=BatchNormalization()(x) 
    x=layers.ReLU()(x)
    for i in range(len(cnn2dfilters)):
        print(i,cnn2dfilters[i])
        x=depthsepar2dattention(x,cnn2dfilters[i],multiscale=multiscales[i],attention=attentions[i])
        print('xdepthwise',x.shape)
    x2d=layers.GlobalAveragePooling2D()(x)
    print('x2d',x2d.shape)

    # x2d=layers.Conv1D(256,(3,),strides=(2,),padding='same',kernel_initializer='he_normal', use_bias=False)(x2d)
    # print('x2d',x2d.shape)
    # x2d=layers.GlobalAveragePooling1D()(x2d)
    x1d= cnn1d(input1d,cnn1dfilters,cnn1dattention)
    # x2d=Dense(256)(x2d)
   
    if fusionattention==1:
        x=featurefusion(cnn2df=x2d,cnn1df=x1d,ratio=8,eca=eca)
    else:
        x=layers.concatenate([x2d,x1d],axis=-1)
    x= layers.Dropout(dropout)(x)
    x=Dense(64)(x)
    x= layers.Dropout(dropout)(x)
    output_layer=Dense(2,activation='softmax')(x) #,activation='sigmoid'
    # output_layer=Dense(1, activation='sigmoid')(x)
    print('1111111111111111111111111111111111111111111111111111111')

    return Model([input2d,input1d],output_layer)

def dualsparableCnn2d(inputshape, cnn2dfilters, cnn1dfilters, multiscales, attentions,
                      cnn1dattention=0, fusionattention=0, dropout=0):
    """
    Build a dual-scale CNN model with spatial and channel attention.

    Args:
        inputshape (tuple): Input shape of the data.
        cnn2dfilters (list): List of 2D CNN filter sizes.
        cnn1dfilters (list): List of 1D CNN filter sizes.
        multiscales (list): List of scales for multiscale attention.
        attentions (list): List of attention types.
        cnn1dattention (int): Attention type for 1D CNN.
        dropout (float): Dropout rate.

    Returns:
        Model: Compiled Keras model.
    """
    # Input layer
    input = keras.Input(shape=inputshape)

    # First two convolutional layers
    x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal', use_bias=False)(input)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                       kernel_initializer='he_normal', use_bias=False)(x)

    # Apply depth-wise separable 2D attention layers
    for i in range(len(cnn2dfilters)):
        x = depthsepar2dattention(x, cnn2dfilters[i], multiscale=multiscales[i],
                                   attention=attentions[i])

    # Global Average Pooling and dense layer
    x2d = layers.GlobalAveragePooling2D()(x)
    # x2d = Dense(256)(x2d)
    x1d= cnn1d(input,cnn1dfilters,cnn1dattention)
    if fusionattention:
        x=featurefusion(cnn2df=x2d,cnn1df=x1d,ratio=8)
    else:
        x=layers.concatenate([x2d,x1d],axis=-1)

    # Dropout and dense layers
    x = layers.Dropout(dropout)(x)
    x = Dense(64)(x)
    x = layers.Dropout(dropout)(x)

    # Output layer
    output_layer = Dense(2, activation='softmax')(x)

    # Return compiled model
    return Model(input, output_layer)


# model=dualsparableCnn2d((11,11,15),[64,128,256],[32,64,128,256],[1,1,1],[1,1,1])
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()
# x=np.random.random((100,11,11,15))
# y=np.random.randint(2, size=(100, 2))
# model.fit(x,y,epochs=2)
# model.save('model.h5')













