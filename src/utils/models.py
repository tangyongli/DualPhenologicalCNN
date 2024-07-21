import numpy as np
import time
import random
import tensorflow as tf
from keras import backend as K
from keras import callbacks
import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import Conv3D,Reshape, LayerNormalization, multiply, Dense, Activation, Add, Flatten, Lambda, Concatenate, Conv1D, Conv2D,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation,DepthwiseConv2D,MaxPooling2D,GlobalAveragePooling2D,GlobalAveragePooling3D,GlobalAveragePooling1D
from tensorflow.keras.layers import Input,Dense,Dropout, Softmax, Flatten, Lambda, Concatenate
from tensorflow import expand_dims
from keras.models import Model
from tensorflow.keras.layers import LSTM,TimeDistributed
import math
import os


def channel_attention(inputs,ratio=8):
    avg_pool= layers.GlobalAveragePooling2D()(inputs)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    # print(avg_pool.shape,max_pool.shape)
    avg_pool=Reshape((1,1,avg_pool.shape[-1]))(avg_pool)
    max_pool = Reshape((1,1,max_pool.shape[-1]))(max_pool)
    channel =  avg_pool.shape[-1]  # 获取通道维度
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    avg_pool=shared_layer_one(avg_pool)
    avg_pool=shared_layer_two(avg_pool)
    max_pool= shared_layer_one(max_pool)
    max_pool= shared_layer_two(max_pool)
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return multiply([inputs, cbam_feature])

def spatial_attention(inputs, kernel_size=5):
    avg_pooling = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
    max_pooling = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
    concat = Concatenate(axis=-1)([avg_pooling, max_pooling])
    cbam_feature = Conv2D(filters=1, kernel_size=(kernel_size,kernel_size), strides=1, padding='same', activation='sigmoid',  use_bias=False)(concat)
    return multiply([inputs, cbam_feature])

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
    # [c,1]==>[c,1] 1D卷积输入3纬
    # [c,1]==>[c,1] 1D卷积输入3纬
    # x=GlobalAveragePooling2D()(inputs)
    x = layers.Reshape((in_channel, 1))(inputs) # (None, 256, 1)
    # print('xreshape',x.shape)
    
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    #x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
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



def conv2dbnblock(inputs,kernel_size,num_filters):
    x = layers.Conv2D(num_filters, kernel_size, strides=1,padding="same",kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization()(x) 
  
    return x



def convbnrelu(inputs,cbrm=1):
    x = Conv2D(16, (3, 3), activation='relu')(inputs)
    # x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    # x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    if cbrm:
        # x=channel_attention(x,ratio=8)
        x=spatial_attention(x)
    x=GlobalAveragePooling2D()(x)
    return x

def featurefusion(cnn2df,cnn1df,ratio,eca):
    xc=layers.concatenate([cnn2df,cnn1df],axis=-1)
    # print('xc',xc.shape)
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
######################## For dualcnn1d #################################
def single1D(inputs):
    inputheight,inputwidth,inputchannels=inputs.shape[1],inputs.shape[2],inputs.shape[-1]
    xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputchannels])(inputs)
    xcenter=Reshape((xcenter.shape[-1],1))(xcenter) # 13,1
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=(5), padding='valid')(xcenter)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    x = Conv1D(filters=32, kernel_size=(5),padding='valid')(x)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    x= Conv1D(filters=64, kernel_size=(3),padding='valid')(x)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    x= Conv1D(filters=128, kernel_size=(3),padding='valid')(x)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    output=GlobalAveragePooling1D()(x)
    print(output.shape)

    return output

# inputs=keras.Input(shape=(11,11,13))
# single1D(inputs).shape


def hycnn1d(inputshape,drop,fusion,single1):
    inputs= Input(shape=inputshape)
    mask_input = Input(shape=(9,9,26))
    inputheight,inputwidth,inputchannels=mask_input.shape[1],mask_input.shape[2],mask_input.shape[-1]
    mask=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputchannels])(mask_input)
    print('mask',mask.shape)

    xfloodinput=layers.Lambda(lambda x: x[...,0:13])(inputs*mask)
    print('xfloodinput',xfloodinput.shape)
    inputheight,inputwidth,inputchannels=xfloodinput.shape[1],xfloodinput.shape[2],xfloodinput.shape[-1]
    xpeakinput=layers.Lambda(lambda x: x[...,13:26])(inputs*mask)
    
    print( inputheight,inputwidth,inputchannels)
    if single1==1:
        x=single1D(xfloodinput)
    if single1==2:
        x=single1D(xpeakinput)
    if single1==0:
        x1=single1D(xfloodinput)
        x2=single1D(xpeakinput)
        if fusion:
            x=featurefusion(x1,x2,8,0)
        else:
            x=layers.Concatenate(axis=-1)([x1,x2])
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(drop)(x)
    dense_layer2 = Dense(units=128, activation='relu')(x)
    dense_layer2 = Dropout(drop)(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)
    model = Model(inputs=[inputs, mask_input],outputs=output_layer)
    return model

# hycnn1d(inputshape=(11,11,26),drop=0,fusion=1,single1=0).summary()
######################## For dualcnn2d #################################

def masked_global_average_pooling_2d(input_data, mask):
    """
    Performs masked global average pooling on a 2D input tensor.
    """
    masked_input = input_data * mask[..., 0:1]
    masked_sum = tf.reduce_sum(masked_input, axis=[1, 2])
    # print('masked_sum',masked_sum.shape) # masked_sum (None, 64)

    mask_sum = tf.reduce_sum(mask[...,0:1], axis=[1, 2])
    # print('mask_sum',mask_sum) #shape=(None, 1)
    mask_sum = tf.maximum(mask_sum, 1e-8)  
    return masked_sum / mask_sum
def single(input,cbrm,maskmissing):
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same',use_bias=False)(input)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    x = Conv2D(filters=32, kernel_size=(3, 3),padding='same',use_bias=False)(x)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    if cbrm:
        # x= channel_attention(x,ratio=8)
        x=spatial_attention(x)
    x= Conv2D(filters=64, kernel_size=(3, 3),padding='same',use_bias=False)(x)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    x= Conv2D(filters=128, kernel_size=(3, 3),padding='same',use_bias=False)(x)
    x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    
    # x=spatial_attention_lowhigh(low,high,kernel_size=7)
    # if filter==64:
    #     x = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='valid')(x)
    # x= Conv2D(filters=128, kernel_size=(3,3),padding='same',use_bias=False)(x)
    # x=BatchNormalization(axis=-1)(x) 
    # x = layers.ReLU()(x)
    # if maskmissing:
    #     x=masked_global_average_pooling_2d(x, mask)
    # else:
    x=GlobalAveragePooling2D()(x)
    # print('xmask',x.shape)
    # print(x.shape[0])
    # 调整 mask 的维度
    # mask1 = tf.reshape(mask, [tf.shape(x)[0], -1]) 
    # print(mask1.shape, 'mask')
    return x
def hycnn2d(inputshape,drop,cbrm,fusion,single1,maskmissing):
    inputs= Input(shape=inputshape)
    # mask_input = Input(shape=inputshape)  # 假设 mask 的通道数为 1
    # mask0=mask_input[...,0:13]
    # mask1=mask_input[...,13:26]
    xflood=layers.Lambda(lambda x: x[...,0:13])(inputs)
    xpeak=layers.Lambda(lambda x: x[...,13:26])(inputs)
    if single1==0:
        x=single(xflood,cbrm,maskmissing)
    if single1==1:
        x=single(xpeak,cbrm,maskmissing)
    if single1==2:
        x1=single(xflood,cbrm,maskmissing)
        x2=single(xpeak,cbrm,maskmissing)
        if fusion:
            x3=layers.Concatenate(axis=-1)([x1,x2])

            # x=featurefusion(x1,x2,8,0)
            # 计算每个样本的绝对值以用于有效性判断
            # abs_output_branch1 = tf.abs(x1)

            # # 计算权重：每行的绝对值归一化
            # weights_branch1 = abs_output_branch1 / (tf.reduce_sum(abs_output_branch1, axis=1, keepdims=True) + 1e-6)

            # # 加权融合
            # x = weights_branch1 * x1 + (1 - weights_branch1) * x2
            # x=layers.Concatenate(axis=-1)([x1,x2])
            # x=tf.keras.layers.Dense(, activation='relu'),
            is_x1_zero = tf.reduce_all(tf.equal(x1, 0), axis=1, keepdims=True)
            is_x2_zero = tf.reduce_all(tf.equal(x2, 0), axis=1, keepdims=True)
            print('is_x1_zero',is_x1_zero)

            normal_attention_weights = tf.keras.layers.Dense(1, activation='sigmoid')(x3)
            # attention_weights = tf.where(
            #     is_x1_zero,
            #     tf.zeros_like(normal_attention_weights),
            #     normal_attention_weights
            # )
            attention_weights = tf.where(
            is_x1_zero,  # 如果 is_x1_zero 为真
            tf.zeros_like(normal_attention_weights),  # 则 attention_weights 为全零
            tf.where(
                is_x2_zero,  # 否则，如果 is_x2_zero 为真
                tf.ones_like(normal_attention_weights),  # 则 attention_weights 为全一
                normal_attention_weights  # 否则，使用 normal_attention_weights
            )
        )

            inverse_attention_weights = 1 - attention_weights

            weighted_output_1 = x1 * attention_weights
            weighted_output_2 = x2 * inverse_attention_weights
            print('weighted_output_1',weighted_output_1)
            x = layers.Add()([weighted_output_1, weighted_output_2])

    
        else:
            x=layers.Concatenate(axis=-1)([x1,x2])
    ## fully connected layers
    
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(drop)(x)
    dense_layer2 = Dense(units=128, activation='relu')(x)
    dense_layer2 = Dropout(drop)(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)
    model = Model(inputs=[inputs], outputs=output_layer)
    return model
# hycnn2d(inputshape=(11,11,26),drop=0,cbrm=0,fusion=0,single1=2).summary()
# a=7
def hycnn3d(inputshape,drop,cbrm=1,filter=64):
    input_layer = Input(shape=inputshape)
    input_layer = expand_dims(input_layer, axis=-1)

    ## convolutional layers
    # pad_depth = pad_height = 1
    # pad_width = 0
    # # Pad the input layer along the depth and height dimensions
    # padded_input = tf.pad(input_layer, [[0, 0], [pad_depth, pad_depth], [pad_height, pad_height], [0, 0], [0, 0]])

    # Define the convolutional layer
   
    x= tf.keras.layers.Conv3D(filters=8*2, kernel_size=(3, 3, 7),  padding='valid')(input_layer)
    # x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    # conv_layer1 = Conv3D(filters=8*2, kernel_size=(3, 3, 7), activation='relu',padding='valid')(input_layer)
    x= Conv3D(filters=16*2, kernel_size=(3, 3, 5), padding='valid')(x)
    # x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)
    # x= Conv3D(filters=64*2, kernel_size=(3, 3, 3), activation='relu',padding='valid')(x)
    x=Conv3D(filters=32*2, kernel_size=(3, 3, 3), padding='valid')(x)
    x = layers.ReLU()(x)
    # x=Conv3D(filters=64*2, kernel_size=(3, 3, 3), padding='valid')(x)
    # x = layers.ReLU()(x)
    
    # x= Reshape((x_shape[1], x_shape[2], x_shape[3]*x_shape[4]))(x)
    # x= Conv3D(filters=64*2, kernel_size=(1, 1, 3), activation='relu',padding='valid')(x)
    # conv_layer35= Conv3D(filters=32*2, kernel_size=(5, 5, 3), activation='relu',padding='same')(conv_layer2)
    # conv_layer31= Conv3D(filters=32*2, kernel_size=(1, 1, 3), activation='relu',padding='same')(conv_layer2)
    # x=Concatenate(axis=-1)([conv_layer31,conv_layer33,conv_layer35])
    if cbrm:
        x= channel_attention(x,ratio=8)
        # print('x1',x.shape)
        x=spatial_attention(x)
        # print('x2',x.shape)
    x=Lambda(lambda x: K.mean(x, axis=-2, keepdims=False))(x)
    if filter==64:
        x = Conv2D(filters=64, kernel_size=(3,3),padding='valid')(x)
        # x=BatchNormalization(axis=-1)(x) 
        x = layers.ReLU()(x)

    x = Conv2D(filters=128, kernel_size=(3,3),padding='valid')(x)
    # x=BatchNormalization(axis=-1)(x) 
    x = layers.ReLU()(x)

    x=GlobalAveragePooling2D()(x)
    
    flatten_layer = Flatten()(x)

    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(drop)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(drop)(dense_layer2)
    # if eca==1:
    # x=eca_block(dense_layer2 , b=1, gama=2)
    
    output_layer = Dense(units=2, activation='softmax')(x)
    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# hycnn3d((11,11,13),drop=0.4,cbrm=0,filter=128,eca=1).summary()


class SelfAttention(layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units

        self.query = layers.Dense(units)
        self.key = layers.Dense(units)
        self.value = layers.Dense(units)

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))
        output = tf.matmul(attention_weights, value)

        return output
    


def self_attention(inputs, ratio):
    units=inputs.shape[-1]//ratio
    query = layers.Dense(units)(inputs)
    key = layers.Dense(units)(inputs)
    value = layers.Dense(units)(inputs)

    # attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))
    attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(units, tf.float32)), axis=-1)
    # if mask is not None:
    #         scores += (mask * -1e9)
    #     attention_weights = tf.nn.softmax(scores, axis=-1)
    #     context = tf.matmul(attention_weights, value)
    output = tf.matmul(attention_weights, value)
    print('att',output.shape)

    return output


def lstm(input_channels,timestep, n_classes, drop,inputs=None):
    """
    LSTM-based model for remote sensing image classification
    """
    
    if inputs is None:
        inputs = tf.keras.Input(shape=(timestep,input_channels))

    # 定义 LSTM 层
    lstm_layer1 = tf.keras.layers.LSTM(units=64, activation='relu', return_sequences=True)
    lstm_layer2 = tf.keras.layers.LSTM(units=64, activation='relu', return_sequences=True)
    lstm_layer3 = tf.keras.layers.LSTM(units=64, activation='relu', return_sequences=True)
    lstm_layer4 = tf.keras.layers.LSTM(units=64, activation='relu', return_sequences=False)
    # lstm_layer5 = tf.keras.layers.LSTM(units=256, activation='relu', return_sequences=False)
    drop= Dropout(0)

    # 定义全连接层
    fc1= Dense(units=128, activation='relu')
    
    fc2 = tf.keras.layers.Dense(n_classes, activation=None)
    

    # 正向传播
    x = lstm_layer1(inputs)
    x = lstm_layer2(x)
    x = lstm_layer3(x)
    x = lstm_layer4(x)
    # x = lstm_layer5(x)
   
    x = fc1(x)
    x=Dropout(0.4)(x)
    x=fc2(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model


# m=lstm(input_channels=13,timestep=10, n_classes=2, patch_size=11, dilation=1, inputs=None)
# m.summary()







import tensorflow as tf

def HamidaEtAl(input_channels, n_classes, patch_size=11, dilation=1, inputs=None):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    
    if inputs is None:
        inputs = tf.keras.Input(shape=(None, None, None, input_channels))

    # Define convolutional layers
    conv1 = tf.keras.layers.Conv3D(20, (3, 3, 3), strides=(1, 1, 1), dilation_rate=(dilation, 1, 1), padding='same', activation='relu')
    pool1 = tf.keras.layers.Conv3D(20, (1, 1, 3), strides=(1, 1, 2), dilation_rate=(dilation, 1, 1), padding='same', activation='relu')
    conv2 = tf.keras.layers.Conv3D(35, (3, 3, 3), strides=(1, 1, 1), dilation_rate=(dilation, 1, 1), padding='same', activation='relu')
    pool2 = tf.keras.layers.Conv3D(35, (1, 1, 3), strides=(1, 1, 2), dilation_rate=(dilation, 1, 1), padding='same', activation='relu')
    conv3 = tf.keras.layers.Conv3D(35, (1, 1, 3), strides=(1, 1, 1), dilation_rate=(dilation, 1, 1), padding='same', activation='relu')
    conv4 = tf.keras.layers.Conv3D(35, (1, 1, 2), strides=(1, 1, 2), dilation_rate=(dilation, 1, 1), padding='same', activation='relu')
    flatten = tf.keras.layers.Flatten()
    fc = tf.keras.layers.Dense(n_classes, activation=None)

    # Forward pass
    x = conv1(inputs)
    x = pool1(x)
    x = conv2(x)
    x = pool2(x)
    x = conv3(x)
    x = conv4(x)
    x = flatten(x)
    x = fc(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

# Example usage:
# model = HamidaEtAl(input_channels=10, n_classes=10)
# model.summary()
# inputshape=(11,11,15,1)
# inputs=keras.Input(shape=inputshape)
# model=HamidaEtAl(input_channels=15, n_classes=2, patch_size=11, dilation=1, inputs=inputs)
# model.summary()
















