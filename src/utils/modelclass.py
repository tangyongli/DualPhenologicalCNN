import numpy as np
import time
import random
import tensorflow as tf
from keras import backend as K
from keras import callbacks

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from semantic_segmentation.dataset.transformer_ImageDataGenerator import dataagument,val_data_generator#augment_data
from semantic_segmentation.dataset import transformer
from semantic_segmentation.dataset.transformer import  Compose,RandomRotation,RandomContrast,RandomScale,RandomHorizontalFlip,RandomVerticalFlip
from keras.utils import to_categorical
from utils.model.models import *
# from model.models import singlebranch
from keras.layers import Input, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling1D
from utils.model.loss import *
from utils.plot import *
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


class DepthSeparable2DSpatialAttention(keras.layers.Layer):
    def __init__(self, filters):
        super(DepthSeparable2DSpatialAttention, self).__init__()
        self.depthwise_conv = keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')
        self.filters = filters
        self.pointwise_conv = keras.layers.Conv2D(self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
        self.batch_norm = keras.layers.BatchNormalization()
        self.spatial_attention = self.spatial_attention_lowhigh
       
        self.relu=keras.layers.ReLU()
        self.kernel_size=5
        self.cbam_feature=layers.Conv2D(filters=1, kernel_size=self.kernel_size, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.channeladjust=keras.layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
    def build(self, input_shape):
        # Dynamically set the size of the second channel reduction layer based on the concatenated output
        # Assuming the concatenated output is of shape (batch_size, features), we use features to set units
        channel_size = input_shape[-1]  # Adjust this according to the actual structure of your input
        print('channel_size',channel_size)
        self.low_channels_adjusted = keras.layers.Conv2D(filters=channel_size, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
        super(DepthSeparable2DSpatialAttention, self).build(input_shape)  # Finalize the build proces

    def spatial_attention_lowhigh(self, low, high):
    
        concat = keras.layers.Concatenate(axis=-1)([
            keras.layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(low),
            keras.layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(high)
        ])
        cbam_feature = self.cbam_feature(concat)
        low_channels_adjusted = self.channeladjust(low)
        return keras.layers.Multiply()([low_channels_adjusted, cbam_feature])

    def call(self, input):
        x = self.depthwise_conv(input) # 32
        x = self.pointwise_conv(x) #64
        x = self.batch_norm(x)
        tf.print('input',input.shape)
        x = self.spatial_attention(input, x)   
        tf.print('xatte',x.shape)
        input= self.channeladjust(input)
        x = layers.Add()([input, x]) 
        x = self.relu(x)
        return x


class cnn2d(keras.Model):
    def __init__(self,l2):
        super(cnn2d, self).__init__()
        self.conv1 = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
        self.conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)

        self.depthsepar2dspatialattention1 = DepthSeparable2DSpatialAttention(filters=64)
        self.depthsepar2dspatialattention2 = DepthSeparable2DSpatialAttention(filters=128)
        self.depthsepar2dspatialattention3 = DepthSeparable2DSpatialAttention(filters=256)
        self.GlobalAveragePooling2D=GlobalAveragePooling2D()
        self.regularizers=keras.regularizers.l2(l=l2)
        self.Dense=Dense(256,kernel_regularizer= self.regularizers)

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.depthsepar2dspatialattention1(x)  # Corrected: Call with parentheses and pass 'x'
        tf.print('x1',x.shape)
        x = self.depthsepar2dspatialattention2(x)  # Corrected: Call with parentheses and pass 'x'
        tf.print('x2',x.shape)
        x = self.depthsepar2dspatialattention3(x)  # Corrected: Call with parentheses and pass 'x'
        x=self.GlobalAveragePooling2D(x)
        x=self.Dense(x)
        return x




class cnn1d(keras.layers.Layer):
    def __init__(self):
        super(cnn1d, self).__init__()
        self.conv1 = layers.Conv1D(32, (3,), strides=(2,), padding='same', kernel_initializer='he_normal', activation='relu',use_bias=False)
        self.conv2 = layers.Conv1D(64, (3,), strides=(2,), padding='same', kernel_initializer='he_normal', activation='relu',use_bias=False)
        self.conv3 = layers.Conv1D(128, (3,), strides=(2,), padding='same', kernel_initializer='he_normal', activation='relu',use_bias=False)
        self.conv4 = layers.Conv1D(256, (3,), strides=(2,), padding='same', kernel_initializer='he_normal',activation='relu', use_bias=False)

    def call(self,input):
        inputheight,inputwidth,inputschannels=input.shape[1],input.shape[2],input.shape[3]
        xcenter=layers.Lambda(lambda x: x[...,inputheight//2:inputheight//2+1,inputwidth//2:inputwidth//2+1,0:inputschannels])(input)
        # print('xcentershape',xcenter.shape)
        xcenter=Reshape((xcenter.shape[-1],1))(xcenter)
        # print('xcentershape',xcenter.shape)
        xcenter=self.conv1(xcenter)
        xcenter=self.conv2(xcenter)
        xcenter=self.conv3(xcenter)
        xcenter=self.conv4(xcenter)
        xcenter=layers.GlobalAveragePooling1D(name='average')(xcenter)
        # print('xcentershape',xcenter.shape)
        xcenter=Reshape((xcenter.shape[-1],))(xcenter)#256,
        # print('xcenter',xcenter.shape)
        return xcenter

class dual(keras.layers.Layer):
    def __init__(self, nc, dropout, l2):
        super(dual, self).__init__()
        
        self.nc = nc 
        self.dropout = dropout
        self.l2=l2
        # Define variables outside the call method
        self.regularizer = keras.regularizers.l2(l=l2)
        self.dense1 = keras.layers.Dense(64, kernel_regularizer=self.regularizer)
        self.dropout_layer = keras.layers.Dropout(dropout)
        self.output_layer = keras.layers.Dense(nc, activation='softmax', kernel_regularizer=self.regularizer)
        '''
       
        1.the __init__ method should only be used to set up initial parameters and layers but not for processing data. don't call the function in the call method,otherwise the error is 'index beyond range';
 
        2.ValueError: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function
        explain gpt4:when TensorFlow detects the creation of new variables during subsequent calls to a tf.function or within a model's call method, which is expected to be executed in a graph mode where variables should be created only once at the model's build time, not during execution
        how to solve?  
        ①.create variables or layer in __init__ method,in call method without introducing new variables or layers during the call.
        ②. in 1, if variable or layers has parmters, make sure to create them in __init__ method.
        ③. in 1, no paramters variables or layers, such as reshape, Add,concatention,Multipy,you can use it directly in call method.
        
        3. __init__method 
        self.nc=nc, self.nc is the properities of class,and the value is assigned by def in __init__(self, nc, dropout, l2).add()
        if don't use self.nc to define the properity, you can not use it in call method.

        4. build method?

        

        '''
        self.cnn1d = cnn1d() # not cnn1d(input) input should passed to call method for processing data.
        self.cnn2d = cnn2d(l2)
        self.channelreduction=keras.layers.Dense(512// 8, kernel_regularizer=self.regularizer)
        self.channelrecover=keras.layers.Dense(512, kernel_regularizer=self.regularizer)
        self.activation=keras.layers.Activation('sigmoid')
    # how to serialize the model when save model
    def get_config(self):
        config = super().get_config()
        config.update({
            "nc": self.nc,
            "dropout": self.dropout,
            "l2":self.l2
        })
        return config
        

    def attention(self, input):
        # c = input.shape[-1]
        tf.print('input',input.shape)
        x = self.channelreduction(input)
        x =self.channelrecover(x)
        tf.print('x',x.shape)
        xattention = self.activation(x)
        x = keras.layers.Multiply()([xattention, input])
        return x

    def call(self,input):
        x = layers.concatenate([self.cnn1d(input), self.cnn2d(input)])
        tf.print('x',x.shape)
        x = self.attention(x)
        x = self.dropout_layer(x)
        x = self.dense1(x)
        x = self.dropout_layer(x)
        output = self.output_layer(x)
        return output

    def summary(self):
        input_shape = (11, 11, 15)  # Dummy input shape
        x = keras.Input(shape=input_shape)
        model = keras.Model(inputs=x, outputs=self.call(x))
        return model.summary()

        
# inputs = tf.random.normal(shape=(1,11, 11, 15)) #tf.keras.Input(shape=(11, 11, 3))

# # input=keras.Input(shape=(11,11,15))
# output=dual(2,0,0)(inputs)
# print(output.shape)
# Model(inputs=inputs,outputs=output)
inputs = Input(shape=(11, 11, 15))

dualinstance= dual(2, 0,0)
dualinstance.summary()
output = dualinstance(inputs)
print(inputs,output)
model = Model(inputs=inputs, outputs=output)
print(model.summary())


input1 = tf.random.normal(shape=(50,11, 11, 15))
y=np.random.randint(2, size=(50, ))
y=to_categorical(y,num_classes=2)
model.compile(optimizer='adam' ,loss= 'categorical_crossentropy', metrics=['accuracy'])
model.fit(input1,y,epochs=1)
# model.save('dualmodel.h5')
# # failed to load model
# keras.models.load_model('dualmodel.h5',custom_objects={"nc":2,"dropout":0,"l2": 0})

# Found unexpected instance while processing input tensors for keras functional model. Expecting KerasTensor which is from tf.keras.Input() or output from keras layer call() method.




        












class DepthSeparable2DSpatialAttention(keras.layers.Layer):
        def __init__(self, filters):
            super(DepthSeparable2DSpatialAttention, self).__init__()
            self.depthwise_conv = keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')
            self.pointwise_conv = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
            self.batch_norm = keras.layers.BatchNormalization()
            self.filters = filters

        def spatial_attention_lowhigh(self, low, high):
            kernel_size = 5
            concat = keras.layers.Concatenate(axis=-1)([
                keras.layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(low),
                keras.layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(high)
            ])
            cbam_feature = keras.layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
            low_channels_adjusted = keras.layers.Conv2D(filters=high.shape[-1], kernel_size=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(low)
            return keras.layers.Multiply()([low_channels_adjusted, cbam_feature])

        def call(self, inputs, high):
            x = self.depthwise_conv(inputs)
            x = self.pointwise_conv(x)
            x = self.batch_norm(x)
            x = self.spatial_attention_lowhigh(inputs, high)
            x = layers.Add()([inputs, x])
            x = layers.ReLU()(x)
            return x

    class cnn2d(keras.Model):
        def __init__(self):
            super(cnn2d, self).__init__()
            self.conv1 = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
            self.conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)
            self.depthsepar2dspatialattention64 = DepthSeparable2DSpatialAttention(64)
            self.depthsepar2dspatialattention128 = DepthSeparable2DSpatialAttention(128)
            self.depthsepar2dspatialattention256 = DepthSeparable2DSpatialAttention(256)

        def call(self, inputs):
            x = self.conv1(inputs)
            x = self.conv2(x)
            x = self.depthsepar2dspatialattention64(x, x)
            x = self.depthsepar2dspatialattention128(x, x)
            x = self.depthsepar2dspatialattention256(x, x)
            return x
