# def binary_weighted_cross_entropy(y_true, y_pred, beta=1.):
#     """ 适用于二类分类，解决类别不平衡问题 """
#     def convert_to_logits(y_pred):
#         """ 将经过sigmoid变换后的概率值转回logit """
#         # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
#         return tf.math.log(y_pred / (1 - y_pred))

#     y_pred = convert_to_logits(y_pred)
#     #  # Apply sigmoid activation to y_pred if it hasn't been applied
#     # y_pred = tf.nn.sigmoid(y_pred)
    
#     # Calculate the weighted cross-entropy loss
#     loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)
#     # loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
#     return tf.reduce_mean(loss)
#定义数据增强函数
import tensorflow as tf
from cfgs import *
# def augment_channel(img_channel,seed):
#     img_channel = tf.image.random_flip_left_right(img_channel,seed=seed)
#     img_channel = tf.image.random_flip_up_down(img_channel,seed=seed)
#     # img_channel = tf.image.stateless_rot90(img_channel, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32,seed=seed))  # 随机旋转90度
#     # img_channel = tf.image.stateless_random_contrast(img_channel, lower=0.8, upper=1.2,seed=seed)
#     # img_channel = tf.image.random_brightness(img_channel, max_delta=0.2) #输入的只能有3个通道
#     # img_channel = tf.image.random_saturation(img_channel, lower=0.8, upper=1.2)
# def augment_data(img, seed=None):
#     # Reshape为(11, 11, 32)
#     img=tf.transpose(img,(1,2,0,3))
#     print(img.shape)
#     img = tf.reshape(img, (patch_size, patch_size, channels))
#     print(img.shape)
#     imgagu=augment_channel(img,seed) # 进行数据增强,随机种子必须是()两个数，是不是每epoch一个随机数？
#     print( imgagu.shape,'img')
#     img = tf.reshape(imgagu, (11, 11, 4,8))
#     img=tf.transpose(img,(2,0,1,3))
#     # label=tf.expand_dims(label,axis=0)
#     # label=tf.expand_dims(label,axis=1)
#     # labelagu=augment_channel(label) # 1 1 2
#     # print(labelagu.shape)
#     # label=tf.squeeze(labelagu,axis=0)
#     # label=tf.squeeze(label,axis=0)
#     # print(label.shape) #2
#     return img

from keras.preprocessing.image import ImageDataGenerator
'''

zoom_range: 可以是一个浮点数，也可以是包含两个浮点数的列表或元组。如果是一个浮点数，它指定缩放范围为 [1 - zoom_range, 1 + zoom_range]。如果是包含两个浮点数的列表或元组，它们分别表示缩放的上下限。例如，zoom_range=(0.2, 0.2)表示将图像在水平和垂直方向上缩放 20%。
这样的缩放操作可以使模型对不同尺寸的物体更具鲁棒性，并提供一定程度的平移不变性。在图像分类等任务中，通过引入随机缩放，模型可以学到物体在不同尺寸下的表征，从而提高泛化能力。

'''
img_data_gen_args = dict(
    rotation_range=45, 
                    #  width_shift_range=0.3,
                    #  height_shift_range=0.3,
                    #  shear_range=0.5,
                    #  zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True)
                    #  fill_mode='reflect')

mask_data_gen_args = dict(
    rotation_range=45,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                    #  shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True)
                    #  fill_mode='reflect',
                    #  preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 


#image_data_generator.fit(X_train, augment=True, seed=seed)

batch_size= 32
#mask_data_generator.fit(y_train, augment=True, seed=seed)

# valid_mask_generator = mask_data_generator.flow(y_test, seed=seed, batch_size=batch_size)  #Default batch size 32, if not specified here
import numpy as np
# Custom generator for ytrain without augmentation
def batch_generator(ytrain, batch_size):
    for i in range(0, len(ytrain), batch_size):
      yield ytrain[i:i + batch_size]



def dataagument( xtrain,ytrain,seed):
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    # flow select data randomly
    train_generator = image_data_generator.flow(xtrain, ytrain,seed=seed, batch_size=batch_size)
    print( train_generator)
    for img, mask in  train_generator:
        # img= np.reshape(img, (img.shape[0], img.shape[1],img.shape[2], times,-1))
        # img=np.transpose(img,(0,3,1,2,4))
        # if img.shape[1]==1:
        #     img=np.squeeze(img,axis=1)
        #     # print(img.shape)
        
        yield (img, mask)
# Validation data generator (without augmentation)
def val_data_generator(xval, yval, batch_size):
    for i in range(0, len(xval), batch_size):
        yield (xval[i:i + batch_size], yval[i:i + batch_size])


# x = image_generator.next()
# y = mask_generator.next()
# for i in range(0,1):
#         image = x[i]
#         mask = y[i]
#         plt.subplot(1,2,1)
#         plt.imshow(image[:,:,0], cmap='gray')
#         plt.subplot(1,2,2)
#         plt.imshow(mask[:,:,0])
#         plt.show()


# steps_per_epoch = 3*(len(X_train))//batch_size


# history = model.fit_generator(my_generator, validation_data=validation_datagen, 
#                     steps_per_epoch=steps_per_epoch, 
#                     validation_steps=steps_per_epoch, epochs=25)

