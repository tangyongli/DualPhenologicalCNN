

from keras import backend as K

def tversky(y_true, y_pred,smooth=0):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred,smooth):
    return 1 - tversky(y_true,y_pred,smooth)
def focal_tversky_loss(y_true, y_pred, smooth=0,gamma=0.75):
    tv = tversky(y_true, y_pred,smooth)
    return K.pow((1 - tv), gamma)


import numpy as np
import tensorflow as tf
def compute_class_weights(labels):
    labels=np.argmax(labels,axis=-1)
    # Count number of postive and negative bags.
    negative_count = len(np.where(labels == 1)[0])
    positive_count = len(np.where(labels == 0)[0])
    total_count = negative_count + positive_count
    print(negative_count, positive_count, total_count)
    # Build class weight dictionary.
    '''
    
    1677 1530 3207
{1: 0.9561717352415027, 0: 1.0480392156862743}
    '''
    return {
        1: (1 / negative_count) * (total_count/2),
        0: (1 / positive_count) * (total_count/2),
    }

def compute_mclass_weights(labels):
    labels=np.argmax(labels,axis=-1)
    # Count number of postive and negative bags.
    othercrops = len(np.where(labels == 1)[0])
    rice = len(np.where(labels == 0)[0])
    ve=len(np.where(labels == 2)[0])
    water=len(np.where(labels == 3)[0])
    urban=len(np.where(labels == 4)[0])
    total_count = rice+othercrops+ve+water+urban
    # print(negative_count, positive_count, total_count)
    # Build class weight dictionary.
    '''
    
    1677 1530 3207
{1: 0.9561717352415027, 0: 1.0480392156862743}
    '''
    return {
       
        0: (1 / rice) * (total_count/2),
        1: (1 / othercrops) * (total_count/2),
        2: (1 / ve) * (total_count/2),
        3: (1 / water) * (total_count/2),
        4: (1 / urban) * (total_count/2),
    }
def binary_weighted_cross_entropy(y_true, y_pred, mask,beta=1.):
    """ 适用于二类分类，解决类别不平衡问题 """
    def convert_to_logits(y_pred):
        """ 将经过sigmoid变换后的概率值转回logit """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))
        
    # 计算每个样本的有效像素数量
    valid_pixels = tf.reduce_sum(mask, axis=[0,1, 2, 3]) 
    # 计算有效像素的比例
    valid_ratio = valid_pixels / tf.cast(tf.size(mask), tf.float32)
    # print('valid_ratio',valid_ratio)

    y_pred = convert_to_logits(y_pred)
    # Calculate the weighted cross-entropy loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)
    # print('lossbinarty',loss.shape)
    return tf.reduce_mean(loss)* valid_ratio 
def nomask_binary_weighted_cross_entropy(y_true, y_pred, beta=1.):
    """ 适用于二类分类，解决类别不平衡问题 """
    def convert_to_logits(y_pred):
        """ 将经过sigmoid变换后的概率值转回logit """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))
    # print(maskingvalue)
    
    y_pred = convert_to_logits(y_pred)
    # Calculate the weighted cross-entropy loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)
    # print('lossbinarty',loss.shape)
    return tf.reduce_mean(loss)