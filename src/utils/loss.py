
from keras import backend as K
import numpy as np
import tensorflow as tf





def weight_loss(y_true,y_pred,mask,is_mask=False):
    '''
    y_true: 真实标签,one-hot编码。
    y_pred: 预测概率,shape=(64,2),tf.reduce_sum(y_pred,axis=1)=1
    class_weights 是一个字典,根据函数compute_class_weights得到的权重
    
    mask: 布尔数组, shape is (batch_size, height, width, channels)

    '''
    class_weights = tf.constant([0.8361688720856963,1.2436738519212747])

    # 计算每个样本的交叉熵损失
    # y precit是概率值
    ylogit=tf.math.log(tf.clip_by_value(y_pred, 1e-10, 1.0))
    # print('ylogit',y_true * ylogit) #(64, 2)
    y=np.argmax(y_true, axis=-1)
    weights = tf.gather(class_weights, y)
    cross_entropy = -tf.reduce_sum(y_true * ylogit,axis=1)*weights # 64,1
    meanloss = tf.reduce_mean(cross_entropy)
    if is_mask:
        #计算每个样本的有效像素数量 
        valid_pixels = tf.reduce_sum(mask, axis=[1, 2, 3]) 
        # 计算每个样本中输入batch大小 heightxwidthxchannels
        valid_ratio = valid_pixels / tf.cast(tf.size(mask[0:1,...]), tf.float32)
        # 一个batch中的像素有效率
        valid_ratio = tf.reduce_mean(valid_ratio)
        return meanloss*valid_ratio

    return meanloss


def compute_class_weights(labels):
    '''
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data?hl=zh-cn#%E7%B1%BB%E6%9D%83%E9%87%8D
    
    '''
    labels=np.argmax(labels,axis=-1)
    # Count number of postive and negative bags.
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count
    print(negative_count, positive_count, total_count)

    return {
        0: (1 / negative_count) * (total_count/2),
        1: (1 / positive_count) * (total_count/2),
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


def binary_weighted_cross_entropy(y_true, y_pred, mask, weight):
    """ 适用于二类分类，解决类别不平衡问题 
    y_true: one-hot编码,shape is (batch_size, 2)
    y_predict: 预测概率,shape is (batch_size, 2)
    weight:负类的样本/正类的样本
    mask: 布尔数组, shape is (batch_size, height, width, channels)
    """
    def convert_to_logits(y_pred):
        """ 将经过sigmoid变换后的概率值转回logit """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

  
    # 计算每个样本的有效像素数量 
    valid_pixels = tf.reduce_sum(mask, axis=[1, 2, 3]) 

    # 计算每个样本中输入batch大小 heightxwidthxchannels
    total_pixels =tf.cast(tf.size(mask[0:1,...]), tf.float32) 
    valid_ratio = valid_pixels / total_pixels

    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weight)
    loss=tf.reduce_mean(loss)
    return loss*valid_ratio