
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
def binary_weighted_cross_entropy(y_true, y_pred, beta=1.):
    """ 适用于二类分类，解决类别不平衡问题 """
    def convert_to_logits(y_pred):
        """ 将经过sigmoid变换后的概率值转回logit """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))

    y_pred = convert_to_logits(y_pred)
    #  # Apply sigmoid activation to y_pred if it hasn't been applied
    # y_pred = tf.nn.sigmoid(y_pred)
    
    # Calculate the weighted cross-entropy loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=beta)
    # loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
    return tf.reduce_mean(loss)
