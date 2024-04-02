
import numpy as np
import random
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
class RandomRotation:
    def __init__(self, p=0.5):
        
        self.p = p
    def __call__(self, data_numpy, label_numpy):
        degree =np.random.randint(5,45)
        # if random.random() < self.p:
        rows, cols, _ = data_numpy.shape
        # print(type(data_numpy))
        mat = cv2.getRotationMatrix2D(
            ((cols-1)/2.0, (rows-1)/2.0), degree, 1)
        data_numpy = cv2.warpAffine(data_numpy, mat, (cols, rows))
            # label_numpy = cv2.warpAffine(label_numpy, mat, (cols, rows))
        return data_numpy, label_numpy
        # data_numpy = tf.reshape(data_numpy, (data_numpy.shape[0] , data_numpy.shape[1], 2,16))
        # data_numpy=tf.transpose(data_numpy,(2,0,1,3))
class RandomRotationAll:
    def __init__(self,  p=0.5):
       
        self.p = p
    def rotate_data_numpy(self, data_numpy, label_numpy,degree=np.random.randint(5,45)):
        degree = np.random.randint(5, 45)
        # print('degree',degree) # 这样设置degree才是每个patch都不一样

        # if random.random() < self.p:
        rows, cols, _ = data_numpy.shape
        mat = cv2.getRotationMatrix2D(
            ((cols-1)/2.0, (rows-1)/2.0), degree, 1)
        data_numpy = cv2.warpAffine(data_numpy, mat, (cols, rows))
            # print('before', data_numpy.shape)
            # label_numpy = cv2.warpAffine(label_numpy, mat, (cols, rows))
        # data_numpy = tf.reshape(data_numpy, (data_numpy.shape[0] , data_numpy.shape[1], 2,16))
        # data_numpy=tf.transpose(data_numpy,(2,0,1,3))
        # print('after', data_numpy.shape)
        return  data_numpy,label_numpy
    def __call__(self, data_numpy, label_numpy):
      
        # Apply rotation to each data_numpy in the batch
        data_numpy_rotated, label_numpy_rotated = zip(*[self.rotate_data_numpy(data, label) for data, label in zip(data_numpy, label_numpy)])

        return np.array(data_numpy_rotated), np.array(label_numpy_rotated)

# class RandomContrast(object):
#     """ Random Contrast """
    
#     def __init__(self, contrast=0.4,p=0.5):
#         self.contrast = contrast
#         self.p=p

#     def __call__(self, data_numpy,label):
#         # [] specifies the shape of the random value to be generated. In this case, it's an empty shape, which means a scalar value will be generated
#         #随机生成0-1+contrast之间的数
#         s = tf.random.uniform([], maxval=1 + self.contrast, dtype=tf.float32)
#         print('tensor s',s)
#         mean = tf.reduce_mean(data_numpy, axis=(0, 1))
        
#         data_numpy=((data_numpy - mean) * s + mean)#.astype(np.float32)
#             # print('DATA',data_numpy)
#         # data_numpy = tf.reshape(data_numpy, (data_numpy.shape[0] , data_numpy.shape[1], 2,16))
#         # data_numpy=tf.transpose(data_numpy,(2,0,1,3))
#         # print(data_numpy.shape)

#         return data_numpy,label

class RandomContrast(object):
    """ Random Contrast """
    
    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample,label):
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = np.mean(sample, axis=(0, 1))
        
        return ((sample - mean) * s + mean),label





class RandomChannelDrop(object):
    """ Random Channel Drop """
    def __init__(self, min_n_drop=1, max_n_drop=3):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, data_numpy,label):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(data_numpy.shape[-1]), size=n_channels, replace=False)
        # print(channels,'channels')

        for c in channels:
            data_numpy[ :,:,c:c+1] = 0        
        return data_numpy,label 


class RandomBrightness(object):
    """ Random Brightness """
    
    def __init__(self, brightness=0.6,p=0.5):
        self.brightness = brightness
        self.p=p

    def __call__(self, data_numpy,label):
        # if random.random() < self.p:
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
    # print('sa',s)
        img = data_numpy * s
        # print('img',img.shape)
        # data_numpy = tf.reshape(img, (11,11,2,16))
        # data_numpy=tf.transpose(data_numpy,(2,0,1,3))
        
        return img,label
class ToNumpy:
    """ Convert tensorflow-tensor to numpy-array """

    def __call__(self, data_tensor, label_tensor):
        return data_tensor.eval(), label_tensor.eval()




class ToNumpy:
    """Convert TensorFlow tensor to NumPy array"""

    def __call__(self, data_tensor, label_tensor):
        return data_tensor.numpy(), label_tensor.numpy()

# class RandomHorizontalFlip:
#     """Randomly flip the TensorFlow tensors at the horizontal axis"""

#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, data, label):
#         # if random.random() < self.p:
#         data_numpy = tf.image.flip_left_right(data)
#             # label = tf.image.flip_left_right(label)
#         # data_numpy = tf.reshape(data, (11,11,2,16))
#         # data_numpy=tf.transpose(data_numpy,(2,0,1,3))
#         return data_numpy, label
    
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img, label):
        if random.random() < 0.5:
            return  np.fliplr(img),label
        return img,label
class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    # np.flipud(img) Flip array in the up/down direction.
    def __call__(self, img,label):
        if random.random() < 0.5:
            # print(img.shape)
            flipped_img = np.flipud(img)  # Transpose the array
            print(flipped_img.shape)
            return flipped_img, label
        return img,label

# class RandomVerticalFlip:
#     """Randomly flip the TensorFlow tensors at the vertical axis"""

#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, data, label):
#         # if random.random() < self.p:
#         data_numpy = tf.image.flip_up_down(data)
#             # label = tf.image.flip_up_down(label)
#         # data_numpy = tf.reshape(data, (11,11,2,16))
#         # data_numpy=tf.transpose(data_numpy,(2,0,1,3))
#         return data_numpy, label

class RandomScale:
    """ Randomly scale the numpy-arrays """

    def __init__(self, scale_range=(0.9, 1.1), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        # if random.random() < self.p:
        scale =np.random.random() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        # print('scale',scale)
        # img_h, img_w, _ = data_numpy.shape
        img_h, img_w=11,11
        M_rotate = cv2.getRotationMatrix2D(
            (img_w / 2, img_h / 2), 0, scale)
        # print(M_rotate)
        data_numpy = cv2.warpAffine(data_numpy, M_rotate, (img_w, img_h))
        # print('rescale',data_numpy)
        # label_numpy = cv2.warpAffine(
        #     label_numpy, M_rotate, (img_w, img_h), flags=cv2.INTER_NEAREST)
        # label_numpy = np.expand_dims(
        #     label_numpy, axis=-1) if label_numpy.ndim == 2 else label_numpy

        return data_numpy, label_numpy




    
class Compose:
    """ Compose multi augmentation methods"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self,data,label):
    #     for _t in self.transforms:
    #         data, label = _t(data, label)
    #     return data, label
        for transform in self.transforms:
            result = transform(data, label) #this is a common convention in Python to indicate that the variable is intended to be used as a throwaway variable and its value won't be used inside the loop.
            print('result',result)
            if result is not None:
                data, label = result
        return data, label

class RandomApply:
    """ Randomly apply augmentation methods """

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, data, label):
        for _t in self.transforms:
            if random.random()<self.p:
                data, label = _t(data, label)
        return data, label
# x=np.random.random((11,11,3))
# y=np.random.random((11,11,3))
# fig, axes = plt.subplots(3, 3, figsize=(10, 10))
# # Display 3x3 grid of rotated images
# for i in range(3):
#     for j in range(3):
#         x1,y1=RandomScale()(x,y)
#         print(x1.shape,y1.shape)
#         axes[i, j].imshow(x1[..., 0], cmap='gray')  # Assuming the image is grayscale, adjust colormap if needed
#         axes[i, j].axis('off')

# plt.show()
  
# def RandomRotation(data_numpy, label_numpy,p=0.5):
#     # data_numpy =  data_numpy.numpy() #AttributeError: 'Tensor' object has no attribute 'numpy'
#     degree=np.random.randint(5,45)
#     # degree=np.random.random_integers(5,45)
#     if random.random() < p:
#             rows, cols, _ = data_numpy.shape
#             print(rows,cols)
#             mat = cv2.getRotationMatrix2D(
#                 ((cols-1)/2.0, (rows-1)/2.0), degree, 1)
#             data_numpy = cv2.warpAffine(data_numpy, mat, (cols, rows))
             
#             # data_numpy = np.reshape(data_numpy, (data_numpy.shape[0] , data_numpy.shape[1], 2,16))
#             # data_numpy=np.transpose(data_numpy,(2,0,1,3))
#             # label_numpy = cv2.warpAffine(label_numpy, mat, (cols, rows))
#     # must use tf,not np,otherwise can not change the shape
#     data_numpy = tf.reshape(data_numpy, (data_numpy.shape[0] , data_numpy.shape[1], 2,16))
#     data_numpy=tf.transpose(data_numpy,(2,0,1,3))

#     return data_numpy, label_numpy



            



# class GaussianBlur(object):
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         #x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         #return x
#         return cv2.GaussianBlur(x,(0,0),sigma)
        






#     def __getitem__(self, idx):
#         img = self.imgs[idx]
#         field_mask = self.field_masks[idx]
#         if self.split_type == 'train':
#             img, field_mask = self.augment(img, field_mask)
#             img, field_mask = self.crop(img, field_mask)
#         return torch.FloatTensor(img[:, self.feat_arr]), torch.FloatTensor(self.areas[idx:idx+1]), torch.FloatTensor(field_mask), self.gts[idx]