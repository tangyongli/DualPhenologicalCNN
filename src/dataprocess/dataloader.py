from src.dataprocess.transformer import *
from tensorflow.keras.utils import to_categorical
import numpy as np





from sklearn.model_selection import train_test_split



def dataprogress(savetrainxPath, savetrainyPath,patchsize,xcentermeanstd=1):
    xtrain,ytrain=np.load(savetrainxPath),np.load(savetrainyPath)
    print(ytrain.shape)
    ytrain=ytrain.astype(np.int32)
    print(ytrain)
    patchshape=xtrain.shape[1]
    channels=xtrain.shape[-1]
    ytrain= tf.convert_to_tensor(ytrain)
    ytrain=to_categorical(ytrain,num_classes=2,dtype=np.float32)

    xtrain=xtrain[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
    
    print(np.unique(ytrain,return_counts=True))
    
    x=xtrain#np.concatenate([xtrain,xval])
    mean=np.nanmean(x,axis=(0,1,2))
    std=np.nanstd(x,axis=(0,1,2))
    # print('mean',mean.shape,'std',std.shape)
    xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
    xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
   

   
    return xtrain,ytrain,xval,yval
# def dataprogressindependval(savetrainxPath, savetrainyPath,savexvalPath,saveyvalPath,patchsize,xcentermeanstd):
#     xtrain,ytrain=np.load(savetrainxPath),np.load(savetrainyPath)
#     xval,yval=np.load(savexvalPath),np.load(saveyvalPath)
   
#     ytrain=ytrain.astype(np.int32)
#     print(ytrain)
#     patchshape=xtrain.shape[1]
#     channels=xtrain.shape[-1]
#     ytrain= tf.convert_to_tensor(ytrain)
#     ytrain=to_categorical(ytrain,num_classes=2,dtype=np.float32)
#     yval=tf.convert_to_tensor(yval)
#     yval=to_categorical(yval,num_classes=2,dtype=np.float32)

#     xtrain=xtrain[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
#     xval=xval[...,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    
    
#     x=np.concatenate([xtrain,xval])
#     mean=np.nanmean(x,axis=(0,1,2))
#     std=np.nanstd(x,axis=(0,1,2))
#     np.save(rf'D:\DLRicemap\dataset\xmean{x.shape[0]}_p{patchsize}_c{mean.shape[-1]}.npy',mean)
#     np.save(rf'D:\DLRicemap\dataset\xstd{x.shape[0]}_p{patchsize}_c{mean.shape[-1]}.npy',std)

#     xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
#     xval=np.where(np.isnan(xval), 0, (xval- mean) / std)
#     if xcentermeanstd:
#          xtraincenter=xtrain[...,patchshape//2:patchshape//2+1,patchshape//2:patchshape//2+1,0:channels]
#          xvalcenter=xval[...,patchshape//2:patchshape//2+1,patchshape//2:patchshape//2+1,0:channels]
#          xcenter=x[...,patchshape//2:patchshape//2+1,patchshape//2:patchshape//2+1,0:channels]
#          mean=np.nanmean(xcenter,axis=(0,))
#          std=np.nanstd(xcenter,axis=(0,))
#          print('mean',mean.shape,'std',std.shape)
#          xtraincenter=np.where(np.isnan(xtraincenter), 0, (xtraincenter - mean) / std)
#          xvalcenter=np.where(np.isnan(xvalcenter), 0, (xvalcenter- mean) / std)
#          print(xcenter.shape)
#          return xtrain,ytrain,xval,yval,xtraincenter,xvalcenter
         

   
#     return xtrain,ytrain,xval,yval,xcenter

#'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','ndvimax','evimax','lswimax','vhmin','vhmedian','vhmax'，'vvmin','vvmedian','vvmax','ndvistd','evistd','lswistd','vhstd','vvstd'
def dataprogressindependval(savetrainxPath, savetrainyPath, savexvalPath, saveyvalPath, patchsize, xcentermeanstd=1):
    xtrain, ytrain = np.load(savetrainxPath), np.load(savetrainyPath)
    print(xtrain.shape,ytrain.shape)
    xval, yval = np.load(savexvalPath), np.load(saveyvalPath)
   
    ytrain = ytrain.astype(np.int32)

    patchshape = xtrain.shape[1]
    channels =15# xtrain.shape[-1]
    ytrain = tf.convert_to_tensor(ytrain)
    ytrain = to_categorical(ytrain, num_classes=2, dtype=np.float32)
    yval = tf.convert_to_tensor(yval)
    yval = to_categorical(yval, num_classes=2, dtype=np.float32)

    s2xtrain = xtrain[..., patchshape//2-patchsize//2:patchshape//2+patchsize//2+1, patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    s2xval = xval[..., patchshape//2-patchsize//2:patchshape//2+patchsize//2+1, patchshape//2-patchsize//2:patchshape//2+patchsize//2+1,0:channels]
    
    s2x = np.concatenate([s2xtrain, s2xval])
    mean = np.nanmean(s2x, axis=(0, 1, 2))
    std = np.nanstd(s2x, axis=(0, 1, 2))
    print(mean.shape,std.shape)
    np.save(rf'D:\DLRicemap\dataset\s2xmean3_9_{s2x.shape[0]}_p{patchsize}_c{mean.shape[-1]}.npy', mean)
    np.save(rf'D:\DLRicemap\dataset\s2xstd3_9_{s2x.shape[0]}_p{patchsize}_c{mean.shape[-1]}.npy', std)

    s2xtrain = np.where(np.isnan(s2xtrain), 0, (s2xtrain - mean) / std)
    s2xval = np.where(np.isnan(s2xval), 0, (s2xval - mean) / std)
    
    if xcentermeanstd:
        patchsize=1
        xtraincenter = xtrain[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:channels]
        xvalcenter = xval[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:channels]
        xcenter = np.concatenate([xtraincenter, xvalcenter], axis=0)
        # max1 = np.nanmax(xcenter, axis=(0,))
        # min1 = np.nanmin(xcenter, axis=(0,))
        # print('mean', mean.shape, 'std', std.shape)
        # xtraincenter = np.where(np.isnan(xtraincenter), 0, (xtraincenter - min1) / (max1-min1))
        # xvalcenter = np.where(np.isnan(xvalcenter), 0, (xvalcenter - min1) / (max1-min1))
        mean = np.nanmean(xcenter, axis=(0,))
        std = np.nanstd(xcenter, axis=(0,))
        print('mean', mean.shape, 'std', std.shape)
        xtraincenter = np.where(np.isnan(xtraincenter), 0, (xtraincenter - mean) / std)
        xvalcenter = np.where(np.isnan(xvalcenter), 0, (xvalcenter - mean) / std)
        np.save(rf'D:\DLRicemap\dataset\s1vh_s2indexxmean3_9_{xcenter.shape[0]}_p{patchsize}_c{mean.shape[-1]}.npy', mean)
        np.save(rf'D:\DLRicemap\dataset\s1vh_s2indexxstd3_9_{xcenter.shape[0]}_p{patchsize}_c{mean.shape[-1]}.npy', std)
        # print(xcenter.shape)
        return s2xtrain, ytrain, s2xval, yval, xtraincenter, xvalcenter
    else:
         xtraincenter, xvalcenter = None, None

    return xtrain, ytrain, xval, yval,xtraincenter, xvalcenter


def dataagument(xtrain,ytrain,xuntrain,p,strong=1):
    strongcompose_transform = RandomApply([RandomVerticalFlip(),RandomHorizontalFlip(),RandomRotation(),RandomScale()],p) 
    weakcompose_transform = RandomApply([RandomRotation(),RandomVerticalFlip(),RandomHorizontalFlip()],p)
    h,w,c=xtrain.shape[-3],xtrain.shape[-2],xtrain.shape[-1]
    def strongapply_transforms(x, y):
                if strong:
                    x_transformed, y_transformed = strongcompose_transform(x, y)
                    # print( x_transformed.shape, y_transformed.shape)
                else:
                    x_transformed, y_transformed = weakcompose_transform(x, y)
               
              
                
                return x_transformed, y_transformed
    if strong:
        xtrainarray=np.zeros((len(xtrain),h,w,c),dtype=np.float32)
        ytrainarray=np.zeros((len(ytrain),2),dtype=np.float32)
        for i, (x,y) in enumerate(zip(xtrain,ytrain)):
                xtrainarray[i],ytrainarray[i]=strongapply_transforms(x,y)
        return xtrainarray,ytrainarray
    else:
        xuntrainarray=np.zeros((len(xuntrain),h,w,c),dtype=np.float32)
        for i, (x,y) in enumerate(zip(xuntrain,xuntrain)):
           
            xuntrainarray[i],_=strongapply_transforms(x,y)
        return xuntrainarray