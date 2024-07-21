from src.dataprocess.transformer import *
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
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

def dataprogressindependval(modelname,savetrainxPath, savetrainyPath,savemodeldir,period):
   
#    cnn2dgru: samplesize,timestep,patchsize,patchsize,channels (none,10,11,11,2)
#    cnn3d_cbrm: samplesize,patchsize,patchsize,channels (none,11,11,13) 
#    dual_branch: samplesize,patchsize,patchsize,channels (none,11,11,15)
#    根据不同模型进行相应标准化,mean std文件和模型保存在一起,方便后续推理调用;
    '''
    ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','EVI','LSWI','NDSVI','REP']

    '''
    xtrain, ytrain = np.load(savetrainxPath)[...,2:9,2:9,0:26], np.load(savetrainyPath) # [...,1:10,1:10,0:30]
    ytrain = ytrain.astype(np.int32) 
    # 交叉划分训练集和验证集
    patchshape = xtrain.shape[1]
    channels =xtrain.shape[-1]
    print(np.unique(ytrain))
    # ytrain=ytrain-1
    print(np.unique(ytrain))
    ytrain = tf.convert_to_tensor(ytrain)
    ytrain = to_categorical(ytrain, num_classes=2, dtype=np.float32)

    y=ytrain #np.concatenate([ytrain,yval])
    x = xtrain #np.concatenate([xtrain, xval])
    print('x',x.shape)
    if period==0:
        # xcenter= xtrain[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:13]
        # xcenter=xtrain[...,0:13]
        # Find indices where xtrain is NaN
        # nan_indices = np.isnan(xcenter).any(axis=(1, 2, 3))
        # print(nan_indices.sum())
        # Delete elements from both xtrain and ytrain at the NaN indices
        # xtrain = xtrain[~nan_indices,...]
        # ytrain = ytrain[~nan_indices,...]
        # xtrain = xtrain[nan_indices,...]
        # ytrain = ytrain[nan_indices,...]
        xtrain=xtrain
        print(xtrain.shape) # 248个样本在401-510中心像素存在空缺值；196个样本在401-510全为空；376个样本存在缺失值；
        a=3
        
    if period==1:
        xcenter = xtrain[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:13]
        # Find indices where xtrain is NaN
        nan_indices = np.isnan(xcenter).any(axis=(1, 2, 3))
        # Delete elements from both xtrain and ytrain at the NaN indices
        xtrain = xtrain[~nan_indices]
        ytrain = ytrain[~nan_indices]
    if period==2:
        xtrain=xtrain


    if modelname=='cnngru': # none,10,11,11,2
        # print(xtrain.shape) #(1926, 10, 11, 11, 2)
        # print(x[0:1926,1:5,...])

        x=np.transpose(x,(0,2,3,1,4)) # none,11,11,10,2
        # print('x',x.shape)
        # print(x[0:1,...,0:1,0:2])
        # print('cccccccccccccccccccccccccccc')
        x=np.reshape(x,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]*x.shape[4])) # none,11,11,20
        mean=np.nanmean(x,axis=(0,1,2))
        std=np.nanstd(x,axis=(0,1,2))
        # print(mean.shape,std.shape) # 20,
        # print(x[...,0:1])
       
        # print(x[...,1:2])
        xtrain=np.transpose(xtrain,(0,2,3,1,4))
        xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],-1))
        # print(xtrain.shape)
        xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
        xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],10,-1))
        xtrain=np.transpose(xtrain,(0,3,1,2,4))
        # xval=np.transpose(xval,(0,2,3,1,4))

        # xval=np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],-1))
        # xval=np.where(np.isnan(xval), 0, (xval - mean) / std)
        # xval=np.reshape(xval,(xval.shape[0],xval.shape[1],xval.shape[2],10,-1))
        # xval=np.transpose(xval,(0,3,1,2,4))

        savemean2DPath= os.path.join(savemodeldir,f"mean.npy")
        savestd2DPath= os.path.join(savemodeldir,f"std.npy")

        np.save(savemean2DPath,mean) 
        np.save(savestd2DPath,std)
    if modelname in ['cnn1d']: # none,11,11,13
        xtraincenter = xtrain[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:channels]
        # xvalcenter = xval[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:channels]
        # xcenter = np.concatenate([xtraincenter, xvalcenter], axis=0) # NONE,15
        xcenter=xtraincenter
        print('xcenter',xcenter.shape) # 1,1,15
        mean1d = np.nanmean(xcenter, axis=(0,))
        std1d = np.nanstd(xcenter, axis=(0,))
        xtrain = np.where(np.isnan(xtraincenter), 0, (xtraincenter - mean1d) / std1d)
        # xvalcenter = np.where(np.isnan(xvalcenter), 0, (xvalcenter - mean1d) / std1d)
        savemean2DPath= os.path.join(savemodeldir,f"mean.npy")
        savestd2DPath= os.path.join(savemodeldir,f"std.npy")
        np.save(savemean2DPath,mean1d) 
        np.save(savestd2DPath,std1d)
        print('meanstd',mean1d.shape,std1d.shape)
        
    if modelname in ['cnn3d','cnn2d']: # none,11,11,13
        mean=np.nanmean(x,axis=(0,1,2))
        std=np.nanstd(x,axis=(0,1,2))
        print(mean.shape,std.shape) # 13,
        xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean) / std)
        # xval=np.where(np.isnan(xval), 0, (xval - mean) / std)
        savemean2DPath= os.path.join(savemodeldir,f"mean.npy")
        savestd2DPath= os.path.join(savemodeldir,f"std.npy")
        np.save(savemean2DPath,mean) 
        np.save(savestd2DPath,std)
        xtraincenter,xvalcenter=None,None
    if modelname=='dual_branch': # none,11,11,13
        # cnn2d分支归一化
        mean2d=np.nanmean(x,axis=(0,1,2))
        std2d=np.nanstd(x,axis=(0,1,2))
        xtrain=np.where(np.isnan(xtrain), 0, (xtrain - mean2d) / std2d)
        savemean2DPath= os.path.join(savemodeldir,f"mean2d.npy")
        savestd2DPath= os.path.join(savemodeldir,f"std2d.npy")
        np.save(savemean2DPath,mean2d) 
        np.save(savestd2DPath,std2d)
        # cnn1d分支归一化
        xtraincenter = xtrain[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:channels]
        # xvalcenter = xval[..., patchshape//2:patchshape//2+1, patchshape//2:patchshape//2+1, 0:channels]
        # xcenter = np.concatenate([xtraincenter, xvalcenter], axis=0) # NONE,15
        xcenter=xtraincenter
        print('xcenter',xcenter.shape) # 1,1,15
        mean1d = np.nanmean(xcenter, axis=(0,))
        std1d = np.nanstd(xcenter, axis=(0,))
        xtraincenter = np.where(np.isnan(xtraincenter), 0, (xtraincenter - mean1d) / std1d)
        # xvalcenter = np.where(np.isnan(xvalcenter), 0, (xvalcenter - mean1d) / std1d)
        savemean1DPath= os.path.join(savemodeldir,f"mean1d.npy")
        savestd1DPath= os.path.join(savemodeldir,f"std1d.npy")
        np.save(savemean2DPath,mean1d) 
        np.save(savestd2DPath,std1d)
        print('meanstd',mean1d.shape,std1d.shape,mean2d.shape,std2d.shape)
      
    # xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.3, random_state=42)

    return xtrain, ytrain


def dataagument(xtrain,ytrain,xuntrain,p,droptimestep,strong=1):
    # strongcompose_transform = RandomApply([RandomVerticalFlip(),RandomHorizontalFlip(),RandomRotation(),RandomScale()],p) 
    # weakcompose_transform = RandomApply([RandomRotation(),RandomVerticalFlip(),RandomHorizontalFlip()],p)
    strongcompose_transform =RandomApply([RandomChannelDrop(droptimestep)],p)
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