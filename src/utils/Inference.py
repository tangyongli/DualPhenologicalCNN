#%%
import tensorflow as tf
from keras import backend as K
import rasterio as rio
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import Affine
# from models import cnn3dattention
from keras import backend as K
import re
import  argparse
import geopandas as gpd
from rasterio.mask import mask
from loss import *

def load_tiles( modelname,f,mean2d,std2d,startrow,endrow,startcol,endcol):
        
        imgbands=[]
        print(f)
        files = [os.path.join(f, file) for file in os.listdir(f) if file.endswith("tif")]
        img = rio.open(files[0])
        trans = img.transform 
        
        for f1 in os.listdir(f):
                
            if f1.endswith("tif") :
                # if f1.startswith("1"):
                    f2=os.path.join(f, f1)
                    print(f2)

                    img1=rio.open(f2).read() [...,startrow:endrow,startcol:endcol]
                    img1 = np.transpose(img1, (1, 2, 0))
                    print(img1.shape)
                    imgbands.append(img1)
        img=np.concatenate(imgbands,axis=-1)
          
        
    
        print('imgshape',img.shape) # h,w,20
       
        if modelname in ["cnn1d",'cnn2d']: # 11,11,13
            img=np.where(np.isnan(img), 0, (img - mean2d) / std2d)
            print(img.shape)
        mask1= tf.where(img== 0, 0, 1)
        mask1 = tf.cast(mask1, dtype=tf.float32)  # 将 mask 转换为 float32
        mask = tf.convert_to_tensor(mask1)
        return img,mask


def predict(modelname,model,img2d,mask,batchsize,patchsize,startrow,startcol,endrow,endcol): # height,width is the no padding image 

   

    results=[] #np.zeros(width,height)
    resultsnot=[]
    patches=[]
    maskes=[]
    patchcount=0
    margin=patchsize//2
    i=0
    # 这样定义意味着加载进来的图像区域都要预测完
    startpredictrow=0
    startpredictcol=0
    endpredictrow=endrow-startrow-patchsize+1
    endpredictcol=endcol-startcol-patchsize+1
    # img=load_img(img_dir)
    for row in range(margin+startpredictrow,endpredictrow+margin,1): #(2000,2000,3000,3000) 预测的endrow ,endcol要在startrow,startcol上减去patchsize，再加上1
        for col in range(margin+startpredictcol,endpredictcol+margin,1):  # endcol:endcol+7----col-6:col+1
            if modelname in ['cnn2d']:
                patch=img2d[row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
                maskpatch=mask[row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
                # print(patch.shape,maskpatch.shape)
            if modelname in ['cnn1d']:
                patch=img2d[row-margin+patchsize//2:row-margin+patchsize//2+1,col-margin+patchsize//2:col-margin+patchsize//2+1,:]
                # print(patch.shape)
           
            maskes.append(maskpatch)
            patches.append(patch) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
            patchcount+=1

            # print('patchcount',patchcount) 
            # # 分批预测
            if patchcount==batchsize:
                patches,mask1=np.array(patches),np.array(maskes) # (512,6, 11,11, 10
                pred=model.predict([patches,mask1],verbose=0) 
                result=np.argmax(pred,axis=1)
                # print('result',result.shape)
                results.append(result)
                patchcount=0
                patches=[]
                maskes=[]
    
    if patchcount!=batchsize:
        if patchcount==0:
            resultsnot==0
        else:         
            patches,mask1= np.array(patches),np.array(maskes)
            print('patches',patches.shape) #patches (36, 11, 11, 15)
            pred=model.predict([patches,mask1],verbose=0) 
            resultsnot.append(np.argmax(pred, axis=1))
                        
    return np.array(results),np.array(resultsnot)
def predresult(modelname,model,img2d,mask,batchsize,patchsize,startrow,startcol,endrow,endcol,predictarraydir,predictjpgdir,saveVersion):

    prebatch,pred1=predict(modelname,model,img2d,mask,batchsize,patchsize,startrow,startcol,endrow,endcol) #
    print(prebatch.shape,pred1.shape)
    shapes1=(prebatch.shape[0])*(prebatch.shape[1])

    if pred1.shape[0]==0:
        finalpred=prebatch
    else:
        shapes2=(pred1.shape[0])*(pred1.shape[1])
        prebatch=prebatch.reshape(shapes1,-1)
        pred1=pred1.reshape(shapes2,-1)
        finalpred=np.append(prebatch,pred1,axis=0)
    os.makedirs(predictarraydir,exist_ok=True)


    finalpred=finalpred.reshape((endrow-startrow-patchsize+1,endcol-startcol-patchsize+1))
    np.save(os.path.join(predictarraydir,saveVersion+f'{startrow}-{endrow}x{startcol}-{endcol}.npy'),finalpred)
    # np.save(os.path.join(predictarray,f"cnnbest3dattentionmeanstdwrongfullyuechi"),finalpred)
    value,count=np.unique(finalpred,return_counts=True)
    print(finalpred.shape,value,count)
    plt.imshow(finalpred)
    plt.title(f'{saveVersion}-{startrow}-{endrow}x{startcol}-{endcol}')
    os.makedirs(predictjpgdir,exist_ok=True)
    savejpg=os.path.join(predictjpgdir,saveVersion+f'{startrow}-{endrow}x{startcol}-{endcol}.jpg')
    plt.savefig(savejpg)
    # plt.show()
    return finalpred

def convertarraytolabeltif(orgintif,labelarray,patchsize,startrow,startcol,endrow,endcol,savetifpath,saveVersion):
        tif=rio.open(orgintif)
        transform=rio.open(orgintif).transform
        rowtop=startrow+patchsize//2
        colleft=startcol+patchsize//2
        rowbottom=endrow-patchsize//2
        colright=endcol-patchsize//2
        left, bottom= transform * (colleft, rowbottom)
        right,top= transform*(colright, rowtop)
        dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs=tif.crs,
        dst_crs=tif.crs,
        width=colright-colleft,
        height=rowbottom-rowtop,
        left=left,
        bottom=bottom,
        right=right,
        top=top
        )
        # labelarray=np.loalabelarray,dtypelabelarray)
        print( 'dst',dst_width, dst_height) #995 883
        os.makedirs(savetifpath,exist_ok=True)
        outpath=os.path.join(savetifpath,saveVersion+f"{startrow}-{endrow}x{startcol}-{endcol}.tif") #{saveVersion_}_{startrow}-{endrow}x{startcol}-{endcol}
        # outpath=os.path.join(outtifpath,f"{startrow}-{endrow}x{startcol}-{endcol}.tif")
        print(' outpath', outpath)
      
        with rio.open(outpath, 'w', driver='GTiff', width=dst_width, height=dst_height, count=1, dtype=np.uint8, 
                        crs=tif.crs, transform=dst_transform) as dst:
            dst.write(labelarray, 1)
        print('arraytotifffinshed!!!!')
        return 0


def predict1(missingvalue,startrow,startcol,endrow,endcol): 
       
            
            modelpath=[os.path.join(directory,file) for file in os.listdir(directory) if file.endswith("h5")][0]#config['train']['model_path']
            print(modelpath)
            aw=2
            if missingvalue:
                print('111')
                 
                model=tf.keras.models.load_model(modelpath,custom_objects={"K": K,"binary_weighted_cross_entropy":binary_weighted_cross_entropy})
                
            else:
                model=tf.keras.models.load_model(modelpath,custom_objects={"K": K,"nomask_binary_weighted_cross_entropy": nomask_binary_weighted_cross_entropy})
            model.summary()
            if modelname=="dual_branch":

                mean2dpath=os.path.join(directory,'mean2d.npy')
                std2dpath= os.path.join(directory,'std2d.npy')
               
                mean2d,std2d=np.load(mean2dpath),np.load(std2dpath)
                print('mean',mean2d.shape)
                print('mean2dpath',mean2dpath)
                
            else:
                meanpath=os.path.join(directory,'mean.npy')
                stdpath= os.path.join(directory,'std.npy')
                mean2d,std2d=np.load(meanpath),np.load(stdpath)
              
            saveVersion= modeltag +"401510_620810median"#"pjnew610810_710920"+modeltag+"7ndsvirep" #newareapj
            print(saveVersion)

    
     
            img2d,mask=load_tiles( modelname,imgpath,mean2d,std2d,startrow,endrow,startcol,endcol)
       
            labelarray= predresult(modelname,model,img2d,mask,args.batchsize,patchsize,startrow,startcol,endrow,endcol,predictarraydir,predictjpgdir,saveVersion)
            # labelarray=np.load(r"D:\DLRicemap\log\predict\guanan\array\3addcount0_mask0patch7_dropflood0.0period2_cnn2d_64_satt0_drop0.2401510_620810median4000-4500x11500-12500.npy")
            convertarraytolabeltif(origintifpath,labelarray,patchsize,startrow,startcol,endrow,endcol,predicttifdir,saveVersion)
        
            
 
                   













# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command-line arguments.')
    parser.add_argument('--batchsize', type=int, default=256*6, help='Batch size')
    parser.add_argument('--modeldir', type=str, default=r"D:\DLRicemap\log\train\401501620801", help='modeldir')
    parser.add_argument('--modelname', type=str, default=r"cnn2d", help='modelname')
    parser.add_argument('--imgpath', type=str, default=r"D:\DLRicemap\datasets\guanan\620801", help='Path to predicted imagery')
    parser.add_argument('--patchsize', type=int, default=7, help='CNN input height and width')
    parser.add_argument('--predictdir', type=str, default=r'D:\DLRicemap\log\predict\guanan', help='Directory to save prediction')
    parser.add_argument('--originimgpath', type=str, default=r"D:\DLRicemap\datasets\guanan\620801\flood2401510.tif", help='help to define the spatial extent of predicted imagery')
  
    parser.add_argument('--range1', type=str,  default='3000,4000,4000,5000', help='Start row, end row, start col,end col, separated by commas')
    parser.add_argument('--range2', type=str,  default='5500,6500,4000,5000', help='Start row, end row, start col,end col, separated by commas')
    parser.add_argument('--range3', type=str,  default='4000,6000,11000,13000', help='Start row, end row, start col,end col, separated by commas')

    # Parse the range
    args = parser.parse_args()
    startrow1, endrow1, startcol1,endcol1 = [int(x) for x in args.range1.split(',')]
    startrow2, endrow2, startcol2,endcol2 = [int(x) for x in args.range2.split(',')]
    startrow3, endrow3, startcol3,endcol3 = [int(x) for x in args.range3.split(',')]
   
    imgpath=args.imgpath
    patchsize=args.patchsize
    # modelname=args.modelname
    origintifpath=args.originimgpath
    savemodeldir =args.modeldir
    os.makedirs(args.predictdir,exist_ok=True)
    predicttifdir=os.path.join(args.predictdir,'tif')
    predictarraydir=os.path.join(args.predictdir,'array')
    predictjpgdir=os.path.join(args.predictdir,'jpg')
    os.makedirs(predicttifdir,exist_ok=True)
    os.makedirs(predictjpgdir,exist_ok=True)
    os.makedirs(predictarraydir,exist_ok=True)
    model=tf.keras.models.load_model(r"D:\DLRicemap\log\train\fusion\2floodpeakbn_cnn2d_64_satt1_drop0.4_fusion1\floodpeakbn_cnn2d_64_satt1_drop0.4_fusion1.h5",custom_objects={"K": K,"focal_tversky_loss": focal_tversky_loss})
    print(model.summary())

    for modeltag in os.listdir(savemodeldir):
        print( modeltag)
        if modeltag[0:1] in ['6']:

        
            print( modeltag[0:1])
            modelname='cnn2d' #modeltag.split('_')[1]

            print(modelname)
            directory=os.path.join(savemodeldir,modeltag)
            print(directory)
            if modeltag[0:1]=='6':
                missingvalue=0
            else:
                missingvalue=1
            
            
            
            
            predict1(missingvalue,startrow3,startcol3,endrow3,endcol3)
            predict1(missingvalue,startrow2,startcol2,endrow2,endcol2)
    
            
 
                   













# %%
