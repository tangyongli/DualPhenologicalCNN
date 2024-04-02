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
def load_tiles( folder,s2mean,s2std,s1mean,s1std,startrow,startcol,endrow,endcol,startband,endband):
       
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
        print(files)
        imgbands=[]
        
        img = rio.open(files[0])
        trans = img.transform 
        for f in os.listdir(folder):
                if f.startswith('1') and f.endswith("tif"):
     
                    print('f',f)
                    f=os.path.join(folder, f)
                    img1 = rio.open(f).read()[...,startrow:endrow,startcol:endcol]#[1:10,...]
                    print('img1',img1.shape)
                    # print('img',img1.shape)
                    img1 = np.transpose(img1, (1, 2, 0))
                    print(img1.shape)
                    imgbands.append(img1)
              
    
        # img=np.array(imgbands)
        img=np.concatenate(imgbands,axis=-1)
        print(img.shape)
       
      

        # img = rio.open(imgpath).read()[startband:endband,startrow:endrow,startcol:endcol]
        s2img=img[...,0:endband]
        
        print('s2img',s2img.shape)
        # img=img[..., startrow:endrow,startcol:endcol]
   
        # s2img = np.transpose(s2img, (1, 2, 0))
    
        print('imgmedianmax',img.shape)
        print('s2mean',s2mean.shape)

        s2img=np.where(np.isnan(s2img), 0, (s2img - s2mean) / s2std)
        print('endband',endband)

        s1img=img[...,0:endband]
        s1img=np.where(np.isnan(s1img), 0, (s1img - s1mean) / s1std)
        return s2img,s1img, trans


def predict(modelpath,s2img,s1img,batchsize,patchsize,startrow,startcol,endrow,endcol): # height,width is the no padding image 
    import os
    print(os.path.isfile(modelpath))

    # model=tf.keras.models.load_model(modelpath,custom_objects={"K": K,'binary_weighted_cross_entropy': binary_weighted_cross_entropy})
    model=tf.keras.models.load_model(modelpath,custom_objects={"K": K})
    print(model.summary())
    results=[] #np.zeros(width,height)
    resultsnot=[]
    patches=[]
    patchcount=0
    pixels=[]
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
            patch=s2img[row-margin:row-margin+patchsize,col-margin:col-margin+patchsize,:]
            pixel=s1img[row:row+1,col:col+1,:]
            # print('pixel',pixel.shape)
            if patch.shape != (patchsize,patchsize, s2img.shape[-1]):
                   i=i+1
                   print(row,col,patch.shape)
                    # 调整形状并填充为-1
                   patch = np.full((patchsize,patchsize, s2img.shape[-1]), -1)
          
            patches.append(patch) #AttributeError: 'numpy.ndarray' object has no attribute 'append'
            pixels.append(pixel)
            patchcount+=1
            if patchcount==batchsize:
                patches=np.array(patches) # (512,6, 11,11, 10
                pixel=np.array(pixels)
                # print('pixel2',pixel.shape)
                pred=model.predict([patches,pixel]) #np.array(patches) #.reshape(-1,2)#(16,2)
                result=np.argmax(pred,axis=1)
                # print('result',result.shape)
                results.append(result)
                patchcount=0
                pixels=[]
                patches=[]
    print(patchcount,i) #4
    if patchcount!=batchsize:
            if patchcount==0:
                resultsnot==0
            else:         
                patches = np.array(patches)
                print('patches',patches.shape) #patches (36, 11, 11, 15)
                a=3
                print(pixels[0].shape,pixels[-1].shape) # (1, 1, 15) (1, 1, 15) np.concatenate(pixels,axis=0).shape (36, 1, 15)
                pixel=np.array(pixels)
                
                print('pixel3',pixel.shape)
                # 批量预测剩余的 patches
                pred = model.predict([patches,pixel]) #np.array(patches)
                resultsnot.append(np.argmax(pred, axis=1))
                        
    return np.array(results),np.array(resultsnot)
def predresult(modelpath,s2img,s1img,batchsize,patchsize,startrow,startcol,endrow,endcol,predictarraydir,predictjpgdir,saveVersion):

    prebatch,pred1=predict(modelpath,s2img,s1img,batchsize,patchsize,startrow,startcol,endrow,endcol) #
  
    shapes1=(prebatch.shape[0])*(prebatch.shape[1])
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

if __name__ == "__main__":
    print('111111111111111111111111111111111111')

    batchsize=256*3
    inputtag='10median3max'
   
    year,patchsize= 2022,11#modeldir.split('\\')[-1]
    modeldir=r"D:\DLRicemap\run\lognew"


    imgpath=r"D:\DLRicemap\dataset"
    
    startrow,startcol,endrow,endcol=3000,3000,5000,4000#2500,3000,3000,3500#7500,4500,8000,5000#2000,1000,2500,1500#4000,4000,5000,5000#3000,2000,4000,3000#5000,3000,6000,4000#2500,1000,3500,2000#2000,1000,2500,1500#4000,0,8000,2000
   
    startrow1,startcol1,endrow1,endcol1=2000,3000,3000,4000#3000,3000,3500,3500
    startrow2,startcol2,endrow2,endcol2=4500,4500,5000,5000
    startrow3,startcol3,endrow3,endcol3=2000,1000,3000,3000
    startrow4,startcol4,endrow4,endcol4=3000,3500,4000,4000
    predictdir=rf'D:\DLRicemap\run\predict'
    os.makedirs(predictdir,exist_ok=True)
    predicttifdir=os.path.join(predictdir,'tif')
    predictarraydir=os.path.join(predictdir,'array')
    predictjpgdir=os.path.join(predictdir,'jpg')
    
    os.makedirs(predicttifdir,exist_ok=True)
    os.makedirs(predictjpgdir,exist_ok=True)
    os.makedirs(predictarraydir,exist_ok=True)
    # s2mean=np.load(r"D:\DLRicemap\dataset\s2xmean1203_p11_c15.npy")
    # s2std=np.load(r"D:\DLRicemap\dataset\s2xstd1203_p11_c15.npy")
    
    # origintif=r"D:\DLRicemap\dataset\1yuechis1s2medianmaxindex.tif"
    origintif=r"D:\DLRicemap\dataset\1yuechis1vh_s2index301-901-001.tif"#"D:\DLRicemap\dataset\transfer\1roi2s1vh_s2index.tif"#"D:\DLRicemap\dataset\transfer\1roi1s1vh_s2index.tif" #(3040, 4037, 15)
    s2mean=np.load(rf"D:\DLRicemap\dataset\s2xmean3_9_1203_p11_c15.npy")
    s2std=np.load(rf"D:\DLRicemap\dataset\s2xstd3_9_1203_p11_c15.npy")
                        
    s1mean=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxmean3_9_1203_p1_c15.npy")
    s1std=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxstd3_9_1203_p1_c15.npy")
      
    def predict1(modeldir,startrow,startcol,endrow,endcol): 
        for modeltag in os.listdir(modeldir): 
            if modeltag[0:2] in ["69"]: # 10是10bands,12是9bands,11是10bands,'11' not multiscale l,13 是原来的7,16bands
                    # # print(mean.shape,std.shape)
                    if modeltag[0:2]=='69':
                        startband,endband=0,15
                        # s2mean=np.load(rf"D:\DLRicemap\dataset\s2xmean1203_p11_c15.npy")
                        # s2std=np.load(rf"D:\DLRicemap\dataset\s2xstd1203_p11_c15.npy")
                        
                        # s1mean=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxmean1203_p1_c15.npy")
                        # s1std=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxstd1203_p1_c15.npy")
                        s2mean=np.load(rf"D:\DLRicemap\dataset\s2xmean3_9_1203_p11_c15.npy")
                        s2std=np.load(rf"D:\DLRicemap\dataset\s2xstd3_9_1203_p11_c15.npy")
                                            
                        s1mean=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxmean3_9_1203_p1_c15.npy")
                        s1std=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxstd3_9_1203_p1_c15.npy")

                        print(s1mean.shape,s1std.shape)
                       
                      
        
                    elif modeltag[0:2]=='59':
                        startband,endband=0,15
                        s2mean=np.load(rf"D:\DLRicemap\dataset\s2xmean1203_p11_c15.npy")
                        s2std=np.load(rf"D:\DLRicemap\dataset\s2xstd1203_p11_c15.npy")
                        s1std=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxstd1203_p1_c15.npy")
                        s1mean=np.load(r"D:\DLRicemap\dataset\s1vh_s2indexxmean1203_p1_c15.npy")
                        
                    
                    # # print(mean.shape,std.shape)
                    # if modeltag[0:2]=='10':
                    #     startband,endband=0,10
                    #     mean=np.load(rf"D:\DLRicemap\dataset\xmean1201_p11_c{modeltag[0:2]}.npy")
                    #     std=np.load(rf"D:\DLRicemap\dataset\xstd1201_p11_c{modeltag[0:2]}.npy")
                    # elif modeltag[0:2]=='11':
                    #     startband,endband=0,10
                    #     mean=np.load(rf"D:\DLRicemap\dataset\xmean1201_p11_c10.npy")
                    #     std=np.load(rf"D:\DLRicemap\dataset\xstd1201_p11_c10.npy")
                    # elif modeltag[0:2]=='12':
                    #     startband,endband=1,10
                    #     mean=np.load(rf"D:\DLRicemap\dataset\xmean1201_p11_c9.npy")
                    #     std=np.load(rf"D:\DLRicemap\dataset\xstd1201_p11_c9.npy")
                    # elif modeltag[0:2]=='13':
                    #     startband,endband=0,16
                    #     mean=np.load(rf"D:\DLRicemap\dataset\xmean1201_p11_c16.npy")
                    #     std=np.load(rf"D:\DLRicemap\dataset\xstd1201_p11_c16.npy")
                    saveVersion=modeltag#'dualseparablecnn2d'
                    print(saveVersion)
                
                    modelpath=os.path.join(modeldir,modeltag)
                    modelpath=[os.path.join(modelpath, f) for f in os.listdir(modelpath) if  f.endswith('.h5')][0] # [3 1 2] 3是返回                #with open(logpath, 'r') as log_file
                    s2img,s1img,trans=load_tiles(imgpath,s2mean,s2std,s1mean,s1std,startrow,startcol,endrow,endcol,startband,endband)
                    # log_file_path=os.path.join(predicttifdir,saveVersion+'.txt')
                    # import logging
                    # import inspect
                    # # Configure logging
                    # logging.basicConfig(
                    # level=logging.INFO,
                    # format='%(asctime)s - %(levelname)s - %(message)s',
                    # handlers=[
                    #     logging.FileHandler(log_file_path),
                    #     logging.StreamHandler()  # To also print logs to console
                    # ])
                    
                
                    # logging.getLogger().setLevel(logging.INFO)
                
                    # # logging.info('saveMeanPath: %s', saveMeanPath)
                    # # logging.info('saveStdPath: %s', saveStdPath)
                    # logging.info('saveMOdelPath:%s', modelpath)
            
                    labelarray= predresult(modelpath,s2img,s1img,batchsize,patchsize,startrow,startcol,endrow,endcol,predictarraydir,predictjpgdir,saveVersion)
                    # labelarray=np.load(r"D:\DLRicemap\run\predict\array\69dualsparables2Cnn2d_s1cnn1dseparnor_fusatt1_nobc_cnn2d15bandsattention1bncnn-cnn1d15bandsnobn_attent32x64x128x256step2_multiscale0x0x1-att0x0x1-lrexponentialdecay_2000-3000x3000-4000.npy")
                    origintif=r"D:\DLRicemap\dataset\1yuechis1vh_s2index301-901-001.tif"
                    convertarraytolabeltif(origintif,labelarray,patchsize,startrow,startcol,endrow,endcol,predicttifdir,saveVersion)
    predict1(modeldir,startrow,startcol,endrow,endcol)
    # predict1(modeldir,startrow1,startcol1,endrow1,endcol1)
    # # predict1(modeldir,startrow2,startcol2,endrow2,endcol2)
    predict1(modeldir,startrow3,startcol3,endrow3,endcol3)
                   












