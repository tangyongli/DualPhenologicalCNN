import os
import json
import pandas as pd
import numpy as np
import rasterio as rio

import cfgs
'''
csv format
class:0 1 
area:0 yuechi 1 linshui 2 quxian
grid:




'''

def resample(file_name,gridlist):
    resampledf= pd.DataFrame()
    df= pd.read_csv(file_name)
    df=df.rename(columns={'classification': 'class'})
    df=df[df['area']!=2]
    print(df.head(5))
    for grid in gridlist:
        dfgrid=df[df['grid']==grid]
        print(f'grid{grid}的各类数目为:',dfgrid['class'].value_counts())
        
        class0=dfgrid[dfgrid['class']==0]
        class1=dfgrid[dfgrid['class']==1] 
        class2=dfgrid[dfgrid['class']==2] 
        class3=dfgrid[dfgrid['class']==3] 
        class4=dfgrid[dfgrid['class']==4] 
        # class0=class0.sample(1700,random_state=10,
        #         replace=False)
        # class1=class1.sample(1000,random_state=10,
        #         replace=False)
        # class2=class2.sample(500,random_state=10,
        #         replace=False)
        # class3=class3.sample(250,random_state=10,
        #         replace=False)
        # class4=class4.sample(130,random_state=10,
        #         replace=False)
        class0=class0.sample(1000,random_state=10,
                replace=False)
        class1=class1.sample(600,random_state=10,
                replace=False)
        class2=class2.sample(200,random_state=10,
                replace=False)
        class3=class3.sample(200,random_state=10,
                replace=False)
        class4=class4.sample(100,random_state=10,
                replace=False)
        resampledf=pd.concat([resampledf,class0,class1,class2,class3,class4],ignore_index=True)
    # 随机打乱数据框
    resampledf = resampledf.sample(frac=1, random_state=42).reset_index(drop=True)
    # resampledf['class'].replace([1,2,3,4], [1,1,1,1], inplace=True)
    print( resampledf['class'].value_counts())
    return resampledf #resampledf
    
resampledf=resample('RF_DL/dataset/combinedgrid613172429.csv',gridlist=[6,13,17,24,29])
resampledf.to_csv('RF_DL/dataset/resamplegrid613172429.csv',index=False) 
# sample_path = r"D:\ricemodify\dataset\riceyuanandricemyboth_yuechiquxianlinshui3671_4648samplewithgridid.csv"
# resampledf=resample(sample_path,gridlist=[6,13,17])
# resampledf.to_csv('RF_DL/dataset/resampleyuechiquxianlinshui3671_4648_grid61317.csv',index=False)   



class PatchDataLoader:
    def __init__(self, sample, img_yuechi_path,  img_linshui_path, traingridlist,validgridlist,patchsize,year,time_size):
        self.df = sample#pd.read_csv(sample_path)#.sample(1000,random_state=42)
       
        self.df['imgpatch'] = [None] * len(self.df)
        self.img_yuechi, self.trans_yuechi,self.band_size = self.load_tiles(img_yuechi_path)
        # self.img_quxian, self.trans_quxian,self.band_size = self.load_tiles(img_quxian_path)
        self.img_linshui, self.trans_linshui,self.band_size = self.load_tiles(img_linshui_path)
        self.areas = {
            'yuechi': self.df[self.df['area'] == 0],
            'linshui': self.df[self.df['area'] == 1],
            'quxian': self.df[self.df['area'] == 2]
        }
        self.traingridlist=traingridlist
        self.validgridlist=validgridlist
        self.patchsize = patchsize
        self.year=year
        self.time_size = time_size

       
    # def load_tiles(self, folder,sar=False):
    #     imgbands = []
    #     glcm=[]
    #     files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
    #     img = rio.open(files[0])
    #     trans = img.transform 
    #     # img = img.read()[5:14,...]
    #     # print(img.shape)
    #     for f in os.listdir(folder):
    #         if f.endswith("tif"):
    #             if f.startswith('1'):
    #                 print('f',f)
    #                 f=os.path.join(folder, f)
    #                 img1 = rio.open(f).read()#[5:14, ...]
    #                 # print('img',img1.shape)
    #                 img1 = np.transpose(img1, (1, 2, 0))
    #                 print(img1.shape)
    #                 imgbands.append(img1)
    #             else:
    #                 f=os.path.join(folder, f)
    #                 print(f)
    #                 img1 = rio.open(f).read()
    #                 img1 = np.transpose(img1, (1, 2, 0))
    #                 print(img1.shape)
    #                 glcm.append(img1)
    #     glcm= np.array(glcm)
    #     glcm=np.transpose(glcm,(1,2,0,3))
    #     glcm=np.reshape(glcm,(glcm.shape[0],glcm.shape[1],-1))
    #     bands=np.array(imgbands)
    #     bands=np.transpose(bands,(1,2,0,3))
    #     bands=np.reshape(bands,(bands.shape[0],bands.shape[1],-1))
    #     img=np.concatenate([bands,glcm],axis=-1)
    #     band_size=img.shape[-1]
    #     print(img.shape)
    #     # print(glcm.shape,bands.shape)
    #     return img, trans,band_size
    def load_tiles(self, folder):
        imgbands = []
    
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
        img = rio.open(files[0])
        trans = img.transform 
        # img = img.read()[5:14,...]
        # print(img.shape)
        for f in os.listdir(folder):
            # if f.startswith("1"):
                if f.endswith("tif"):
     
                    print('f',f)
                    f=os.path.join(folder, f)
                    img1 = rio.open(f).read()#[1:10,...]
                    # print('img',img1.shape)
                    img1 = np.transpose(img1, (1, 2, 0))
                    print(img1.shape)
                    imgbands.append(img1)
              
    
        img=np.array(imgbands)
        img=np.transpose(img,(1,2,0,3))
        img=np.reshape(img,(img.shape[0],img.shape[1],-1))
        band_size=img.shape[-1]
        print(img.shape)
        # print(glcm.shape,bands.shape)
        return img, trans,band_size

    def patch_from_point(self,area, img, trans):
        areadf = self.areas[area]
       
        print('areadf',areadf.head(3))
        # areadf['lon'] = areadf['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][0])
        # areadf['lat'] = areadf['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][1])
        for index, row in areadf.iterrows():
            radius = self.patchsize // 2
            print('row',radius)
            arealon,arealat=row["lon"],row["lat"]
            col, row = ~trans * (arealon, arealat)
            patch_row_top, patch_row_bottom, patch_col_left, patch_col_right = (
                int(row - radius),
                int(row + radius + 1),
                int(col - radius),
                int(col + radius + 1),
            )
            img_patch = img[patch_row_top:patch_row_bottom, patch_col_left:patch_col_right, :]
            print(img_patch.shape)
           
            if img_patch.shape == (self.patchsize, self.patchsize, self.band_size):
                    print(img_patch.shape)
                    if np.any(np.isnan(img_patch)):
                        print('nan')
                        areadf.at[index, "imgpatch"] = None
                    else:
                        areadf.at[index, "imgpatch"] = img_patch

            else:
                    print('imgshapeout',img_patch.shape)
                    areadf.at[index, "imgpatch"] = None
            # return areadf # if return is located here,the fucntion will be only run one times,for the first row/
            # return statement terminates the function and returns control to the caller.
        return areadf

    def generate_patches(self):

        yuechi_patchdf = self.patch_from_point('yuechi',  self.img_yuechi, self.trans_yuechi)  # ('yuechi', lat, self.img_yuechi, self.trans_yuechi)
        linshui_patchdf = self.patch_from_point('linshui', self.img_linshui, self.trans_linshui)
        # quxian_patchdf = self.patch_from_point('quxian', self.img_quxian, self.trans_quxian)
        
        # print(quxian_patchdf.head(5)) #(3544, 9)
        # print(quxian_patchdf['imgpatch'][0].shape) 
        yuechi_patchdf= yuechi_patchdf[ yuechi_patchdf['imgpatch'].apply(lambda x: x is not None)]
        linshui_patchdf= linshui_patchdf[ linshui_patchdf['imgpatch'].apply(lambda x: x is not None)]
        print(yuechi_patchdf.head(5),linshui_patchdf.head(4))
        # quxian_patchdf= quxian_patchdf[ quxian_patchdf['imgpatch'].apply(lambda x: x is not None)]
        # print(quxian_patchdf['imgpatch'][0].shape) 
        df=pd.concat([yuechi_patchdf,  linshui_patchdf], ignore_index=True)
        arraydf=np.array(df['imgpatch'].tolist())
        print( 'imgarray',arraydf.shape)
        mean=np.nanmean(arraydf,axis=(0,1,2))
        std=np.nanstd(arraydf,axis=(0,1,2))
        traindata= df[df['grid'].isin(self.traingridlist)]
        valdata=df[df['grid'].isin(self.validgridlist)]
        
        trainimgtoarray=np.array(traindata['imgpatch'].tolist())
        # trainimgtoarray=(trainimgtoarray-mean)/std
      
        trainlabeltoarray=np.array(traindata['class'].tolist())
        print('traindata',np.unique(trainlabeltoarray,return_counts=True))
        valimgtoarray=np.array(valdata['imgpatch'].tolist())
        # valimgtoarray=(valimgtoarray-mean)/std
        
        vallabeltoarray=np.array(valdata['class'].tolist())
        traincount,valcount=trainimgtoarray.shape[0],valimgtoarray.shape[0]
        totalsample=traincount+valcount
        # print('valdata',np.unique(vallabeltoarray,return_counts=True))
        # print( 'imgarray',valimgtoarray.shape,vallabeltoarray.shape)
        work_dir=os.getcwd()
        data_dir=os.path.join(work_dir,f'datasetRF\{self.year}pathsize{self.patchsize}\sevenper')
        # traindata_dir=os.path.join(data_dir,'traingrid61324')
        # valdata_dir=os.path.join(data_dir,'valgrid1729')
        os.makedirs(data_dir,exist_ok=True)
        np.save(os.path.join(data_dir,f'xmean{totalsample}_patch{self.patchsize}_{mean.shape[0]}.npy'),mean)
        np.save(os.path.join(data_dir,f'xstd{totalsample}_patch{self.patchsize}_{std.shape[0]}.npy'),std)
        labelcount=np.unique(vallabeltoarray).shape[0]
        np.save(os.path.join(data_dir,rf'trainxgrid61324nomeanstd_{ trainimgtoarray.shape[0]}x{trainimgtoarray.shape[1]}x{trainimgtoarray.shape[2]}x{trainimgtoarray.shape[-1]}.npy'),trainimgtoarray)
        np.save(os.path.join(data_dir,rf'trainygrid61324_{ trainlabeltoarray.shape[0]}x{trainimgtoarray.shape[1]}x{trainimgtoarray.shape[2]}x{labelcount}.npy'),trainlabeltoarray)

        np.save(os.path.join(data_dir,rf'valxgrid1729nomeanstd_{ valimgtoarray.shape[0]}x{trainimgtoarray.shape[1]}x{trainimgtoarray.shape[2]}x{valimgtoarray.shape[-1]}.npy'),valimgtoarray)
        np.save(os.path.join(data_dir,rf'valygrid1729_{ vallabeltoarray.shape[0]}x{valimgtoarray.shape[1]}x{valimgtoarray.shape[2]}x{labelcount}.npy'),vallabeltoarray)
 
           
        yuechi_patchdf,linshui_patchdf=None,None

        return  trainimgtoarray,trainlabeltoarray,valimgtoarray,vallabeltoarray
    
    



    

'''
The result is datasetRF\pathsize11
spilt train and val based on the grid id;
Also has one mean std normination 
y without onehot encoder
x without nan fill. 



'''
if  __name__ == "__main__":
    print(os.path.abspath(__file__))
    work_dir=os.getcwd()
    patchsize=11
    time_size=1
    band_size=25
    # data_dir=os.path.join(work_dir,f'datasetRF\pathsize{patchsize}')
    # traindata_dir=os.path.join(data_dir,'traingrid61324')
    # valdata_dir=os.path.join(data_dir,'valgrid1729')
    # os.makedirs(data_dir,exist_ok=True)
    # os.makedirs(traindata_dir,exist_ok=True)
    # os.makedirs(valdata_dir,exist_ok=True)
   
    #D:\ricemodify\dataset\s1s2medianmaxmincloudmask\ynomeanstd5196.npy
    img_yuechi_path =r"D:\ricemodify\datasetRF\origin\yuechi2022"
    img_quxian_path =r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\quxian'
    img_linshui_path =r'D:\ricemodify\datasetRF\origin\linshui2022' 
    # img_yuechi_path =r"D:\riceyuechi\dataset\yuechi\image"
    # img_linshui_path =r"D:\riceyuechi\dataset\linshui\image"
    # img_quxian_path =r"D:\riceyuechi\dataset\quxian\image"
    # img_yuechi_path =r"D:\ricemodify\dataset\s1s2medianmaxmincloudmask\yuechi"
    # img_quxian_path =r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\quxian'
    # img_linshui_path =r'D:\ricemodify\dataset\s1s2medianmaxmincloudmask\linshui' 
    patchsize = 15
    year=2022
    data_loader = PatchDataLoader(resampledf, img_yuechi_path,  img_linshui_path,  traingridlist=[6,13,24],validgridlist=[17,29],patchsize=patchsize,year=year,time_size=time_size)
    
    x,y,xval,yval=data_loader.generate_patches()
    # patch_df = data_loader.get_data_frame()
    print('xy',x.shape,y.shape)
    print('xy',xval.shape,yval.shape)
    # if x.shape[0]==1:
    #     x=np.squeeze(x,axis=0)
    # xpath=rf'D:\ricemodify\dataset\s1s270mask\datasplit\xyuanme_nomeanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy'
    # ypath=rf'D:\ricemodify\dataset\s1s270mask\datasplit\yyuanme_nomeanstd{y.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy'
    # cfgs.xpath=xpath
    # np.save(rf'dataset\withcloud\datasplit\xnomeanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy',x)
    # np.save(rf'dataset\withcloud\datasplit\ynomeanstd{y.shape[0]}.npy',y)
    # np.save(rf'dataset\s1s2medianmaxmincloudmask\xnomeanstd{x.shape[0]}x{x.shape[1]}.npy',x)
    # np.save(rf'dataset\s1s2medianmaxmincloudmask\ynomeanstd{y.shape[0]}.npy',y)
   