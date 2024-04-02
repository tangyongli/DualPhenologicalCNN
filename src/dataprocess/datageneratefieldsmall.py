import os
import json
import pandas as pd
import numpy as np
import rasterio as rio


'''
csv format
class:0 1 
area:0 yuechi 1 linshui 2 quxian
grid:




'''

def sample(file_name):
    
    df= pd.read_csv(file_name)
    # df['lon'] = df['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][0])
    # df['lat'] = df['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][1])
    # 随机打乱数据框
    resampledf = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # resampledf['class'].replace([1,2,3,4], [1,1,1,1], inplace=True)
    print( resampledf['class'].value_counts())
    return resampledf #resampledf
    


class PatchDataLoader:
    def __init__(self, sampledf, img_yuechi_path, patchsize,year,time_size):
        self.df = sampledf#pd.read_csv(sample_path)#.sample(1000,random_state=42)
       
        self.df['imgpatch'] = [None] * len(self.df)
        self.img_yuechi, self.trans_yuechi,self.band_size = self.load_tiles(img_yuechi_path)
        self.patchsize = patchsize
        self.year=year
        self.time_size = time_size

  
    def load_tiles(self, folder):
        imgbands = []
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith("tif")]
        print(files)
        img = rio.open(files[0])
        trans = img.transform 
        # img = img.read()[5:14,...]
        # print(img.shape)
        for f in os.listdir(folder):
                if f.startswith("1") and f.endswith("tif"):
     
                    print('f',f)
                    f=os.path.join(folder, f)
                    img1 = rio.open(f).read()#[1:10,...]
                    print('img1',img1.shape)
                    # print('img',img1.shape)
                    img1 = np.transpose(img1, (1, 2, 0))
                    print(img1.shape)
                    imgbands.append(img1)
              
    
        # img=np.array(imgbands)
        img=np.concatenate(imgbands,axis=-1)
     
   
        band_size=img.shape[-1]
        print(img.shape)
        return img, trans,band_size

    def patch_from_point(self,img, trans):
        # img,trans,band_size=self.load_tiles(r"D:\ricemodify\datasetRF\origin\yuechi2022")
        img,trans=self.img_yuechi, self.trans_yuechi
        print('image',img.shape)
     
        for index, row in self.df.iterrows():
            radius = self.patchsize // 2
            print('radius',radius)
            lon,lat=row["lon"],row["lat"]
            col, row = ~trans * (lon, lat)
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
                        self.df.at[index, "imgpatch"] = None
                    else:
                        self.df.at[index, "imgpatch"] = img_patch

            else:
                    print('imgshapeout',img_patch.shape)
                    self.df.at[index, "imgpatch"] = None
          
        return self.df

    def generate_patches(self):

        df = self.patch_from_point(self.img_yuechi, self.trans_yuechi)  # ('yuechi', lat, self.img_yuechi, self.trans_yuechi)
    
        df= df[ df['imgpatch'].apply(lambda x: x is not None)]
       
        print(df.head(5))
    


        imgtoarray=np.array(df['imgpatch'].tolist())
        # trainimgtoarray=(trainimgtoarray-mean)/std
      
        labeltoarray=np.array(df['class'].tolist())
  
        work_dir=r"D:\DLRicemap\dataset"
        data_dir=os.path.join(work_dir,f'{self.year}pathsize{self.patchsize}')
        os.makedirs(data_dir,exist_ok=True)
        # np.save(os.path.join(data_dir,f'xmean{imgtoarray.shape[0]}_patch{self.patchsize}_{mean.shape[0]}.npy'),mean)
        # np.save(os.path.join(data_dir,f'xstd{imgtoarray.shape[0]}_patch{self.patchsize}_{std.shape[0]}.npy'),std)

        np.save(os.path.join(data_dir, f"sarsamplesvalimage3_9{imgtoarray.shape[0]}" ),imgtoarray)
        np.save(os.path.join(data_dir, f"sarsamplesvallabel3_9{labeltoarray.shape[0]}" ),labeltoarray)
      
           
    

        return  imgtoarray,labeltoarray
    
    



    

'''
The result is datasetRF\pathsize11
spilt train and val based on the grid id;
Also has one mean std normination 
y without onehot encoder
x without nan fill. 



'''
if  __name__ == "__main__":
    print(os.path.abspath(__file__))
    work_dir=r"D:\DLRicemap\dataset"
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
    img_yuechi_path =r"D:\DLRicemap\dataset"
    patchsize = 11
    year=2022
    # sampledf=sample(r"D:\DLRicemap\dataset\yuechitrainsample.csv")
    sampledf=sample(r"D:\DLRicemap\dataset\yuechival408.csv")
   
    data_loader = PatchDataLoader(sampledf, img_yuechi_path,  patchsize=patchsize,year=year,time_size=time_size)
    
    x,y=data_loader.generate_patches()
    # patch_df = data_loader.get_data_frame()
    print('xy',x.shape,y.shape)
   
    # if x.shape[0]==1:
    #     x=np.squeeze(x,axis=0)
    # xpath=rf'D:\ricemodify\dataset\s1s270mask\datasplit\xyuanme_nomeanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy'
    # ypath=rf'D:\ricemodify\dataset\s1s270mask\datasplit\yyuanme_nomeanstd{y.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy'
    # cfgs.xpath=xpath
    # np.save(rf'dataset\withcloud\datasplit\xnomeanstd{x.shape[0]}x{x.shape[1]}x{x.shape[2]}x{x.shape[3]}x{x.shape[4]}.npy',x)
    # np.save(rf'dataset\withcloud\datasplit\ynomeanstd{y.shape[0]}.npy',y)
    # np.save(rf'dataset\s1s2medianmaxmincloudmask\xnomeanstd{x.shape[0]}x{x.shape[1]}.npy',x)
    # np.save(rf'dataset\s1s2medianmaxmincloudmask\ynomeanstd{y.shape[0]}.npy',y)
   