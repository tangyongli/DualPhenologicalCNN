import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
#pip install pyyaml

import yaml

'''
csv format
class:0 1 
area:0 yuechi 1 linshui 2 quxian
grid:

'''

def sample(file_name,f2):
    
    df1= gpd.read_file(file_name)
    df2=gpd.read_file(f2)
    df=pd.concat([df1,df2],axis=0)
    print(df.head())
    print(len(df))
    # df2=pd.read_csv(f2)
    # df=pd.concat([df1,df2],axis=0)
    print('df',df)
    # df['count']=0
    # df['class']=df['name']
    # 去除重复的地理坐标数据
    df = df.drop_duplicates(subset=['geometry'])
    df['class']= df['Name'].apply(lambda x: 1 if x == "1" else 0)
    # df.to_csv(r"D:\DLRicemap\trainval4791.csv",index=False)
    # print(len(df))
    # print(df['class'].value_counts())
    # df=df[df['class']!=0]
    # df=df[df['class']!=6]
  
    # ricesample = df[df['class'] == 1].sample(n=500, random_state=1)
    # others=df[df['class'] != 1]
    # df=pd.concat([ricesample,others])
    # print('df1',np.unique(df['class'],return_counts=True))
    # print(df1)
    # for val
    # df["class"]=df["Name"]
    # df['lon'] = df['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][0])
    # df['lat'] = df['.geo'].apply(lambda geo_str: json.loads(geo_str)['coordinates'][1])
    ### for shp file
    df['lon'] = df.geometry.x
    df['lat'] = df.geometry.y
    resampledf = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(resampledf)
    # resampledf['class'].replace([1,2,3,4], [1,1,1,1], inplace=True)
    print( file_name)
    # resampledf.to_csv(r"D:\DLRicemap\datasets\samples\guanansamplefloodcount0401510.csv",index=False)
    return resampledf #resampledf
    
# sample(r"C:\Users\TANG\Downloads\guanancount0flood_401510.csv")
a=4

def load_tiles(f):
        imgbands = []

     
        files = [os.path.join(f, file) for file in os.listdir(f) if file.endswith("tif")]
        img = rio.open(files[0])
        trans = img.transform 
        for f1 in os.listdir(f):
            if f1.endswith("tif") :
                f2=os.path.join(f, f1)
                print(f2)
                # if f1.startswith("1"):
                img1 = rio.open(f2).read()#[0:15, ...]
                img1 = np.transpose(img1, (1, 2, 0))
                print(img1.shape)
                imgbands.append(img1)
                # print('img',img1.shape)
                # if f1.startswith("2"):
                #     img1= rio.open(f2).read()[0:2,...]
            
                #     img1 = np.transpose(img1, (1, 2, 0))
                #     print(img1.shape)
                #     imgbands.append(img1)
                # if f1.startswith("3"):
                #     img1= rio.open(f2).read()[0:1,...]
            
                #     img1 = np.transpose(img1, (1, 2, 0))
                #     print(img1.shape)
                #     imgbands.append(img1)
            
            
        
        
        # img1 = rio.open(f).read()#[0:1,...]
        # transimg= rio.open(f)
        # trans = transimg.transform 
        
        # print('img1',img1.shape)
        # # print('img',img1.shape)
        # img1 = np.transpose(img1, (1, 2, 0))
        # print(img1.shape)
        # imgbands.append(img1)
    
    
        # img=np.array(imgbands)
        img=np.concatenate(imgbands,axis=-1)
     
   
        band_size=img.shape[-1]
        print(img.shape)
        return img, trans,band_size

class PatchDataLoader:
    def __init__(self, category,sampledf,img, trans,band_size,patchsize,year,time_size,saveimgtitle,savelabeltitle):
        self.category=category
        self.df = sampledf #pd.read_csv(sample_path)#.sample(1000,random_state=42)
        # self.img_yuechi, self.trans_yuechi,self.band_size = load_tiles(img_yuechi_path)
        self.df['imgpatch'] = [None] * len(self.df)
        self.dfcount=pd.DataFrame()
        self.img_yuechi=img
        self.trans_yuechi=trans
        self.band_size=band_size
        self.patchsize = patchsize
        self.year=year
        self.time_size = time_size
        self.saveimgtitle=saveimgtitle
        self.savelabeltitle=savelabeltitle
        self.invalid_indices = []

  
 
    def patch_from_point(self,img, trans):
        # img,trans,band_size=self.load_tiles(r"D:\ricemodify\datasetRF\origin\yuechi2022")
        img,trans=self.img_yuechi, self.trans_yuechi
        print('image',img.shape)
        print('trans',trans)

     
        for index, row in self.df.iterrows():
            radius = self.patchsize // 2
            # print('radius',radius)
            lon,lat=row["lon"],row["lat"]
            # print('lon',lon,'lat',lat)
            col, row = ~trans * (lon, lat)
            patch_row_top, patch_row_bottom, patch_col_left, patch_col_right = (
                int(row - radius),
                int(row + radius + 1),
                int(col - radius),
                int(col + radius + 1),
            )
            # print(patch_col_right, patch_col_left, patch_row_top, patch_row_bottom)
            img_patch = img[patch_row_top:patch_row_bottom, patch_col_left:patch_col_right, :]
            # print(img_patch.shape)
            '''
            trans | 0.00, 0.00, 113.81|
            | 0.00,-0.00, 31.44|
            | 0.00, 0.00, 1.00|
            radius 5
            lon 106.3712645 lat 30.51806091
            -82856 -82867 10246 10257
            '''
         
           
            if img_patch.shape == (self.patchsize, self.patchsize, self.band_size):
                    imgcenter=img_patch[5:6,5:6,...]
                    if np.all(np.isnan(imgcenter)):
                  
                    # if np.all(np.isnan(img_patch)):
                        print('centernan')
                     
                        self.df.at[index, "imgpatch"] = None
                        # FOR Import nan value samples in flood period
                        # self.df.at[index, "imgpatch"] = img_patch

                        self.invalid_indices.append(index)  
                    else:
                        self.df.at[index, "imgpatch"] = img_patch

            else:
                    print('imgshapeout',img_patch.shape)
                    self.df.at[index, "imgpatch"] = None
          
        return self.df

    def generate_patches(self,c):

        df = self.patch_from_point(self.img_yuechi, self.trans_yuechi)  # ('yuechi', lat, self.img_yuechi, self.trans_yuechi)
        if c in [0,1,2]:
            df=df[df['countflood']==c]
        # elif c=="5+":
        #     print('ccccccccccccccccccccc')
        #     df=df[df['count']>5]
        else:
            df=df



    
        df=df[ df['imgpatch'].apply(lambda x: x is not None)]
        # uncomment below line for import nan value samples in flood period
        # dfnone =df[df.index.isin(self.invalid_indices)
        #imgtoarray=np.array(dfnone['imgpatch'].tolist())
        # labeltoarray=np.array(dfnone['class'].tolist())

        imgtoarray=np.array(df['imgpatch'].tolist())

      
        labeltoarray=np.array(df['class'].tolist())

      
        if self.category==1:
            
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                # np.save(config['data']['trainx_path']+f'{imgtoarray.shape[0]}.npy',imgtoarray)
                # np.save(config['data']['trainy_path']+f'{labeltoarray.shape[0]}.npy',labeltoarray)
                config['data']['trainx_path']=os.path.join(config['data']['savexyarray'],f'addcount0trainx401510620801{patchsize}_{imgtoarray.shape[0]}x.npy')
                config['data']['trainy_path']=os.path.join(config['data']['savexyarray'],f'addcount0trainy401510620801{patchsize}_{labeltoarray.shape[0]}y.npy')
                np.save(config['data']['trainx_path'],imgtoarray)
                np.save(config['data']['trainy_path'],labeltoarray)
                # yaml.dump(config, file) #追加
                # 更新配置文件
                with open(config_path, 'w') as file:
                    yaml.dump(config, file)
        if self.category==0:
             with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                if c in [0,1,2]:
                    config['data'][f'valx{c}_path']=os.path.join(config['data']['savexyarray'],f'medianval3countflood{c}_{patchsize}_{imgtoarray.shape[0]}x.npy')
                    config['data'][f'valy{c}_path']=os.path.join(config['data']['savexyarray'],f'medianval3countflood{c}_{patchsize}_{labeltoarray.shape[0]}y.npy')
                
                    np.save(config['data'][f'valx{c}_path'],imgtoarray)
                    np.save(config['data'][f'valy{c}_path'],labeltoarray)
                else:
                   
                    config['data'][f'valx_path']=os.path.join(config['data']['savexyarray'],f'15ndsvirepmedianvalx_{patchsize}_{imgtoarray.shape[0]}x.npy')
                    config['data'][f'valy_path']=os.path.join(config['data']['savexyarray'],f'15ndsvirepmedianvaly_{patchsize}_{labeltoarray.shape[0]}y.npy')
                
                    np.save(config['data'][f'trainx_path'],imgtoarray)
                    np.save(config['data'][f'trainy_path'],labeltoarray)
                    # config['data']['valx_path']=config['data']['valx_path']+f'{imgtoarray.shape[0]}.npy'
                    # config['data']['valy_path']=config['data']['valy_path']+f'{labeltoarray.shape[0]}.npy'
                    # yaml.dump(config, file)6
                with open(config_path, 'w') as file:
                    yaml.dump(config, file)
   

  
           
    

        return  imgtoarray,labeltoarray
    
    



    

'''
The result is datasetRF\pathsize11
spilt train and val based on the grid id;
Also has one mean std normination 
y without onehot encoder
x without nan fill. 



'''
import argparse
if  __name__ == "__main__":
    print(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Process command-line arguments.')
    parser.add_argument('--work_dir', type=str, default=r"D:\DLRicemap\transfer\sc_cq\2021_4_9noref\array", help='Path to save label x')
    # parser.add_argument('--sampledf', type=str, default=r"D:\DLRicemap\transfer\sc_cq\sc_cqtrain965_935.csv", help='Path to save label y') #
    # parser.add_argument('--modeltitle', type=int, default=1, help='help to label different training')
    parser.add_argument('--trainlaebl1', type=str, default=r"D:\DLRicemap\datasets\samples\guanansamplev2.shp", help='Path to save label y')
    parser.add_argument('--trainlaebl2', type=str, default=r"D:\DLRicemap\datasets\samples\guanansample401510count0.shp", help='Path to save label y')

    parser.add_argument('--saveimgtitle', type=str,  default='valsc_cqimgnoref_', help='Start row, end row, start col,end col, separated by commas')
    parser.add_argument('--savelabeltitle', type=str,  default='valsc_cqlabelnoref_', help='Start row, end row, start col,end col, separated by commas')
    args = parser.parse_args()
    
    time_size=1
    band_size=13
    # data_dir=os.path.join(work_dir,f'datasetRF\pathsize{patchsize}')
    # traindata_dir=os.path.join(data_dir,'traingrid61324')
    # valdata_dir=os.path.join(data_dir,'valgrid1729')
    # os.makedirs(data_dir,exist_ok=True)
    # os.makedirs(traindata_dir,exist_ok=True)
    # os.makedirs(valdata_dir,exist_ok=True)
   
    #D:\ricemodify\dataset\s1s2medianmaxmincloudmask\ynomeanstd5196.npy
    #img_yuechi_path =r"D:\DLRicemap\transfer\sc_cq\2021_4_9noref"
    config_path=r'D:\DLRicemap\configindepend.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        img_yuechi_path=config['data']['originimgfolder']    
        patchsize=config['data']['patchsize'] 
           
    year=2021
    data_dir=os.path.join(args.work_dir)
    os.makedirs(data_dir,exist_ok=True)
    patchsize = 11

    with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            trainlabel1,vallabel=config['data']['origin_trainlabel'],config['data']['origin_vallabel']
            trainlabel2=args.trainlaebl2

    sampledf=sample(r"D:\DLRicemap\datasets\samples\guanansample.shp",trainlabel2)
    #sampledf=sample(r"D:\DLRicemap\dataset\yuechival557.csv")
   
    for label in [trainlabel1]:
        if label==trainlabel1:
            category=1
        elif label==vallabel:
             category=0
        img,trans,band_size=load_tiles(img_yuechi_path)
        data_loader = PatchDataLoader(category,sampledf, img,  trans,band_size,patchsize=patchsize,year=year,time_size=time_size,saveimgtitle=args.saveimgtitle,savelabeltitle=args.savelabeltitle)
        if category==1:
            for c in ['full']:
                # print(c)
                x,y=data_loader.generate_patches(c)
        else:
            x,y=data_loader.generate_patches('full')
        # patch_df = data_loader.get_data_frame()
        # print('xy',x.shape,y.shape)

   
