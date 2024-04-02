

import tensorflow as tf
from tensorflow import keras
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
import os
import random
import numpy as np
import time
from tensorflow.keras.losses import categorical_crossentropy
from src.dataprocess.dataloader import dataprogressindependval,dataagument
from utils.models import *




def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()

def cycle(iterable):
    # This function creates an infinite iterator
    while True:
        for x in iterable:
            yield x

def calculate_imou(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    minimum_union = np.sum(np.logical_or(y_true, y_pred))
    imou = intersection / minimum_union
    return imou

def iou_score(y_true, y_pred):
    # if USE_MASK:
    #     y_true, y_pred = _masked_tensor(y_true, y_pred)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    # if LABEL_SMOOTH:
    #     y_true = tf.clip_by_value(y_true, 0.05, 0.95)
    # y_true = tf.where(y_true >= 0.5, tf.ones_like(y_true), tf.zeros_like(y_true))
    intersection = tf.reduce_sum(y_true * y_pred)
    iou = (intersection + 1) / (tf.reduce_sum(y_true + y_pred) - intersection + 1)

    return iou


def iou_loss(y_true, y_pred):
    return 1 - iou_score(y_true, y_pred)
           

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 计算混淆矩阵
def confusion_matrix1(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
 

    # 计算生产者精度
    producers_accuracy = np.diag(cm) / np.sum(cm, axis=1)

    # 计算用户精度
    users_accuracy = np.diag(cm) / np.sum(cm, axis=0)
    # print(producers_accuracy,users_accuracy)
     # Plot confusion matrix using seaborn
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()
    a=3
    return producers_accuracy, users_accuracy

def weightsslf(epoch,a,warmupepoch=20):
    if epoch<=warmupepoch:
        weight=0
    elif epoch<30:
        weight=a*(epoch-warmupepoch)/(30-warmupepoch)
    else:
        weight=a
    return weight

def train(model,train_dataset,epoch,lr_strategy,optimizer):
    total_loss = 0
    total_correct=0
    batch_step=0
    total_samples=0
    loss_pseudo=0
    predictionlist=[]
    ylist=[]
    supervisedepochslosslist=[]
    semiviselosslist=[]
    globalstep=0
    
    for x,y,z in train_dataset:#zip(xlabelarray, ylabel):
        x_batch_train=x
        y_batch_train=y
        x_center_train= z
    
        if x_batch_train.shape[0]!=64:
            continue
        total_samples+=x_batch_train.shape[0]
        with tf.GradientTape() as tape:
            if lr_strategy=='exponentialdecay':
                #衰减从第0步开始
                lr = lr_schedule10x95(globalstep).numpy()
                globalstep+=1
            if lr_strategy=='constant':
                lr=0.0001
            optimizer.learning_rate = lr
   
            if modelname=='dualsparables2Cnn2ds1cnn1d':
          
                logits_ed = model([x_batch_train,x_center_train],training=True)
            elif modelname=='dualsparableCnn2d':
                logits_ed = model(x_batch_train,training=True)
            # logits_ed = model(x_batch_train,training=True)
            # print('logits_ed',logits_ed.shape)
            loss_ed = tf.keras.losses.categorical_crossentropy(y_batch_train, logits_ed,from_logits=False) # 32个样本的损失
            supervisedepochslosslist.append(loss_ed)
            loss_ed=tf.reduce_mean(loss_ed)
        
            # 计算预测正确的样本数量
            predictions = tf.argmax(logits_ed, axis=-1)
            predictionlist.append(predictions)
            ylist.append(y_batch_train)
            correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(y_batch_train, axis=-1)), dtype=tf.float32))
            # print('correct',correct)
            total_correct += correct

        #利用每个batch中的混合损失更新梯度，然后更新模型参数
        grads = tape.gradient(loss_ed, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 累计一轮中多个batch的损失
        total_loss += loss_ed
        # print(f'batchstep{batch_step},total_loss is:',total_loss)
        batch_step += 1

    avg_loss=total_loss/batch_step
    accuracy = total_correct / total_samples
    ylist=np.concatenate(ylist)
    predictionlist=np.concatenate(predictionlist)
    # print(ylist.shape,predictionlist.shape)
    
  # 虽然最后一个batch的样本数量可能不一样，但是由于之前是在每个batch取损失均值，最后可以将batch损失均值的总和除以batch数量；因为每个样本的权重一样。
    # 也可以不对batch的损失进行平均，直接计算每个batch中的总损失，然后直接将所有batch的总损失除以样本总数。
    # print(f"Epoch {epoch}, Loss: {total_loss.numpy()/batch_step}")
    # current_learning_rate = optimizer.learning_rate(optimizer.iterations)

           
    return model,avg_loss,accuracy,batch_step,lr

def val(model,xval,yval):
    # print(xval.shape,yval.shape)

    total_correct = 0
    total_samples = 0
    # val_loss = 0
    batch_step=0
    if modelname=='dualsparables2Cnn2ds1cnn1d':
        logits_val = model.predict([xval,xvalcenter])      
    elif modelname=='dualsparableCnn2d':
        logits_val = model.predict(xval)   #model(x_batch_val, training=False)
    # valloss =binary_weighted_cross_entropy(y_batch_val, logits_val,1)#
    valloss=tf.keras.losses.categorical_crossentropy(yval, logits_val,from_logits=False)
    # print(valloss)
    val_loss=tf.reduce_mean(valloss)
    # 计算预测正确的样本
    predictions = tf.argmax(logits_val, axis=-1)
    # print(predictions.shape)
    correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(yval, axis=-1)), dtype=tf.float32))
    total_correct += correct
    total_samples += xval.shape[0]
    batch_step+=1
    # val_loss+=valloss
    # 计算预测的准确性
    # print('batch_step',batch_step)
    accuracy = total_correct / total_samples
    p,u=confusion_matrix1(tf.argmax(yval, axis=-1),predictions)
    # val_loss=val_loss/batch_step
    # losslist.append()
    # print(f"epoch{epoch}Validation Accuracy: {accuracy.numpy()}")
    return accuracy,val_loss,p,u
def learningrate(lr_strategy,initial_lr=0.0001, total_epochs=30):
        if lr_strategy=='cosine':
            lr = cosine_learning_rate(0, initial_lr, total_epochs)
        if lr_strategy=='exponentialdecay':
            lr = lr_schedule10x95
        if lr_strategy=='constant':
            lr = initial_lr
        print('lr',lr)
    
        return lr
def main1(savejpgpath,lr_strategy,lr_schedule,epochs,cnn2dfilters,cnn1dfilters,multiscales,attentions,cnn1dattention,fusionattention,dropout):
    # lr_schedule=learningrate(lr_strategy, initial_lr=0.0001, total_epochs=epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if modelname=='dualsparableCnn2d':
        model=dualsparableCnn2d(inputshape2d,cnn2dfilters=cnn2dfilters,cnn1dfilters=cnn1dfilters,multiscales=multiscales,attentions=attentions,cnn1dattention=cnn1dattention,fusionattention=fusionattention)
        
        print(model.summary())
    if modelname=='dualsparables2Cnn2ds1cnn1d':
        model=dualsparableseparatenorCnn2dcnn1d(inputshape2d,inputshape1d,cnn2dfilters=cnn2dfilters,cnn1dfilters=cnn1dfilters,multiscales=multiscales,attentions=attentions,cnn1dattention=cnn1dattention,dropout=dropout)
        
    if modelname=='resnet':
        model=resnetattention(inputshape,cnn2dfilters=[16,32,64,128,256],cnn1dfilters=[32,64,128,256],two=False,attention=1,dropout=0)
    if modelname=='singlecnn1d':
        model=cnn1dsingle(inputshape,num_filters=[32,64,128,256],cnn1dattention=0,dropout=0)
    model.compile(optimizer=optimizer,loss=categorical_crossentropy, metrics=['accuracy'])
    best_val_loss= float('inf')
    # global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    for epoch in range(1,epochs+1):
   
        train_dataset=tf.data.Dataset.from_tensor_slices((xlabelarray,ylabelarray,xtraincenter)).batch(64)
        train_dataset=train_dataset.shuffle(xlabelarray.shape[0])
        # Attempt to restore the latest checkpoint
        model,avg_loss,accuracy,batch_step,lr=train(model,train_dataset,epoch,lr_strategy,optimizer)
        # global_step+=batch_step
          # Use assign_add for incrementing global_step
        global_step.assign_add(batch_step)
       
        current_learning_rate = optimizer.learning_rate(global_step).numpy()

        print(f"Epoch {epoch}, Learning Rate: {current_learning_rate.numpy()}")

        print("Epoch:", epoch, "Learning rate:", optimizer.learning_rate(globalstep).numpy())
        valaccuracy,valloss,p,u=val(model,xval,yval)
        # print(p,u)
        if valloss < best_val_loss:
            best_val_loss = valloss
            # savepath=os.path.join(savemodelpath,f'{saveVersion}.h5')
            # prinsavemodeldir,f'{saveVersion}.h5')modelpath)
            model.save(savemodelpath,include_optimizer=True)
            # Define global_step and checkpointing
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model,global_step=global_step)
        ckpt.step.assign_add(1)
        manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=3)
        # global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3) 
        def train_and_checkpoint(net, manager):
            ckpt.restore(manager.latest_checkpoint)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

            for _ in range(50):
                ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                   

        
        # save_path = manager.save()
        # print(f"Saved checkpoint for step {int(global_step)}: {save_path}")
        # res=checkpoint.restore(save_path)
        # Attempt to restore the latest checkpoint
        
# Create the checkpoint object
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        latest_checkpoint = r"D:\DLRicemap\src\tf_ckpts\ckpt-1.index"
        if latest_checkpoint:
                checkpoint.restore(latest_checkpoint)
                print(f"Restored from {latest_checkpoint}")
                print(checkpoint)
                print(f"{global_step.numpy()} restored")
                # print(f"{checkpoint.global_step} restored")
                print(f"{checkpoint.model.optimizer.learning_rate(global_step).numpy()}")
                start_epoch = global_step.numpy()  # Adjust based on your batch processing
                print(f"Start epoch: {start_epoch}")
        else:
                print("Initializing from scratch.")
                start_epoch = 0

            # Now, call your train function, passing the start_epoch and global_step if needed
        # avg_loss, accuracy, valaccuracy, valloss, p, u = train(model, train_dataset, epoch=epoch, lr_strategy='constant', optimizer=optimizer, start_epoch=start_epoch, global_step=global_step.numpy())

        metricdf.at[epoch, "trainloss"] = avg_loss.numpy()
        metricdf.at[epoch, "trainaccuracy"] = accuracy.numpy()
        metricdf.at[epoch, "valaccuracy"] = valaccuracy.numpy()
        metricdf.at[epoch, "valloss"] = valloss.numpy()
        metricdf.at[epoch, "norice_producer_accuracy"] = p[0]
        metricdf.at[epoch, "norice_user_accuracy"] = u[0]
        metricdf.at[epoch, "rice_producer_accuracy"] = p[1]
        metricdf.at[epoch, "rice_user_accuracy"] = u[1]

        metricdf.at[epoch, "lr"] = lr
        metricdf.to_csv(os.path.join(savemodeldir1,f'{current_date}.csv'),index=False)
   
    
    plt.figure(figsize=(10,8))
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)

    plt.plot( metricdf["trainloss"])
    plt.plot( metricdf["valloss"])
    plt.title('val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(metricdf["trainaccuracy"])
    plt.plot(metricdf["valaccuracy"])
    plt.title('accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.subplots_adjust(hspace=0.6)
    plt.suptitle(f"{saveVersion}")
    plt.savefig(savejpgpath)
    plt.show()
    time.sleep(10)
    # a=3
    # bool(print(a))



if __name__ == '__main__':
    # data
    batch_size=11
    savelabelxPath=r"D:\DLRicemap\dataset\2022pathsize17\samplestrainimage794.npy"#"D:\ricemodify\limiteddataset\2022pathsize11\labelsamplesx457.npy"
    savelabelyPath=r"D:\DLRicemap\dataset\2022pathsize17\samplestrainlabel794.npy"#"D:\ricemodify\limiteddataset\2022pathsize11\labelsamplesy457.npy"
    savevalxPath=r"D:\DLRicemap\dataset\2022pathsize17\samplesvalimage407.npy"
    savevalyPath=r"D:\DLRicemap\dataset\2022pathsize17\samplesvallabel407.npy"
   
    tf.keras.utils.set_random_seed(999)
    tf.config.experimental.enable_op_determinism()

    # save log setting
    current_date = datetime.now().strftime('%m-%d %H:%M')
    current_date = current_date .replace(' ', '-').replace(':', '-')
    print(current_date)
    savemodeldir=r"D:\DLRicemap\run\log"
    os.makedirs(savemodeldir,exist_ok=True)
     
    
    metricdf=pd.DataFrame(columns=["index","trainloss", "trainaccuracy",'valaccuracy','valloss','rice_producer_accuracy','rice_user_accuracy','norice_producer_accuracy','norice_user_accuracy'])

    # ssl
    threshold=0.95
    warmupepoch=20
    weightssl=1
    # optimizer
    step_everyepoch=12
    lr_schedule10x95 = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=10*12, # 每10抡衰减一次
        decay_rate=0.95,
        staircase=True)
  
    def cosine_learning_rate(epoch, initial_lr=0.0001, total_epochs=30):
        lr = initial_lr * 0.5 * (1 + tf.math.cos(epoch / total_epochs * 3.1415))
        return lr

   

   
    # model=dualsparableCnn2d(inputshape,[64,128,256],[32,64,128,256],[0,0,0],[1,1,1])
    cnn2dfilters=[64,128,256]
    cnn1dfilters=[32,64,128,256]
    modelname='dualsparableCnn2d'#'singlecnn1d'
    if modelname=='dualsparables2Cnn2ds1cnn1d':
        xlabel,ylabel,xval,yval,xtraincenter,xvalcenter=dataprogressindependval(savelabelxPath, savelabelyPath,savevalxPath, savevalyPath,batch_size,1)
        xlabel=xlabel[...,0:-5]
        xval=xval[...,0:-5]
        xtraincenter = xtraincenter[...,-6:]
        xvalcenter = xvalcenter[...,-6:]
        print('xtraincenter,xvalcenter',xtraincenter.shape,xvalcenter.shape)
    elif modelname=='dualsparableCnn2d':
        xlabel,ylabel,xval,yval,xtraincenter,xvalcenter=dataprogressindependval(savelabelxPath, savelabelyPath,savevalxPath, savevalyPath,batch_size,0)
    xlabelarray,ylabelarray=dataagument(xlabel,ylabel,xuntrain=None,p=0.5,strong=1)
       

    attentions=[1,1,1]
    lr_strategy='constant'
    for  lr_strategy in ['exponentialdecay']:
        for modelname in ['dualsparableCnn2d']:
            
            for attentions in [[0,0,1]]:
                for batch_size in [11]:
                    # for loop中的参数是全局的
                    for multiscales in [[0,0,1]]:
                        
                        if lr_strategy=='exponentialdecay':
                            lr_schedule=lr_schedule10x95
                        if lr_strategy=='constant':
                            lr_schedule=0.0001
                    
                       
                       
                        h,w,c=xlabelarray.shape[-3],xlabelarray.shape[-2],xlabelarray.shape[-1]
                        inputshape2d=(h,w,c)
                        inputshape1d=(1,1,3)
                        # xlabelarray,ylabelarray=xtraincenter,ylabel
                        # xval=xvalcenter
                        # print(xlabelarray.shape,ylabelarray.shape,xval.shape,yval.shape) #(794, 1, 1, 16) (794, 2) (407, 1, 1, 16) (407, 2)
                    
                        saveVersion=f'{current_date}-{modelname}cnn2dcnn1dmeanstdnobc_nofuseattent-16bandsrandvhrsbc_multiscale{"x".join(map(str,multiscales))}-att{"x".join(map(str,attentions))}-lr{lr_strategy}_'
                        savemodeldir1=os.path.join(savemodeldir,f'{saveVersion}')
                        os.makedirs(savemodeldir1,exist_ok=True)
                        savemodelpath=os.path.join(savemodeldir1,f'{saveVersion}.h5')
                        savejpgpath=os.path.join(savemodeldir1,f'{current_date}.jpg')
                        main1(savejpgpath,lr_strategy,lr_schedule,50,cnn2dfilters,cnn1dfilters,multiscales,attentions,cnn1dattention=0,fusionattention=1,dropout=0.2)
                        main1(savejpgpath,lr_strategy,lr_schedule,10,cnn2dfilters,cnn1dfilters,multiscales,attentions,cnn1dattention=0,fusionattention=1,dropout=0.2)
                        main1(savejpgpath,lr_strategy,lr_schedule,10,cnn2dfilters,cnn1dfilters,multiscales,attentions,cnn1dattention=0,fusionattention=1,dropout=0.2)
                        
                    






