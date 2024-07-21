import tensorflow as tf
from tensorflow import keras
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
import os
import random
import numpy as np
import yaml
from tensorflow.keras.losses import categorical_crossentropy
from keras import backend as K

from models import *
from plot import *
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loss import *
from sklearn.model_selection import KFold


'''https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()

def evaluate_model(model, xval, yval, maskingvalue,maskval):
    """Evaluates the model on the validation set.

    Args:
        model: The trained model.
        xval: Validation features.
        yval: Validation labels.
        maskval: Validation mask.

    Returns:
        A tuple containing:
            - accuracy: Overall accuracy.
            - val_loss: Validation loss.
            - f1_score: F1 score.
            - kappa: Cohen's Kappa.
            - precision: Precision.
            - recall: Recall.
    """
    total_correct = 0
    total_samples = 0
    logits_val = model.predict(xval,verbose=0)  
    # print('logits_val',logits_val)
    # valloss=tf.keras.losses.categorical_crossentropy(yval, logits_val,from_logits=False)
    if maskingvalue:
        valloss=nomask_binary_weighted_cross_entropy(yval, logits_val,1)
    else:
        valloss=nomask_binary_weighted_cross_entropy(yval, logits_val, beta=1.)
    # print('loss_val',valloss)
    # # valloss=focal_tversky_loss(yval, logits_val,0.0001)
    # 计算预测正确的样本数量
    predictions = tf.argmax(logits_val, axis=-1)
    # print(predictions.shape)
    conf_matrix = confusion_matrix(tf.argmax(yval, axis=-1), predictions)
    # 计算生产者精度（Producer's Accuracy）
    producer_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    # 计算消费者精度（Consumer's Accuracy）
    consumer_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=0)
    # 计算总体精度（Overall Accuracy）
    overall_accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()
    # 计算 Cohen's Kappa
    kappa = cohen_kappa_score(tf.argmax(yval, axis=-1), predictions)
    # print('kappa', kappa)
    correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(yval, axis=-1)), dtype=tf.float32))
    total_correct += correct
    total_samples += xval.shape[0]
    # val_loss+=valloss
    # 计算预测的准确性
    # print('batch_step',batch_step)
    # print('total_correct',total_correct)
    # print('total_samples',total_samples)
    accuracy = total_correct / total_samples
    p,u=confusion_matrix1(tf.argmax(yval, axis=-1),predictions)

    return accuracy,valloss,p,u,kappa

def train_step(model, optimizer, x_batch_train, y_batch_train, maskingvalue,x_mask_train):
    """Performs a single training step.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        x_batch_train: A batch of training features.
        y_batch_train: A batch of training labels.
        x_mask_train: A batch of training mask.

    Returns:
        The loss value for the current batch.
    """
    with tf.GradientTape() as tape:
        logits_ed = model(x_batch_train, training=True)
        # if maskingvalue:
        #     loss_ed=binary_weighted_cross_entropy(y_batch_train, logits_ed,x_mask_train,1.)
        # else:
        loss_ed=nomask_binary_weighted_cross_entropy(y_batch_train, logits_ed, beta=1.)
  
    grads = tape.gradient(loss_ed, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_ed,model

def pmeanstd(df,jpgpath):
    # 绘制带误差棒的验证精度曲线
    # 设置图形和子图
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    plt.rcParams['font.family'] = 'Times New Roman' 
    from matplotlib.font_manager import FontProperties
    import matplotlib.patches as mpatches
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
    meanvalacc,meankappa,meanrf1,meannrf1,meanF1,meanru,meanrp=df['meanvalacc'],df['meankappa'],df['meanrf1'],df['meannrf1'],df['meanF1'],df['meanru'],df['meanrp']
    stdevvalacc,stdevkappa,stdevrf1,stdevnrf1,stdevF1,stdevru,stdevrp=df['stdevvalacc'],df['stdevkappa'],df['stdevrf1'],df['stdevnrf1'],df['stdevF1'],df['stdevru'],df['stdevrp']

    # axs[0,0].errorbar(range(1, len(mean_val_loss) + 1), mean_val_loss, yerr=std_val_loss, fmt='-o', capsize=5,label='valloss')
    axs[0,0].errorbar(range(1, len(meanvalacc) + 1), meanvalacc, yerr=stdevvalacc, fmt='-o', capsize=5,label='Val Accuracy')
    # axs[0,0].errorbar(range(1, len(mean_train_loss) + 1), mean_train_loss, yerr=std_train_loss, fmt='-o', capsize=5,label='trainloss')
    # axs[0,0].errorbar(range(1, len(meantrainacc) + 1), meantrainacc, yerr=stdtrainacc, fmt='-o', capsize=5,label='Train Accuracy')
    # plt.fill_between(range(1,len(mean_val_acc) + 1), mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='orange', alpha=0.2)
    # axs[0, 0].set_title('Lossandaccuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Lossandaccuracy')
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(False)
    
    axs[0,1].errorbar(range(1, len(meanrp) + 1), meanrp, yerr=stdevrp, fmt='-o', capsize=5,label='PA(%)')
    # axs[0,1].errorbar(range(1, len(mean_val_nrp) + 1), mean_val_nrp, yerr=std_val_nrp, fmt='-o', capsize=5,label='norice produceracc')
    axs[0,1].errorbar(range(1, len(meanru) + 1), meanru, yerr=stdevru, fmt='-o', capsize=5,label='UA(%)')
    # axs[0,1].errorbar(range(1, len(mean_val_nru) + 1), mean_val_nru, yerr=std_val_nru, fmt='-o', capsize=5,label='norice useracc')
    # plt.fill_between(range(1,len(mean_val_acc) + 1), mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='orange', alpha=0.2)
    # axs[0, 1].set_title('Riceproducer')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Valproducer')
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].grid(False)

    axs[1,0].errorbar(range(1, len(meankappa) + 1), meankappa, yerr=stdevkappa, fmt='-o', capsize=5,label='Kappa')
    # plt.fill_between(range(1,len(mean_val_acc) + 1), mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='orange', alpha=0.2)
    # axs[1, 0].set_title('kappa')
    axs[1, 0].set_xlabel('epoch')
    axs[1, 0].set_ylabel('kappa')
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].grid(False)

    axs[1,1].errorbar(range(1, len(meanrf1) + 1), meanrf1, yerr=stdevrf1, fmt='-o', capsize=5,label='riceF-Score')
    axs[1,1].errorbar(range(1, len(meannrf1) + 1), meannrf1, yerr=stdevnrf1, fmt='-o', capsize=5,label='noriceF1')
    axs[1,1].errorbar(range(1, len(meanF1) + 1),meanF1, yerr=stdevF1, fmt='-o', capsize=5,label='MacroF1')
    # plt.fill_between(range(1,len(mean_val_acc) + 1), mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='orange', alpha=0.2)
    # axs[1, 1].set_title('producer')
    axs[1, 1].set_xlabel('epoch')
    axs[1, 1].set_ylabel('valproducer')
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].grid(False)

    # 调整子图布局
    plt.tight_layout()
    plt.legend()
    # 显示图像
    plt.savefig(jpgpath)
    # plt.show()
def train_and_evaluate(model, x, y,  maskingvalue,epochs, batch_size, savemodeldir,inputshape2d,modelname,dropout,cbrm,period):
    """Trains and evaluates the model using KFold cross-validation.

    Args:
        model: The model to train.
        x: Training features.
        y: Training labels.
        mask: Training mask.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        savemodeldir: Directory to save the best model.
        lr_schedule: Learning rate scheduler (optional).
    """
    kfold = KFold(n_splits=5, shuffle=False) #random_state=42
    # Define optimizer and compile (you need to do this for each new model)
    
    metricdf = pd.DataFrame(columns=["index"])
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for fold_no,(trainindex, testindex) in enumerate(kfold.split(x)):
        if modelname == 'cnn2d':
            model = hycnn2d(inputshape=inputshape2d, drop=dropout, cbrm=cbrm, fusion=1, single1=period,maskmissing=maskingvalue)
            print(model.summary())
        elif modelname == 'cnn1d':
            model = hycnn1d(inputshape=inputshape2d, drop=dropout, fusion=0, single1=period,maskmissing=maskingvalue)
        else:
            raise ValueError("Invalid model name")

        lr = config['train']['lr_schedule']['initial_learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if maskingvalue:
            model.compile(optimizer=optimizer, loss=nomask_binary_weighted_cross_entropy, metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer, loss=nomask_binary_weighted_cross_entropy, metrics=['accuracy'])

        # Split data and mask into train/test sets for the current fold
        # xtrain, xval = tf.gather(x, trainindex), tf.gather(x, testindex)
        ytrain, yval = tf.gather(y, trainindex), tf.gather(y, testindex)
        xtrain,xval=x[trainindex],x[testindex]
        # ytrain,yval=y[trainindex],y[testindex]
        masktrain,maskval= tf.where(xtrain== 0, 0, 1), tf.where(xval== 0, 0, 1)
        masktrain,maskval= tf.cast(masktrain, dtype=tf.float32) ,tf.cast(maskval, dtype=tf.float32)  # 将 mask 转换为 float32,(2654, 1, 1, 26)
        # print('masktrain',masktrain.shape)
        # Create a data pipeline for training
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, masktrain, ytrain)).batch(batch_size)
        # # Loop over epochs
        for epoch in range(1, epochs + 1):
            # Train the model
            total_loss = 0
            batch_step = 0
            # for x1, x2, y in train_dataset:
            #     if x1.shape[0]!=batch_size:
            #         continue
            #     loss_ed,model= train_step(model, optimizer, x1, y, x2)
            for x1, x2, y1 in train_dataset:
                if x1.shape[0]!=batch_size:
                    continue
                loss_ed,model= train_step(model, optimizer, x1, y1,maskingvalue, x2)
                total_loss += loss_ed
                batch_step += 1
            # print('batch_step: {batch_step}', batch_step)
            # print('loss_ed: {loss_ed}', loss_ed)
            # Evaluate the model
            val_accuracy,val_loss,p,u,kappa = evaluate_model(model, xval, yval, maskingvalue,maskval)

            # Print training and validation results
            # print(f"Epoch {epoch}, Fold {fold_no}:")
            # print(f"   Train Loss: {total_loss / batch_step:.4f}")
            # print(f"   Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_accuracy
                model.save(os.path.join(savemodeldir, 'best_model.h5'), include_optimizer=True)

            # Update metric dataframe
            rf1,nrf1=2*p[1]*u[1]/(p[1]+u[1]),2*p[0]*u[0]/(p[0]+u[0])
            metricdf.at[epoch, f"train{fold_no}loss"] = (total_loss / batch_step).numpy()
            metricdf.at[epoch, f"val{fold_no}loss"] = val_loss.numpy()
            metricdf.at[epoch, f"train{fold_no}acc"] = val_accuracy.numpy()
            metricdf.at[epoch, f"val{fold_no}acc"] = val_accuracy.numpy()
            metricdf.at[epoch, f"{fold_no}kappa"] = kappa
            metricdf.at[epoch, f"{fold_no}ricef1"] =rf1
            metricdf.at[epoch, f"{fold_no}noricef1"] = nrf1
            metricdf.at[epoch, f"{fold_no}Meanf1"] = (rf1+nrf1)/2
            metricdf.at[epoch, f"{fold_no}riceproducer"] = p[1]
            metricdf.at[epoch, f"{fold_no}noriceproducer"] =p[0]
            metricdf.at[epoch, f"{fold_no}riceuser"] = u[1]
            metricdf.at[epoch, f"{fold_no}noriceuser"] =u[0]
   
            
            # Save metrics dataframe            metricdf.to_csv(os.path.join(savemodeldir, 'metrics.csv'), index=True)
        # print(f"Fold {fold_no} completed.")
    metricdf['meanvalloss']=metricdf[[f"val{fold_no}loss" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevvalloss']=metricdf[[f"val{fold_no}loss" for fold_no in range(5)]].std(axis=1)

    metricdf['meanvalacc']=metricdf[[f"val{fold_no}acc" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevvalacc']=metricdf[[f"val{fold_no}acc" for fold_no in range(5)]].std(axis=1)

    metricdf['meankappa']=metricdf[[f"{fold_no}kappa" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevkappa']=metricdf[[f"{fold_no}kappa" for fold_no in range(5)]].std(axis=1)

    metricdf['meanrf1']=metricdf[[f"{fold_no}ricef1" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevrf1']=metricdf[[f"{fold_no}ricef1" for fold_no in range(5)]].std(axis=1)
    metricdf['meannrf1']=metricdf[[f"{fold_no}noricef1" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevnrf1']=metricdf[[f"{fold_no}noricef1" for fold_no in range(5)]].std(axis=1)

    metricdf['meanF1']=metricdf[[f"{fold_no}Meanf1" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevF1']=metricdf[[f"{fold_no}Meanf1" for fold_no in range(5)]].std(axis=1)

    metricdf['meanrp']=metricdf[[f"{fold_no}riceproducer" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevrp']=metricdf[[f"{fold_no}riceproducer" for fold_no in range(5)]].std(axis=1)
    metricdf['meanru']=metricdf[[f"{fold_no}riceuser" for fold_no in range(5)]].mean(axis=1)
    metricdf['stdevru']=metricdf[[f"{fold_no}riceuser" for fold_no in range(5)]].std(axis=1)

    metricdf.to_csv(os.path.join(savemodeldir, 'metrics.csv'), index=True)
    # pmeanstd(metricdf, os.path.join(savemodeldir, 'metrics.csv'))
     


       
    
    # Save best model
    # model.save(os.path.join(savemodeldir, 'best_model.h5'), include_optimizer=True) 

def main():
    config_path=r'D:\DLRicemap\configindepend.yaml'
    with open(config_path, 'r') as file:
        global config
        config = yaml.safe_load(file)

    savelabelxPath = config['data']['trainx_path'] 
    savelabelyPath = config['data']['trainy_path']
    for model, model_params in config["models_params"].items(): 
        for dropout in model_params['params']['dropout']:
            for batch_size in model_params['params']['batch_size']:
                for epochs in model_params['params']['epochs']:
                    for modelname in model_params['params']['modelname']:
                        for cbrm in model_params['params']['cbrm']:
                            for maskingvalue in model_params['params']['maskingvalue']:
                                for ratio in model_params['params']['ratio']:
                                    best_loss = float('inf')
                                    savemodeldir0 = config['train']['model_dir']
                                    os.makedirs(savemodeldir0,exist_ok=True)
                                    if modelname=='cnn1d':
                                        cbrm=0
        
                                    print('maskingvalue',maskingvalue,dropout)
                                    model_name = '_'.join([f'addcount0_fusionnewpatch9_dropflood{ratio}'+'period'+str('2'),modelname,str(batch_size),'satt'+str(cbrm), 'drop'+str(dropout)]) # 
                                    savemodeldir = os.path.join(savemodeldir0,model_name)
                                    # if os.path.exists(savemodeldir):
                                    #     continue
                                    os.makedirs(savemodeldir,exist_ok=True)
                                    print(savemodeldir)
                                    savemodelpath = os.path.join(savemodeldir,f"{model_name}.h5")
                                    savejpgpath=os.path.join(savemodeldir,f'{model_name}.jpg')
                                    
                                    # print('savemodelpath',savemodelpath)
                                    config['train']['model_path']=savemodelpath
                                    with open(config_path, 'w+') as file:
                                        yaml.dump(config, file)
                                    print('x',savelabelxPath)
                                    x, y=dataprogressindependval(modelname,savelabelxPath, savelabelyPath,savemodeldir=savemodeldir,period=2)
                                    from randomdrop import randomdrop_features
                                    x,y=randomdrop_features(x,y,ratio)

                                    
                                    
                                   
                                    # 计算每个样本的有效像素数量
                                    mask1=tf.where(x== 0, 0, 1)
                                    mask1=tf.cast(mask1,tf.float32)
                                    valid_pixels = tf.reduce_sum(mask1, axis=[0,1, 2, 3]) 
                                    # sample*heights*widths*channels
                                    print(tf.cast(tf.size(mask1),tf.float32))
                                    valid_ratio = valid_pixels / tf.cast(tf.size(mask1), tf.float32)
                                    print('valid_ratio',valid_ratio)
                                    print('----------------------------------')
                                    inputshape2d=(x.shape[1],x.shape[1],x.shape[-1])
                                    # with open(save_path, 'w+') as file:
                                    #     yaml.dump(config, file)
   
    
                                    # Train and evaluate the model
                                    train_and_evaluate(model, x, y,  maskingvalue,epochs, batch_size, savemodeldir,inputshape2d,modelname,dropout,cbrm,period=2)
                                    # modelpath=r"D:\DLRicemap\log\train\401501620801\addcount0_mask0lossnovaild_fusionnewpatch7_dropflood0.0period2_cnn2d_64_satt0_drop0.2\best_model.h5"
                                    # model=tf.keras.models.load_model(modelpath,custom_objects={"K": K,"nomask_binary_weighted_cross_entropy":nomask_binary_weighted_cross_entropy})
                                    # predictions, inverse_attention_weights = model.predict([x,mask1],verbose=0)
                                    # 获取 inverse_attention_weights 层的输出
#                                     print(model.summary())
#                                     layer_output = [layer.get_weights() for layer in model.layers if layer.name == "dense_1"]
#                                     print(layer_output[0])
#                                     for layer in model.layers:
#                                         print(layer.name)  
#                                     # 从模型中获取 normal_attention_weights
#                                     x1=np.random.randn(8,64)
#                                     x1[0:3,...]=0
#                                     x2=np.random.randn(8,64)
#                                     x2[4:7,...]=0
#                                     x3=np.concatenate((x1,x2),axis=-1)
#                                     x3_tensor = tf.convert_to_tensor(x3, dtype=tf.float32)
#                                     # is_x1_zero_tensor = tf.reduce_all(tf.equal(x1, 0), axis=-1)
#                                     is_x1_zero = np.all(x1 == 0, axis=1, keepdims=True)
#                                     is_x2_zero = np.all(x2 == 0, axis=1, keepdims=True)
#                                     print('is_x1_zero',is_x1_zero)

#                                     normal_attention_weights_layer = model.get_layer('dense')

#                                     normal_attention_weights = normal_attention_weights_layer(x3_tensor)
#                                     print('normal_attention_weights',normal_attention_weights)

#                                     # 计算 attention_weights
#                                     # attention_weights = tf.where(
#                                     #     is_x1_zero,
#                                     #     tf.zeros_like(normal_attention_weights),
#                                     #     normal_attention_weights
#                                     # )
#                                     attention_weights = tf.where(
#                                     is_x1_zero,  # 如果 is_x1_zero 为真
#                                     tf.zeros_like(normal_attention_weights),  # 则 attention_weights 为全零
#                                     tf.where(
#                                         is_x2_zero,  # 否则，如果 is_x2_zero 为真
#                                         tf.ones_like(normal_attention_weights),  # 则 attention_weights 为全一
#                                         normal_attention_weights  # 否则，使用 normal_attention_weights
#     )
# )
#                                     # 将 attention_weights 转换为 numpy 数组并打印值
#                                     attention_weights_values = attention_weights.numpy()
#                                     print("Attention Weights:", attention_weights_values) # 8,1
#                                     x=x1*attention_weights        #(8,64) * (8,1) 
#                                     print(x.shape)
                               
                                    














def parse_args():
    parser = argparse.ArgumentParser(description='Process command-line arguments.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    # parser.add_argument('--savevalxPath', type=str, default=r"D:\DLRicemap\transfer\sc_cq\2021_49\49sc_cqtrainareapointsimg2327_2324.npy", help='Path to save val x')
   
    parser.add_argument('--savemodeldir', type=str, default=r"D:\DLRicemap\run\logtransfer", help='Directory to save models')
    parser.add_argument('--modelname', type=str, default="cnn3d",help="Model to train. Available:\n" "cnn3d, " "cnn3dnocbrm," "cnn2d," "cnn1d,"  "RF,")
    # parser.add_argument('--modeltitle', type=str, default="",help="Model to train. Available:\n" "cnn3d, " "cnn3dnocbrm," "cnn2d," "cnn1d,"  "RF,")
    parser.add_argument('--ratio', type=int, default=1)
    return parser.parse_args()

    
    
from src.dataprocess.dataloader import dataprogressindependval,dataagument


if __name__ == '__main__':
    # Set the seed value all over the place to make this reproducible.
    tf.keras.utils.set_random_seed(999)
    tf.config.experimental.enable_op_determinism()

    import tensorflow as tf

    # 模拟两个分支的输出
    # output_branch1 = tf.constant([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])  # 部分样本全为0
    # output_branch2 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 所有样本有效

    # # 计算每个样本的绝对值以用于有效性判断
    # abs_output_branch1 = tf.abs(output_branch1)
    # print('abs_output_branch1',abs_output_branch1)
    # print(tf.reduce_sum(abs_output_branch1, axis=1, keepdims=True))

    # # 计算权重：每行的绝对值归一化
    # weights_branch1 = abs_output_branch1 / (tf.reduce_sum(abs_output_branch1, axis=1, keepdims=True) + 1e-6)
    # print('weights_branch1',weights_branch1)

    # # 加权融合
    # fused_output = weights_branch1 * output_branch1 + (1 - weights_branch1) * output_branch2

    # print(fused_output.numpy())  # 打印加权融合的输出
    # print(fused_output.numpy().shape)
    main()

    

    
                                   
                                  
                                    
                
  


    
    


    
  
                        
                    


