import tensorflow as tf
from tensorflow import keras
import pandas as pd
from datetime import datetime
import os
import random
import numpy as np
import yaml
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import inspect

from sklearn.metrics import cohen_kappa_score ,confusion_matrix,cohen_kappa_score,f1_score
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.model_selection import KFold
from keras import backend as K

from models import *
from plot import *
from loss import *
from randomdrop import randomdrop_features
from src.dataprocess.load_data import dataprogressindependval,dataagument


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(999)
   random.seed(999)
   np.random.seed(999)
   tf.random.set_seed(999)
reset_random_seeds()

def evaluate_model(model, xval, yval, maskval):
    """Evaluates the model on the validation set.

    Args:
        model: The trained model.
        xval: Validation features.
        yval: Validation labels.
        maskval: Validation mask.

    """
    total_correct = 0
    total_samples = 0
    logits_val = model.predict([xval,maskval],verbose=0)  
    valloss=weight_loss(yval, logits_val,maskval) 
    # 计算预测正确的样本数量
    valloss=tf.reduce_mean(valloss)
    predictions = tf.argmax(logits_val, axis=-1)
    # 计算 Cohen's Kappa
    kappa = cohen_kappa_score(tf.argmax(yval, axis=-1), predictions)
    # print('kappa', kappa)
    correct = tf.reduce_sum(tf.cast(tf.equal(predictions, tf.argmax(yval, axis=-1)), dtype=tf.float32))
    total_correct += correct
    total_samples += xval.shape[0]
    accuracy = total_correct / total_samples
    p,u=confusion_matrix1(tf.argmax(yval, axis=-1),predictions)

    return accuracy,valloss,p,u,kappa

def train_step(model, optimizer, x_batch_train, y_batch_train, maskloss,x_mask_train):
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
        logits_ed = model([x_batch_train,x_mask_train], training=True)
        if maskloss:
            # loss_ed=binary_weighted_cross_entropy(y_batch_train, logits_ed,x_mask_train) # 64,2
            # loss_ed=keras.losses.CategoricalCrossentropy(from_logits=False)(y_batch_train, logits_ed)
            loss_ed=weight_loss(y_batch_train, logits_ed,x_mask_train)
            # print('loss_ed',loss_ed.shape)
            # print('loss_ed',loss_ed)
           
        else:
            loss_ed=weight_loss(y_batch_train, logits_ed)
            # print('loss_ed',loss_ed.shape)
    grads = tape.gradient(loss_ed, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_ed,model


   

def train_and_evaluate(model, x, y,  maskaverage,maskloss,epochs, batch_size, savemodeldir,inputshape2d,modelname,dropout,cbrm,period,fusion):
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
            model = hycnn2d(inputshape=inputshape2d, drop=dropout, cbrm=cbrm, fusion=fusion, single1=period,maskmissing=maskaverage)
            # print(model.summary())
        elif modelname == 'cnn1d':
            model = hycnn1d(inputshape=inputshape2d, drop=dropout, fusion=0, single1=period,maskmissing=maskloss)
        else:
            raise ValueError("Invalid model name")

        lr = config['train']['lr_schedule']['initial_learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=weight_loss, metrics=['accuracy'])
        # 将loss function写入TXT文件
        loss_function_code = inspect.getsource(model.loss)
        txt_file_path =  os.path.join(savemodeldir,'loss_function.txt')
        with open(txt_file_path, 'w',encoding='utf-8') as file:
            file.write(loss_function_code)

        # Split data and mask into train/test sets for the current fold
        ytrain, yval = tf.gather(y, trainindex), tf.gather(y, testindex)
        xtrain,xval=x[trainindex],x[testindex]
        # ytrain,yval=y[trainindex],y[testindex]
        masktrain,maskval= tf.where(xtrain== 0, 0, 1), tf.where(xval== 0, 0, 1)
        masktrain,maskval= tf.cast(masktrain, dtype=tf.float32) ,tf.cast(maskval, dtype=tf.float32)  
      
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain,  masktrain, ytrain)).batch(batch_size)
        ## Loop over epochs
        for epoch in range(1, epochs + 1):
            # Train the model
            total_loss = 0
            batch_step = 0
           
            for x1, x2, y1 in train_dataset:
                if x1.shape[0]!=batch_size:
                    continue
                loss_ed,model= train_step(model, optimizer, x1, y1,maskloss, x2)
                total_loss += loss_ed
                batch_step += 1
            # print('batch_step: {batch_step}', batch_step)
            # print('loss_ed: {loss_ed}', loss_ed)
            # Evaluate the model
            val_accuracy,val_loss,p,u,kappa = evaluate_model(model, xval, yval, maskval)

            # Print training and validation results
            # print(f"Epoch {epoch}, Fold {fold_no}:")
            # print(f"   Train Loss: {total_loss / batch_step:.4f}")
            # print(f"   Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save the best model based on validation acc
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                model.save(os.path.join(savemodeldir, 'best_model.h5'), include_optimizer=True)

            # Update metric dataframe
            rf1,nrf1=2*p[1]*u[1]/(p[1]+u[1]),2*p[0]*u[0]/(p[0]+u[0])
            metricdf.at[epoch, f"{fold_no}trainloss"] = (total_loss / batch_step).numpy()
            metricdf.at[epoch, f"{fold_no}valloss"] = val_loss.numpy()
            metricdf.at[epoch, f"{fold_no}trainacc"] = val_accuracy.numpy()
            metricdf.at[epoch, f"{fold_no}valacc"] = val_accuracy.numpy()
            metricdf.at[epoch, f"{fold_no}kappa"] = kappa
            metricdf.at[epoch, f"{fold_no}rf1"] =rf1
            metricdf.at[epoch, f"{fold_no}nrf1"] = nrf1
            metricdf.at[epoch, f"{fold_no}F1"] = (rf1+nrf1)/2
            metricdf.at[epoch, f"{fold_no}rp"] = p[1]
            metricdf.at[epoch, f"{fold_no}nrp"] =p[0]
            metricdf.at[epoch, f"{fold_no}ru"] = u[1]
            metricdf.at[epoch, f"{fold_no}nru"] =u[0]
            metricdf.to_csv(os.path.join(savemodeldir, 'metrics.csv'), index=True)
        # print(f"Fold {fold_no} completed.")
    meanstdev_metrics(metricdf, savemodeldir)
    pmeanstd(metricdf, os.path.join(savemodeldir, 'metrics.png'))


def parse_args():
    parser = argparse.ArgumentParser(description='Process command-line arguments.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    return parser.parse_args()

def save_config(config, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def main():
    config_path=r'D:\DLRicemap\configindepend.yaml'
    with open(config_path, 'r') as file:
        global config
        config = yaml.safe_load(file)

    savelabelxPath = config['data']['trainx_path'] 
    savelabelyPath = config['data']['trainy_path']
    savemodeldir0 = config['train']['model_dir']
    os.makedirs(savemodeldir0,exist_ok=True)
    for model, model_params in config["models_params"].items(): 
        for dropout in model_params['params']['dropout']:
            for batch_size in model_params['params']['batch_size']:
                for epochs in model_params['params']['epochs']:
                    for modelname in model_params['params']['modelname']:
                        for fusion in model_params['params']['fusion']:
                            for cbrm in model_params['params']['cbrm']:
                                for maskaverage in model_params['params']['maskaverage']:
                                    for maskloss in model_params['params']['maskloss']:
                                        # if maskaverage==0 and maskloss==1 or (cbrm==0 and maskaverage==1 or maskloss==1):
                                        #     continue
                        
                                        model_name = '_'.join(['spatt'+str(cbrm),'maskaverage'+str(maskaverage), 'masklossmeanvaild'+str(0), 'drop'+str(dropout),'fusion'+str(fusion)]) # 
                                        savemodeldir = os.path.join(savemodeldir0,model_name)
                                        # if os.path.exists(savemodeldir):
                                        #     continue
                                        os.makedirs(savemodeldir,exist_ok=True)
                                        savemodelpath = os.path.join(savemodeldir,f"{model_name}.h5")
                                        save_path=os.path.join(savemodeldir,'config.yaml')
                                        config['train']['model_path']=savemodelpath
                                        save_config(config, config_path)
                                        save_config(config, save_path)
                                        x, y=dataprogressindependval(modelname,savelabelxPath, savelabelyPath,savemodeldir=savemodeldir,period=2)
                                        inputshape2d=(x.shape[1],x.shape[1],x.shape[-1])
                                        # x,y=randomdrop_features(x,y,ratio)

                                        # mask1=tf.where(x== 0, 0, 1)
                                        # mask1=tf.cast(mask1,tf.float32)
                                        # valid_pixels = tf.reduce_sum(mask1, axis=[0,1, 2, 3]) 
                                        # # sample*heights*widths*channels
                                        # print(tf.cast(tf.size(mask1),tf.float32))
                                        # valid_ratio = valid_pixels / tf.cast(tf.size(mask1), tf.float32)
                                        # print('valid_ratio',valid_ratio)
                                        # print('----------------------------------')   
                                        train_and_evaluate(model, x, y, maskaverage,maskloss,epochs, batch_size, savemodeldir,inputshape2d,modelname,dropout,cbrm,period=2,fusion=fusion)
                                        

                                
                                        

if __name__ == '__main__':
    # Set the seed value all over the place to make this reproducible.
    tf.keras.utils.set_random_seed(999)
    tf.config.experimental.enable_op_determinism()
    main()

    

    
                                   
                                  
                                    
                
  


    
    


    
  
                        
                    


