import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import nibabel as nib
import cv2 as cv
from keras import backend as K
import glob
import random as r
import math
from tqdm import tqdm
import nibabel as nib
from Unet2D import *
from Metrics import *
from Functions import *

Path = '/root/project/Brats/BraTs20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
items = os.listdir(Path)
Input_Data = []

for i in tqdm(items[:2], desc = 'Data Preprocessing ', unit = 'Patient'):
    brain_dir = os.path.normpath(Path+'/'+i)
    flair     = glob.glob(os.path.join(brain_dir, '*_flair*.nii'))
    t1        = glob.glob(os.path.join(brain_dir, '*_t1*.nii'))
    t1ce      = glob.glob(os.path.join(brain_dir, '*_t1ce*.nii'))
    t2        = glob.glob(os.path.join(brain_dir, '*_t2*.nii'))
    gt        = glob.glob( os.path.join(brain_dir, '*_seg*.nii'))
    modalities_dir = [flair[0], t1[0], t1ce[0], t2[0], gt[0]]
    P_Data = Data_Preprocessing(modalities_dir)
    Input_Data.append(P_Data)

for i in tqdm(range(1), desc = 'Data Concatenate : Lots of time ', unit = 'Input_Data'):
    InData = Data_Concatenate(Input_Data)
    

for i in tqdm(range(1), desc = 'Concatenate & to array' , unit= 'InData'):
    AIO = concatenate(InData, axis=3)
    AIO = np.array(AIO, dtype='float32')
    
for i in tqdm(range(1), desc = 'To Array', unit = 'Train_image'):
    TR=np.array(AIO[:,:,:,1],dtype='float32')

for i in tqdm(range(1), desc = 'To Array', unit = 'Train_label'):
    TRL=np.array(AIO[:,:,:,4],dtype='float32')
    
X_train , X_test, Y_train, Y_test = train_test_split(TR, TRL, test_size=0.15, random_state=32)

model = model(input_shape = (240,240,1))
model.summary()


# Compiling the model 
Adam=optimizers.Adam(lr=0.001)
model.compile(optimizer=Adam, loss='binary_crossentropy', metrics=['accuracy',dice_coef,precision,sensitivity,specificity])
# Fitting the model over the data
history = model.fit(X_train,Y_train,batch_size=32,epochs=40,validation_split=0.20,verbose=1,initial_epoch=0)


# Evaluating the model on the training and testing data 
model.evaluate(x=X_train, y=Y_train, batch_size=32 , verbose=1, sample_weight=None, steps=None)
model.evaluate(x=X_test, y=Y_test, batch_size=32, verbose=1, sample_weight=None, steps=None)

model.save('./BraTs2020_KG.h5')
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('./BraTs2020.history')