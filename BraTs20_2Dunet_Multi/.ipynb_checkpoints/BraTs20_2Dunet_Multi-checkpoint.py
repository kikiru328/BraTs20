from tqdm import tqdm
import os
import time
import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage import exposure
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.models import Model, load_model
from Metrics import *
from Unet2D_multi import *
from Functions import *

train_set_path = '/root/project/Brats/BraTs20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
patient_IDs = next(os.walk(train_set_path, topdown=True))[1][:350]
val_IDs = next(os.walk(train_set_path, topdown=True))[1][350:370]


K.clear_session()

model = unet()
model.summary()

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, spatient_IDs, dim=(128,128), batch_size = 1, n_channels = 2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = spatient_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*100, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*100, 240, 240))
        Y = np.zeros((self.batch_size*100, *self.dim, 4))


        
        # Generate data
        # for i, ID in enumerate(Batch_ids):
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(train_set_path, i)

            vol_path = os.path.join(case_path, f'{i}_flair.nii');
            flair =read_image(vol_path)      

            vol_path = os.path.join(case_path, f'{i}_t1ce.nii');
            ce =read_image(vol_path)
            
            vol_path = os.path.join(case_path, f'{i}_seg.nii');
            seg =read_image(vol_path)
        
            for j in range(100):
                 X[j +100*c,:,:,0] = cv.resize(flair[:,:,j+22], (128, 128));
                 X[j +100*c,:,:,1] = cv.resize(ce[:,:,j+22], (128, 128));

                 y[j +100*c] = seg[:,:,j+22];
                    
            #=============Preprocess masks===========
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (128, 128));
        return X/np.max(X), Y
        

training_generator = DataGenerator(patient_IDs)
valid_generator = DataGenerator(val_IDs)


callbacks = [
            # ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
            ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor="loss", patience=3, verbose=1)
            ]

model.compile(optimizer = "adam", 
                    loss = 'categorical_crossentropy', 
                    metrics = ["accuracy", dice_coef]
                    ) 

history = model.fit_generator(training_generator,
                              epochs=100,
                              steps_per_epoch=len(patient_IDs),  
                              validation_data = valid_generator,
                              callbacks= callbacks,
                              )  

model.save('./BraTs20_Multi.h5')
hist_df = pd.DataFrame(history.history)
hist_df.to_csv('./BraTs20_Multi_history')