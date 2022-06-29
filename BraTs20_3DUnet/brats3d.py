import os
import numpy as np
from custom_datagen import imageLoader
#import tensorflow as tfs
import keras
from matplotlib import pyplot as plt
import glob
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import tensorflow


train_img_dir = "/root/project/Brats/BraTs20/BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "/root/project/Brats/BraTs20/BraTS2020_TrainingData/input_data_128/train/masks/"

val_img_dir = "/root/project/Brats/BraTs20/BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "/root/project/Brats/BraTs20/BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list = sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list = sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))


batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)


img, msk = train_img_datagen.__next__()

img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)


wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


from  unet3d import model

model = model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)

history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

model.save('./brats_3d.hdf5')
hist_df = pd.DataFrame(history.history)
hist_csv_file = './history.csv'
with open(hist_csv_file, mode = 'w') as f:
    hist_df.to_csv(f)
