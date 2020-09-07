#importing packages
import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras.applications.inception_v3 import InceptionV3 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

#training data directory and parameteres
trainDataGen = ImageDataGenerator(
                    rotation_range = 5,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    rescale = 1.0/255,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = False,
                    fill_mode = 'nearest',
                )

testDataGen = ImageDataGenerator(rescale=1.0/255,)


train_gen = trainDataGen.flow_from_directory(
                    "data/Train",
                    target_size=(75, 75),
                    class_mode="categorical",
                    batch_size=64
                    )


#validation data directory and parameteres
val_gen = testDataGen.flow_from_directory(
                    "data/Test",
                    target_size=(75, 75),
                    class_mode="categorical",
                    batch_size=64
                    )


#building base model
base_model = InceptionV3(input_shape = (75, 75, 3),  
                                include_top = False,  
                                weights = 'imagenet') 
for layer in base_model.layers: 
  layer.trainable = False
  
#stop training is model accuracy reached 99% 
class myCallback(tf.keras.callbacks.Callback): 
  def on_epoch_end(self, epoch, logs={}): 
    if(logs.get('acc')>0.99): 
      self.model.stop_training = True
      
# code 
x = layers.Flatten()(base_model.output) 

x = layers.BatchNormalization()(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512,activation='relu')(x) 
#x = layers.Dropout(0.1)(x)  #dropout is decreased after observation of resutls on dataset
x = layers.BatchNormalization()(x)

x = layers.Dense(108, activation='softmax')(x)            
  
model = Model( base_model.input, x)  


#compiling the model
opt = tf.keras.optimizers.Adam()
model.compile(optimizer = opt,loss = 'categorical_crossentropy',metrics = ['acc']) # was using binary before

#if no change in validation loss - learning rate will be decreased
anne = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

#fitting the model  
history = model.fit( 
            train_gen, 
            validation_data = val_gen, 
            steps_per_epoch = 3167, 
            epochs = 50, 
            validation_steps = 700) 