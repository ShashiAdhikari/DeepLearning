from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image


#To mount the google drive please login from the that gmail in which there is data for classification 
from google.colab import drive
drive.mount('/content/gdrive')

# dimensions of our images.
img_width, img_height = 150,150

train_data_dir = '/content/gdrive/My Drive/New folder/Train' # this is the path where we have put the image folder. In train there is 2 folder 
validation_data_dir = '/content/gdrive/My Drive/New folder/Test' # In validation folder there is mixed image of both the classes
nb_train_samples = 250
nb_validation_samples = 100
epochs = 50
batch_size = 20
verbose = 0

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
    
 # Model    
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
`
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
              
 model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5') # weight 



# to Test after training has been done 
img_pred = image.load_img('/content/gdrive/My Drive/New Folder/Test/00223.JPG', target_size= (150,150)) # pass the image name which you want to test
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)


rslt = model.predict(img_pred)
print(rslt)
if rslt[0][0] == 1:
  prediction = "positive"
else:
  prediction = "NotSp"

print(prediction)

