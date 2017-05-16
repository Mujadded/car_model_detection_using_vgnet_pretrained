
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
K.set_image_dim_ordering('tf')

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


# input image dimensions
img_rows, img_cols = 50, 50

# number of channels
img_channels = 3

train_labels=scipy.io.loadmat('/home/mujadded/Practice/Thesis/with_keras/cars_annos.mat')
annotations = np.array(train_labels['annotations'])[0]
#%%
label=[]
box_x1=[]
box_x2=[]
box_y1=[]
box_y2=[]

for x in annotations:

    label.append(x['class'][0][0])
    box_x1.append(x['bbox_x1'][0][0])
    box_y1.append(x['bbox_y1'][0][0])
    box_x2.append(x['bbox_x2'][0][0])
    box_y2.append(x['bbox_y2'][0][0])


label=np.array(label)






#%%
#  data

path1 = '/home/mujadded/Practice/Thesis/with_keras/car_img'    #path of folder of images
path2 = '/home/mujadded/Practice/Thesis/with_keras/car_img_cropped'  #path of folder to save images
#%%
# DONT RUN IF ALREDY IMAGES ARE CROPPED

listing = os.listdir(path1)
num_samples=size(listing)
print num_samples
img_num=0
listing=sorted(listing)
for file in listing:
    im = Image.open(path1 + '/' + file)
    img = im.crop((box_x1[img_num], box_y1[img_num], box_x2[img_num], box_y2[img_num]))
    img = img.resize((img_rows,img_cols))
    gray = img.convert('RGB')
               #need to do some more processing here
    gray.save(path2 +'/' +  file, "JPEG")
    img_num+=1
#     #img.save(path2 +'/' +  file, "JPEG")

#%%
imlist = os.listdir(path2)
imlist=sorted(imlist)
im1 = array(Image.open(path2+"/"+imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('/home/mujadded/Practice/Thesis/with_keras/car_img_cropped'+ '/' + im2)).flatten()
              for im2 in imlist],'f')

#label1=np.ones((num_samples,),dtype = int)
#label[0:89]=0

#label[89:187]=1
#label[187:]=2

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[167].reshape(img_rows,img_cols,3)
plt.imshow(img)
#plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)


#%%
batch_size = 32
num_classes = 197
epochs = 200
data_augmentation = False

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols,img_channels)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols,img_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

num_train_samples=x_train.shape[0]
num_validation_samples=x_test.shape[0]


#%%

#load vgg16 without dense layer and with theano dim ordering
base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (img_rows, img_cols,img_channels))



x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

#create graph of your new model
head_model = Model(input = base_model.input, output = predictions)

#compile the model
head_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

head_model.summary()




#%%
if not data_augmentation:
    print('Not using data augmentation.')
    hist = head_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                   verbose=1, validation_data=(x_test, y_test))
    
    head_model.save_weights('/home/mujadded/Practice/Thesis/with_keras/using_pretrained/bottleneck_fc_model.h5')
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    hist=model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
#%%

# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])




#%%

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(x_test[1:5]))
print(y_test[1:5])
# -*- coding: utf-8 -*-

