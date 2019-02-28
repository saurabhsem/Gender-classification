#Importing the libraries and functions
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,ZeroPadding2D,Flatten,Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix,f1_score
import random as rn
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from keras.preprocessing import image


#................................Architecture-Algorithms-1-1.......................................#

# Initializing the CNN

classmodel=Sequential()
classmodel.add(Convolution2D(64,3,3,input_shape=(32,32,3),activation='relu'))
classmodel.add(MaxPooling2D(pool_size=(2,2)))
classmodel.add(ZeroPadding2D(1,))
classmodel.add(Convolution2D(64,3,3,activation='relu'))
classmodel.add(Convolution2D(64,3,3,activation='relu'))
classmodel.add(MaxPooling2D(pool_size=(2,2)))
classmodel.add(Dropout(0.25))
classmodel.add(Flatten())

# Adding Dense hidden layers
classmodel.add(Dense(output_dim=64,activation='relu'))
classmodel.add(Dropout(0.25))
classmodel.add(Dense(output_dim=64,activation='relu'))
classmodel.add(Dense(output_dim=64,activation='relu'))
classmodel.add(Dense(output_dim=64,activation='relu'))
classmodel.add(Dense(output_dim=64,activation='relu'))
classmodel.add(Dense(output_dim=64,activation='relu'))
classmodel.add(Dropout(0.25))

#Full connection
classmodel.add(Dense(output_dim=1,activation='sigmoid'))

#Intitializing the optimizer and tuning parameters of it
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#Compiling the network
classmodel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Data Augmentation
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2
                                 ,horizontal_flip=True)


test_datagen=ImageDataGenerator(rescale=1./255)


batch_size_train_arch1=32
batch_size_valid_arch1=32
train_sample_size_arch1=1272 
valid_sample_size_arch1=301

#Fitting the images to the network
training_set_arch1=train_datagen.flow_from_directory('D:\\NUS\\Semester1\CI\\Neural Network\\Assignments_NN\\datasets\\train',
                    target_size=(32,32),batch_size=batch_size_train_arch1,class_mode='binary')

valid_set_arch1=test_datagen.flow_from_directory('D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\validation',
                    target_size=(32,32),batch_size=batch_size_train_arch1,class_mode='binary')


model_learning_arch1=classmodel.fit_generator(training_set_arch1,steps_per_epoch=train_sample_size_arch1//batch_size_train_arch1,nb_epoch=25,
 
                         
                   validation_data=valid_set_arch1,validation_steps=valid_sample_size_arch1//batch_size_valid_arch1)

classmodel.summary()#checking the model summary


test_set=test_datagen.flow_from_directory('D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\test',
                    target_size=(32,32),batch_size=1,class_mode='binary')
# summarize history for accuracy
plt.plot(model_learning_arch1.history['acc'])
plt.plot(model_learning_arch1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_learning_arch1.history['loss'])
plt.plot(model_learning_arch1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Evaluation on the test data
test_files_names= test_set.filenames
os.getcwd()
os.chdir("D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\test")


training_set_arch1.class_indices
predicted_labels_arch1=[]
predicted_category_arch1=[]
for i in range(len(test_files_names)):
    
 test_image = image.load_img(test_files_names[i], target_size = (32, 32))
 test_image = image.img_to_array(test_image)
 test_image = np.expand_dims(test_image, axis = 0)
 
 result_arch1 = classmodel.predict(test_image)
 predicted_labels_arch1.append(result_arch1[0][0]) 
 
 if result_arch1[0][0] == 1:
    prediction = 'woman'
 else:
    prediction = 'man'
 predicted_category_arch1.append(prediction)   
 
 
actual_category_arch1=[]
for i in range(len(test_files_names)):
    actual_category_arch1.append([test_files_names[i].split("\\")][0][0])
    
confusion_matrix(actual_category_arch1,predicted_category_arch1)
tn1, fp1, fn1, tp1 = confusion_matrix(actual_category_arch1,predicted_category_arch1).ravel()
test_sample_size=315
accuracy_arch1=(tp1+tn1)/test_sample_size # Not a evaluting parameter in our case
accuracy_arch1

sensitivity_arch1=(tp1/(tp1+fn1))*100 # how good it is predicting a women as a women
sensitivity_arch1#Recall

precision_arch1=(tp1/(tp1+fp1))*100
precision_arch1

f1_acore_arch1=f1_score(actual_category_arch1,predicted_category_arch1, average='macro')
f1_acore_arch1

specificity_arch2=(tn1/(tn1+fp1))*100  # Not - important in our case - how good it is predicting a man as a man
specificity_arch2


#............................................Architecture-Algorithms-2 begins................................................................................#

# Initializing the CNN

classmodel2=Sequential()



classmodel2.add(Convolution2D(64,3,3,input_shape=(32,32,3),activation='relu'))
classmodel2.add(MaxPooling2D(pool_size=(2,2)))
classmodel2.add(ZeroPadding2D(1,))
classmodel2.add(Convolution2D(64,3,3,activation='relu'))
classmodel2.add(ZeroPadding2D(1,))
classmodel2.add(MaxPooling2D(pool_size=(2,2)))
classmodel2.add(Convolution2D(64,3,3,activation='relu'))
classmodel2.add(MaxPooling2D(pool_size=(2,2)))
classmodel2.add(ZeroPadding2D(1,))
classmodel2.add(Dropout(0.25))
classmodel2.add(Convolution2D(64,3,3,activation='relu'))
classmodel2.add(ZeroPadding2D(1,))
classmodel2.add(MaxPooling2D(pool_size=(2,2)))
classmodel2.add(Dropout(0.25))


#Step-3-Flattening

classmodel2.add(Flatten())

# Adding Dense hidden layers

classmodel2.add(Dense(output_dim=64,activation='relu'))
classmodel2.add(Dropout(0.25))
classmodel2.add(Dense(output_dim=64,activation='relu'))
classmodel2.add(Dense(output_dim=64,activation='relu'))
classmodel2.add(Dense(output_dim=64,activation='relu'))
classmodel2.add(Dropout(0.5))
classmodel2.add(Dense(output_dim=64,activation='relu'))
classmodel2.add(Dense(output_dim=64,activation='relu'))
classmodel2.add(Dense(output_dim=64,activation='relu'))

classmodel2.add(Dropout(0.25))


#Full Connection

classmodel2.add(Dense(output_dim=1,activation='sigmoid'))

#Initializing the optimizer and tuning paarmeters to it
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#Compile the network

classmodel2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting the network to the images
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2
                                 ,horizontal_flip=True)


test_datagen=ImageDataGenerator(rescale=1./255)


batch_size_train_arch2=32
batch_size_valid_arch2=32
train_sample_size_arch2=1272 
valid_sample_size_arch2=301


training_set_arch2=train_datagen.flow_from_directory('D:\\NUS\\Semester1\CI\\Neural Network\\Assignments_NN\\datasets\\train',
                    target_size=(32,32),batch_size=batch_size_train_arch2,class_mode='binary')

valid_set_arch2=test_datagen.flow_from_directory('D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\validation',
                    target_size=(32,32),batch_size=batch_size_train_arch2,class_mode='binary')



model_learning_arch2=classmodel2.fit_generator(training_set_arch2,steps_per_epoch=train_sample_size_arch2//batch_size_train_arch2,nb_epoch=35,
 
                         
                   validation_data=valid_set_arch2,validation_steps=valid_sample_size_arch2//batch_size_valid_arch2)


classmodel2.summary()

# summarize history for accracy
plt.plot(model_learning_arch2.history['acc'])
plt.plot(model_learning_arch2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_learning_arch2.history['loss'])
plt.plot(model_learning_arch2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#.................................Predicting the test data..........................................#

test_set=test_datagen.flow_from_directory('D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\test',
                    target_size=(32,32),batch_size=1,class_mode='binary')
test_files_names= test_set.filenames
os.getcwd()
os.chdir("D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\test")


training_set_arch2.class_indices
predicted_labels_arch2=[]
predicted_category_arch2=[]
for i in range(len(test_files_names)):
    
 test_image = image.load_img(test_files_names[i], target_size = (32, 32))
 test_image = image.img_to_array(test_image)
 test_image = np.expand_dims(test_image, axis = 0)
 
 result_arch2 = classmodel2.predict(test_image)
 predicted_labels_arch2.append(result_arch2[0][0]) 
 
 if result_arch2[0][0] == 1:
    prediction = 'woman'
 else:
    prediction = 'man'
 predicted_category_arch2.append(prediction)   
 
 
actual_category_arch2=[]
for i in range(len(test_files_names)):
    actual_category_arch2.append([test_files_names[i].split("\\")][0][0])
    
confusion_matrix(actual_category_arch2,predicted_category_arch2)
tn2, fp2, fn2, tp2 = confusion_matrix(actual_category_arch2,predicted_category_arch2).ravel()
test_sample_size=315
accuracy_arch2=(tp2+tn2)/test_sample_size # Not a evaluting parameter in our case
accuracy_arch2

sensitivity_arch2=(tp2/(tp2+fn2))*100 # how good it is predicting a women as a women
sensitivity_arch2#Recall

precision_arch2=(tp2/(tp2+fp2))*100
precision_arch2

f1_score_arch2=f1_score(actual_category_arch2,predicted_category_arch2, average='macro') 
f1_score_arch2

specificity_arch2=(tn2/(tn2+fp2))*100  # Not - important in our case - how good it is predicting a man as a man
specificity_arch2



#..........................................architecture1-algorithms2-3.................................................#

# Initializing the CNN

classmodel3=Sequential()

#Adding convolution layers

classmodel3.add(Convolution2D(64,3,3,input_shape=(32,32,3),activation='relu'))
classmodel3.add(MaxPooling2D(pool_size=(2,2)))
classmodel3.add(ZeroPadding2D(1,))
classmodel3.add(Convolution2D(64,3,3,activation='relu'))
classmodel3.add(ZeroPadding2D(1,))
classmodel3.add(Convolution2D(64,3,3,activation='relu'))
classmodel3.add(MaxPooling2D(pool_size=(2,2)))
classmodel3.add(Dropout(0.25))


#Step-3-Flattening

classmodel3.add(Flatten())

#Adding the dense layers

classmodel3.add(Dense(output_dim=64,activation='relu'))
classmodel3.add(Dropout(0.5))
classmodel3.add(Dense(output_dim=64,activation='relu'))
classmodel3.add(Dense(output_dim=64,activation='relu'))
classmodel3.add(Dropout(0.5))
classmodel3.add(Dense(output_dim=64,activation='relu'))
classmodel3.add(Dropout(0.5))
classmodel3.add(Dense(output_dim=64,activation='relu'))
classmodel3.add(Dense(output_dim=64,activation='relu'))
classmodel3.add(Dropout(0.5))



#Step-4-Full Connection.................................#

classmodel3.add(Dense(output_dim=1,activation='sigmoid'))

#Step-5-Compile the CNN

classmodel3.compile(optimizer='RMSprop',loss='mean_squared_error',metrics=['accuracy'])

# Part2

#Step-1-Fitting the network to the images
train_datagen3=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2
                                 ,horizontal_flip=True)


test_datagen3=ImageDataGenerator(rescale=1./255)


batch_size_train_arch3=32
batch_size_valid_arch3=32
train_sample_size_arch3=1272 
valid_sample_size_arch3=301


training_set_arch3=train_datagen.flow_from_directory('D:\\NUS\\Semester1\CI\\Neural Network\\Assignments_NN\\datasets\\train',
                    target_size=(32,32),batch_size=batch_size_train_arch3,class_mode='binary')

valid_set_arch3=test_datagen.flow_from_directory('D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\validation',
                    target_size=(32,32),batch_size=batch_size_train_arch3,class_mode='binary')


model_learning_arch3=classmodel3.fit_generator(training_set_arch3,steps_per_epoch=train_sample_size_arch3//batch_size_train_arch3,nb_epoch=30,
 
                         
                   validation_data=valid_set_arch3,validation_steps=valid_sample_size_arch3//batch_size_valid_arch3)


classmodel3.summary() # checking the model summary


# summarize history for accuracy
plt.plot(model_learning_arch3.history['acc'])
plt.plot(model_learning_arch3.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_learning_arch3.history['loss'])
plt.plot(model_learning_arch3.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#............................predicting on the test data................................#
test_set=test_datagen.flow_from_directory('D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\test',
                    target_size=(32,32),batch_size=1,class_mode='binary')
test_files_names= test_set.filenames
os.getcwd()
os.chdir("D:\\NUS\\Semester1\\CI\\Neural Network\\Assignments_NN\\datasets\\test")


training_set_arch3.class_indices
predicted_labels_arch3=[]
predicted_category_arch3=[]
for i in range(len(test_files_names)):
    
 test_image = image.load_img(test_files_names[i], target_size = (32, 32))
 test_image = image.img_to_array(test_image)
 test_image = np.expand_dims(test_image, axis = 0)
 
 result_arch3= classmodel3.predict(test_image)
 predicted_labels_arch3.append(result_arch3[0][0]) 
 
 if result_arch3[0][0] == 1:
    prediction = 'woman'
 else:
    prediction = 'man'
 predicted_category_arch3.append(prediction)   
 
 
actual_category_arch3=[]
for i in range(len(test_files_names)):
    actual_category_arch3.append([test_files_names[i].split("\\")][0][0])
    
confusion_matrix(actual_category_arch3,predicted_category_arch3)
tn3, fp3, fn3, tp3 = confusion_matrix(actual_category_arch3,predicted_category_arch3).ravel()
test_sample_size=315

accuracy_arch3=(tp3+tn3)/test_sample_size # Not a evaluting parameter in our case
accuracy_arch3

sensitivity_arch3=(tp3/(tp3+fn3))*100 # how good it is predicting a women as a women
sensitivity_arch3#Recall

precision_arch3=(tp3/(tp3+fp3))*100
precision_arch3

f1_acore_arch3=f1_score(actual_category_arch3,predicted_category_arch3, average='macro')
f1_acore_arch3

specificity_arch3=(tn3/(tn3+fp3))*100  # how good it is predicting a man as a man
specificity_arch3


#..........................................Ensembling.................................................#

        
#........................Building the voting system where majority wins...................#
predicted_category_ensemble=[]
for i in range(len(predicted_category_arch3)):
    if ((predicted_category_arch3[i]=="woman" and  predicted_category_arch2[i]=="woman") or ( predicted_category_arch3[i]=="woman" and  predicted_category_arch1[i]=="woman") or (predicted_category_arch1[i]=="woman" and predicted_category_arch2[i]=="woman")):
        
        predicted_category_ensemble.append("woman")
        
    else:
        
        predicted_category_ensemble.append("man")
        
set(predicted_category_ensemble)
len(predicted_category_ensemble)
    
actual_category=actual_category_arch1
#..................................Evaluting on the test data.........................#

   
confusion_matrix(actual_category,predicted_category_ensemble)
tn_e, fp_e, fn_e, tp_e = confusion_matrix(actual_category,predicted_category_ensemble).ravel()
test_sample_size=315
accuracy_ensemble=(tp_e+tn_e)/test_sample_size # Not a evaluting parameter in our case
accuracy_ensemble

sensitivity_ensemble=(tp_e/(tp_e+fn_e))*100 # how good it is predicting a women as a women
sensitivity_ensemble#Recall

precision_ensemble=(tp_e/(tp_e+fp_e))*100
precision_ensemble


f1_acore_ensemble=f1_score(actual_category,predicted_category_ensemble, average='macro')
f1_acore_ensemble


specificity_ensemble=(tn_e/(tn_e+fp_e))*100  # not important in our case how good it is predicting a man as a man
specificity_ensemble


#.......................................CODE ENDS...............................................................#
