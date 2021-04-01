# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
import pickle
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping



### Parameters ###

path = "myData" 
labelFile = 'labels\labels.csv'
batch_size = 32 
epoch = 110
imageDimesions = (32,32,3)
testRatio = 0.2    
validationRatio = 0.2
patience = 20


### Importing the images into images and is count into classNo
count = 0
images = []
classNo = []

myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
for i in range(0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for pic in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+pic)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)  


### Spliting data into train test 

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio, random_state=42)

print("Data Shapes")
print("Train",X_train.shape,y_train.shape)
print("Test",X_test.shape,y_test.shape)

### Preprocessing the Images  

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    # CONVERT TO GRAYSCALE
    img = cv2.equalizeHist(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img
 
X_train = np.array(list(map(preprocessing,X_train)))  
X_test = np.array(list(map(preprocessing,X_test)))

### Adding a depth of 1

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

### Data (Image) Argumnetation for more Generic

data_generator = ImageDataGenerator(rotation_range=10,
                                    rescale=1/255,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=.1,
                                    horizontal_flip=True
                                    )
data_generator.fit(X_train)

#### Converting the dependent variable to categorical data (if yes then 1 else 0) 

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)



### Building Conventional Neural Networks Model

regularization = l2(0.001)
model= Sequential()
model.add(Conv2D(32,(5,5), activation='relu',kernel_regularizer=regularization, input_shape=(imageDimesions[0],imageDimesions[1],1)))  
model.add(Conv2D(32, (5,5), activation='relu', kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3)))

model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularization))
model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularization))
model.add(BatchNormalization())
model.add(MaxPooling2D((3, 3)))
model.add(Dropout(0.10)) 


model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.10)) 
model.add(Dense(43,activation='softmax')) 
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


# Using Callbacks for capturing the best accuracy and saving the model

log_file_path = "Callbacks/model.log"
csv_logger = CSVLogger(log_file_path, append=False)

early_stop = EarlyStopping('val_loss', patience=patience)

trained_models_path = 'Callbacks/model'
model_names = trained_models_path+'.{epoch:02d}_{val_accuracy:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                save_best_only=True)

callbacks = [model_checkpoint, csv_logger, early_stop]

### Training the model

history = model.fit(data_generator.flow(X_train,y_train,batch_size=batch_size), 
                    steps_per_epoch = int(len(X_train)// batch_size),
                    epochs = epoch, verbose=True, callbacks = callbacks,
                    validation_data = (X_val,y_val)
                    )


### Plotting the Accuracy and Validation Accuaracy 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

### Plotting the Loss and Validation Loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



