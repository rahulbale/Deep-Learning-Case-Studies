# Importing the necessary libraries
import cv2
from keras.models import load_model
import pandas as pd
import numpy as np

### Parameters ###
labels = 'labels/labels.csv'
model = 'Callbacks/model.22_0.83.hdf5'
threshold = 0.75   


# Loading the Trained model
model = load_model(model)
model.input_shape

# Getting input model shapes for inference
model_target_size = model.input_shape[1:3]
model_target_size

### Preprocessing the Images 
def preprocessing(img):
    img = cv2.equalizeHist(img)
    img = img/255.0
    return img

### Function for displaying predicted labels
def datalabel(labels,getindex):
    df=pd.read_csv(labels)
    df.set_index('ClassId', inplace=True)
    value = df.loc[getindex, 'Name']
    value = pd.Series(value)
    value_ = list(value.values)
    value_ = str(value_)
    return value_



### Video Capturing
cap = cv2.VideoCapture(0)


while True:
    
    ret, frame = cap.read()   # Reading Image
    grayimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Converting BGR Image into Gray

    
    try:
        img = np.asarray(grayimg)   #Converting Image into Array
        img = cv2.resize(img, (model_target_size))
    except:
        continue
    
    # Processing Image
    img = preprocessing(img)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    
    #Predicting Image
    prediction = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue =np.amax(prediction)
    
    if probabilityValue > threshold:   
        text = "Class : "+ str(classIndex)+" "+str(datalabel(labels,classIndex))
        cv2.putText(frame,text, (50, 35), font, 0.75, (0, 225, 0), 2, cv2.LINE_AA)
        cv2.putText(frame,"PROBABILITY: "+ str(round(probabilityValue*100,2))+"%", (50, 75),  
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 225, 0), 2, cv2.LINE_AA)
    
    
    cv2.imshow("Result", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()
    