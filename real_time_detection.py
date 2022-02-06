import pandas as pd
import cv2
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.optimizers import adam_v2
import matplotlib.pyplot as plt
#-------------------------------------------------------------
# HYPERPARAMETER DECLARATIONS
#-------------------------------------------------------------
classes = 7
width, height = 48,48
input_shape = (48,48,1)
batch_size = 128
epochs = 15
num_features = 64
#-------------------------------------------------------------
#-------------------------------------------------------------
#KERAS MODEL 
#-------------------------------------------------------------
model = Sequential()

# input layer
model.add(Conv2D(num_features/2, kernel_size=(3, 3), activation='relu', input_shape=(width,height,1)))
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 1st hidden layer
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd hidden layer
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd hidden layer
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flatten layer
model.add(Flatten())
model.add(Dense(2*2*num_features, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# output layer
model.add(Dense(classes, activation='softmax'))

model.summary()
#-------------------------------------------------------------
model.load_weights('model_trained.h5')

#-------------------------------------------------------------
#REAL-TIME VIDEO FACIAL EMOTION RECOGNITION
#-------------------------------------------------------------
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# model = load_model(MODELPATH)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#-------------------------------------------------------------