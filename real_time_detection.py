import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAvgPool2D
from tensorflow.keras.models import Model


def create_model(input_shape):
    base_model = ResNet50(input_shape=input_shape,
                          include_top=False,
                          # include_top=False means that the classification layer is no longer present
                          weights='imagenet')

    # declaring the input
    inputs = Input(shape=input_shape)

    # obtaining image embeddings from the second last layer
    x = base_model(inputs, training=False)

    # adding global avg pooling
    x = GlobalAvgPool2D()(x)

    # including dropout with prob=0.2
    x = Dropout(rate=0.2)(x)

    # adding a flatten layer
    x = Flatten()(x)

    # adding the prediction layer with 7 units, softmax activation
    prediction_layer = Dense(units=7, activation='softmax')(x)

    # outputs
    outputs = prediction_layer

    model = Model(inputs, outputs)

    return model


# -------------------------------------------------------------
# REAL-TIME VIDEO FACIAL EMOTION RECOGNITION
# -------------------------------------------------------------

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model = create_model(input_shape=(48, 48, 3))
model.load_weights('ferNet.h5')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cropped_img = np.repeat(cropped_img, 3, -1)
        plt.imshow(cropped_img[0])
        prediction = model.predict(cropped_img)
        cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                    1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
