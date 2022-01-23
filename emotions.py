import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, os.path
import sys

def display(model):
    model.load_weights(model_name)

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary for emotions and emojis and their corresponding number
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    emoji_dict = {0: "angry.png", 1: "disgusted.png", 2: "fearful.png", 3: "happy.png", 4: "neutral.png", 5: "sad.png",
                  6: "surprised.png"}

    # start video
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw frame for face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # find face and make prediction for emotion
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            # add text and emoji for predicted emotion
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
            emoji = cv2.imread(emoji_dict[maxindex])
            emoji = cv2.resize(emoji, (h, w))
            frame[y:y + h, x:x + w] = cv2.addWeighted(frame[y:y + h, x:x + w], 0.4, emoji, 0.6, 0)
        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model(model, train_generator, num_train, batch_size, num_epoch, validation_generator, num_val):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    model.save_weights(model_name)
    print("training complete")


def createModel():
    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",help="train/display")
    ap.add_argument("--batch_size", default=64, type=int, help="size of batch")
    ap.add_argument("--num_epoch", default=50, type=int, help="number of epochs")
    ap.add_argument("--train_dir", default = "data/train", type=str, help="directory for training")
    ap.add_argument("--val_dir", default="data/test", type=str, help="directory for testing")
    ap.add_argument("--model_name", default="face-to-emotion.h5", type=str, help="name for model")

    mode = ap.parse_args().mode
    batch_size = ap.parse_args().batch_size
    num_epoch = ap.parse_args().num_epoch
    train_dir = ap.parse_args().train_dir
    val_dir = ap.parse_args().val_dir
    model_name = ap.parse_args().model_name
    num_train = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_val = sum([len(files) for r, d, files in os.walk(val_dir)])

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    # create and train the model
    model = createModel()
    if mode == "train":
        train_model(model, train_generator, num_train, batch_size, num_epoch, validation_generator, num_val)

    # display emotion and emoji on faces
    elif mode == "display":
        display(model)
