#On a quest to build the stupidest application ever devised.
#Using Tensorflow because fuck the internet. Slow as shit to initiate though.
import numpy as np
import cv2
import os, sys
import silence_tensorflow.auto
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import pandas as pd
import pyttsx3

def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	
	return vgg_face_descriptor



def get_race_data():
    model = loadVggFaceModel()
    base_model_output = Sequential()
    base_model_output = Convolution2D(6, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    race_model = Model(inputs=model.input, outputs=base_model_output)
    race_model.load_weights(os.path.join(sys.path[0],'weights','race_model_single_batch.h5'))

    races = ['Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Hispanic']
    
    faceCascade = cv2.CascadeClassifier(os.path.join(sys.path[0],'haarcascade_frontalface_default.xml'))
    cap = cv2.VideoCapture(0)
    cap.set(3,640) 
    cap.set(4,480) 

    while True:
        ret, img = cap.read()
        if (ret==True):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,     
                scaleFactor=1.2,
                minNeighbors=5,     
                minSize=(20, 20)
            )
            if(len(faces)==1):
                print('Capture complete.')
                cap.release()
                margin_rate = 30
                cv2.imwrite(os.path.join(sys.path[0], 'outputs', 'photo_raw_TF.jpg'), img)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    try:
                        margin_x = int(w * margin_rate / 100)
                        margin_y = int(h * margin_rate / 100)       
                        detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
                        detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
                    except Exception as err:
                        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                        detected_face = cv2.resize(detected_face, (224, 224))
                        print(err)
                    cv2.imwrite(os.path.join(sys.path[0], 'outputs', 'photo_marked_TF.jpg'), detected_face)

                    if detected_face.shape[0] > 0 and detected_face.shape[1] > 0 and detected_face.shape[2] >0: #sometimes shape becomes (264, 0, 3)
                        img_pixels = image.img_to_array(detected_face)
                        img_pixels = np.expand_dims(img_pixels, axis = 0)
                        img_pixels /= 255
                            
                        prediction_proba = race_model.predict(img_pixels)
                        prediction = np.argmax(prediction_proba)
                            
                        race = races[prediction]
                        break 
                break

            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        else:
            print('Waiting for video capture...')

    cv2.destroyAllWindows
    return([race, races, prediction_proba])

def sendCompliments(race):
    speech_engine = pyttsx3.init()

    if(race.lower()=='asian'):
        compliment = 'Rice is tasty, and hentai is the high pinnacle of modern art forms.'
    elif(race.lower()=='white'):
        compliment = 'Your acceptance of psychopaths and crackheads as leaders is wonderful.'
    else:
        compliment = 'I respect and admire your culture and heritage. Have a great day!'

    speech_engine.say(compliment)
    speech_engine.runAndWait()
    return(compliment)

try:
    os.mkdir(os.path.join(sys.path[0], 'outputs'))
except:
    pass

raceData = get_race_data()
race = raceData[0]
races = raceData[1]
raceProb = raceData[2]
compliment = sendCompliments(race)

for i in races:
    predictionString = i+': '+str(raceProb[0][races.index(i)])
    if i == race:
        predictionString = predictionString+'***'
    print(predictionString)
    
print(compliment)
