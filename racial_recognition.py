#On a quest to build the stupidest application ever devised.
#Uploading your image to some janky API (betaface)
import numpy as np
import cv2
import os, sys
import requests
import json
import base64
import pandas as pd
import pyttsx3

def get_face_image():
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
                cv2.imwrite(os.path.join(sys.path[0], 'outputs', 'photo_raw.jpg'),img)
                with open(os.path.join(sys.path[0], 'outputs', 'photo_raw.jpg'), "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(sys.path[0], 'outputs', 'photo_marked.jpg'),img)
                    break
                break
                
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        else:
            print('Waiting for video capture...')

    print('Image captured. Closing...')
    cap.release()
    cv2.destroyAllWindows
    print('Capture closed.')
    return encoded_string

def send_image_to_API(encoded_string):
    apiShit = {
                "api_key": "d45fd466-51e2-4701-8da8-04351c872236",
                "file_base64": encoded_string,
                "detection_flags": "classifiers,content, extended",
                #"recognize_targets": ["all@mynamespace"]
                }

    jsonData = json.dumps(apiShit)
    newHeaders = {'Content-type': 'application/json', 'Accept': 'application/json'}

    print('Sending API request...')
    try:
        response = requests.post('https://www.betafaceapi.com/api/v2/media', headers = newHeaders, data=jsonData)
        print("Status code: ", response.status_code)
    except:
        print("Unable to send API request")

    with open(os.path.join(sys.path[0], 'outputs', 'facial_data.json'), 'w') as f:
        json.dump(response.json(), f)

    return response.json() 

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

capturedImage = get_face_image()
responseJson = send_image_to_API(capturedImage)

dataTags = pd.DataFrame(responseJson['media']['faces'][0]['tags'])
dataTags.to_csv(os.path.join(sys.path[0], 'outputs', 'facial_data_tags.csv'), index=None)

#dataRecognize = pd.DataFrame(responseJson['recognize']['results'][0]['matches'])
#dataRecognize.to_csv(os.path.join(sys.path[0], 'outputs','facial_data_recognize.csv'), index=None)

race = dataTags[dataTags['name'] == 'race'].iat[0,1]
print(race)
compliment = sendCompliments(race)
print(compliment)
