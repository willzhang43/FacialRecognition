#On a quest to build the stupidest application ever devised.
import numpy as np
import cv2, os, sys
import requests
import json
import base64
import pandas as pd
api_shit = {
            "api_key": "d45fd466-51e2-4701-8da8-04351c872236",
            "file_uri": 'http://betafaceapi.com/api_examples/sample.png',
            "detection_flags": "classifiers,content, extended",
            "recognize_targets": ["all@mynamespace"]
            }
jsonData = json.dumps(api_shit)
newHeaders = {'Content-type': 'application/json', 'Accept': 'application/json'}
response = requests.post('https://www.betafaceapi.com/api/v2/media', headers = newHeaders, data=jsonData)
print("Status code: ", response.status_code)
response_Json = response.json()
print(response_Json)
with open('facial_data_test.json', 'w') as f:
    json.dump(response_Json, f)

data_tags = pd.DataFrame(response_Json['media']['faces'][0]['tags'])
data_tags.to_csv(os.path.join(sys.path[0], 'facial_data_tags.csv'), index=None)
data_recognize = pd.DataFrame(response_Json['recognize']['results'][0]['matches'])
data_recognize.to_csv(os.path.join(sys.path[0], 'facial_data_recognize.csv'), index=None)
race = data_tags[data_tags['name'] == 'race'].iat[0,1]
if(race.lower()=='asian'):
    print('Make me an iPhone.')
elif(race.lower()=='white'):
    print('Go back to Starbucks.')
else:
    print('I respect all racial backgrounds. Have a good day!')