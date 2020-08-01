import base64
import requests
import sys, os
import json
import pandas as pd


""" with open(os.path.join(sys.path[0],'photo_raw.jpg'), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
text_file = open("image.txt", "w")
n = text_file.write(encoded_string.decode('utf-8'))
text_file.close() """

with open(os.path.join(sys.path[0], 'facial_data.json')) as data_file:
    data = json.load(data_file)


data_tags = pd.DataFrame(data['media']['faces'][0]['tags'])
data_tags.to_csv(os.path.join(sys.path[0], 'facial_data_tags.csv'), index=None)

data_recognize = pd.DataFrame(data['recognize']['results'][0]['matches'])
data_recognize.to_csv(os.path.join(sys.path[0], 'facial_data_recognize.csv'), index=None)

race = data_tags[data_tags['name'] == 'race'].iat[0,1]

if(race.lower()=='asian'):
    print('Make me an iPhone.')
else:
    print('I respect all racial backgrounds. Have a good day!')

print(data_recognize.iat[0,0])

api_shit = {
            "api_key": "d45fd466-51e2-4701-8da8-04351c872236",
            "faces_uuids": [data_recognize.iat[0,0]],
            "action": 0,
            "parameters": ""
            }
jsonData = json.dumps(api_shit)
newHeaders = {'Content-type': 'application/json', 'Accept': 'application/json'}
response = requests.post('https://www.betafaceapi.com/api/v2/transform', headers = newHeaders, data=jsonData)
print("Status code: ", response.status_code)
response_Json = response.json()
print(response_Json)
with open(os.path.join(sys.path[0], 'facial_transform.json'), 'w') as f:
    json.dump(response_Json, f)
g = open(os.path.join(sys.path[0], "transformed.jpg"), "wb")
g.write(base64.b64decode(response_Json['image_base64']))
g.close()

