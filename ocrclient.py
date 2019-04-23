import requests
import json
import os


url = 'http://127.0.0.1:5000/ocrjson'
payload = {"minConfidence": 0.5, "width": 320, "height": 320, "padding": 0, "language":"eng", "oem":1, "psm": 7}
file='/home/lukas/Desktop/example_03.jpg'
files = {
     'json': (None, json.dumps(payload), 'application/json'),
     'image': (os.path.basename(file), open(file, 'rb'), 'application/octet-stream')
}

r = requests.post(url, files=files)
print(r.content)