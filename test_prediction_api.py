import requests
import json

with open("payload.json") as f:
    payload = json.load(f)

response = requests.post("http://localhost:1234/invocations", json=payload)
print(response.json())