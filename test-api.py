import requests

url = "http://localhost:1234/invocations"
headers = {"Content-Type": "application/json"}

with open("payload.json", "rb") as f:
    data = f.read()

response = requests.post(url, headers=headers, data=data)

print("Response status code:", response.status_code)
print("Response JSON:", response.json())