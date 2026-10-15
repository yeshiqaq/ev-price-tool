import requests

url = "http://127.0.0.1:8000/health"

response = requests.get(url, timeout=10)

print(response.status_code)
print(response.json())