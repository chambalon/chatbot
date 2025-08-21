import requests
import json


URL = "http://127.0.0.1:5500/chatbot" 

payload = {
    "message": "Hello"
}

headers = {
    "Content-Type": "application/json"
}

requests.post(URL, data=json.dumps(payload), headers=headers)


'''
try:
   
    response = requests.post(URL, data=json.dumps(payload), headers=headers)

   
    if response.status_code == 200:
        print("Request successful!")
        print("Response JSON:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"Request failed with status code: {response.status_code}")
        print("Response text:")
        print(response.text)

except requests.exceptions.ConnectionError as e:
    print(f"Error connecting to the Flask app: {e}")
    print("Please ensure your Flask application is running at the specified URL.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
'''