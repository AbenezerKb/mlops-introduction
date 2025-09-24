import requests
import json

url = 'http://127.0.0.1:5000/invocations'
headers = {
    "Content-Type": "application/json"
}

data = {
    "dataframe_split": {
        "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "data": [[5.1, 3.5, 1.4, 0.2]]
    }
}

response = requests.post(url, headers=headers,json=data)

print("Status Code:", response.status_code)
print("Response Body:", response.text)